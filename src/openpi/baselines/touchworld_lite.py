"""A compact, task-ID-conditioned TouchWorld-inspired adaptation for XHand.

This is an independent PyTorch baseline, not the authors' TouchWorld model.
It omits the semantic planner, Wan video backbone, and human-data pretraining.
It retains future visual/tactile goal prediction, goal-conditioned nominal
actions, and a separate residual controller using newly observed touch.

Training has three explicit stages:
1. ``representation_loss`` learns observation autoencoders by reconstruction.
2. ``freeze_observation_encoders`` fixes the latent coordinate system;
   ``nominal_loss`` supervises both future goals and nominal actions. The action
   decoder ALWAYS receives predicted goals, including during training.
3. ``freeze_nominal`` fixes the entire nominal policy. Generate a real plan
   with ``predict_nominal`` at the chunk origin, then train ``residual_loss``
   using tactile history and state observed at the sampled execution offset.

Data contract (all observations are time-aligned by the caller):
* RGB: float [B,3,height,width], scaled to [-1,1].
* Touch history: float [B,T,5,120,3], oldest first, ending at the current time.
* State: float [B,18]; actions: float [B,H,18], in the SAME normalized action
  coordinate system for nominal prediction, residual targets and deployment.
* task_id: long [B], REQUIRED, using the saved training task-to-ID mapping.
  This model does not encode free-form language or generalize to unseen IDs.
* Future supervision: one RGB [B,3,height,width] and tactile [B,5,120,3]
  observation at the fixed t+H goal time; future_valid_mask is bool [B].
* action_valid_mask: bool [B,H], false for episode padding. Do not construct a
  valid goal by repeating the last episode frame.
* Offset m means m commands have executed. At refinement only histories ending
  at t+m and the state at t+m may be supplied. Initial-plan RGB/context stay
  fixed. No inference API accepts future ground-truth observations or actions.

Fit state/action/tactile normalization on TRAINING episodes only, save it with
the checkpoint, and apply the same transform at inference. These encoders do
not establish physical force units. Split episodes before constructing windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TouchWorldLiteConfig:
    num_tasks: int
    action_horizon: int = 16
    window_size: int = 16
    state_dim: int = 18
    action_dim: int = 18
    history_frames: int = 10
    num_fingers: int = 5
    taxels_per_finger: int = 120
    force_dim: int = 3
    width: int = 256
    latent_dim: int = 128
    image_grid: int = 4
    num_heads: int = 8
    goal_layers: int = 2
    nominal_layers: int = 4
    residual_layers: int = 4
    residual_action_indices: tuple[int, ...] = tuple(range(6, 18))
    visual_goal_weight: float = 1.0
    tactile_goal_weight: float = 1.0
    action_loss_weight: float = 1.0
    residual_regularization: float = 1e-4

    def __post_init__(self):
        for name in (
            "num_tasks", "action_horizon", "window_size", "state_dim", "action_dim",
            "history_frames", "num_fingers", "taxels_per_finger", "force_dim",
            "width", "latent_dim", "image_grid", "num_heads", "goal_layers",
            "nominal_layers", "residual_layers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        if self.window_size > self.action_horizon:
            raise ValueError("window_size must not exceed action_horizon")
        if not self.residual_action_indices:
            raise ValueError("residual_action_indices cannot be empty")
        if len(set(self.residual_action_indices)) != len(self.residual_action_indices):
            raise ValueError("residual_action_indices must be unique")
        if any(i < 0 or i >= self.action_dim for i in self.residual_action_indices):
            raise ValueError("residual action index outside action_dim")
        if self.visual_goal_weight <= 0 or self.tactile_goal_weight <= 0:
            raise ValueError("Both future observation objectives must have positive weight")
        if self.action_loss_weight <= 0 or self.residual_regularization < 0:
            raise ValueError("Invalid action/residual loss weight")


@dataclass(frozen=True)
class NominalPlan:
    """Cache a *predicted* initial plan; never substitute demonstration actions.

    ``actions`` is [B,H,A], ``context`` is [B,C,width], and goal tensors are
    [B,image_grid**2,latent_dim] and [B,num_fingers,latent_dim]. Cache only within
    this chunk, retain the original nominal actions across all refinements, and
    discard the plan at an episode/chunk boundary.
    """

    actions: Tensor
    context: Tensor
    predicted_visual_goal: Tensor
    predicted_tactile_goal: Tensor


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Zero invalid entries before arithmetic, including NaN episode padding."""
    mask = torch.broadcast_to(mask, prediction.shape)
    prediction = torch.where(mask, prediction.float(), 0.0)
    target = torch.where(mask, target.detach().float(), 0.0)
    return (prediction - target).square().sum() / mask.sum().clamp_min(1)


class _ObservationAutoencoders(nn.Module):
    def __init__(self, cfg: TouchWorldLiteConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((cfg.image_grid, cfg.image_grid)),
            nn.Conv2d(128, cfg.latent_dim, 1),
        )
        self.vision_decoder = nn.Sequential(
            nn.ConvTranspose2d(cfg.latent_dim, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Tanh(),
        )
        force_width = cfg.taxels_per_finger * cfg.force_dim
        self.tactile_encoder = nn.Sequential(
            nn.Linear(force_width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.latent_dim)
        )
        self.tactile_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.width), nn.SiLU(), nn.Linear(cfg.width, force_width)
        )

    def encode_rgb(self, rgb: Tensor) -> Tensor:
        return self.vision_encoder(rgb).flatten(2).transpose(1, 2)

    def encode_touch(self, tactile: Tensor) -> Tensor:
        return self.tactile_encoder(tactile.flatten(-2))

    def reconstruction_loss(self, rgb: Tensor, tactile: Tensor) -> dict[str, Tensor]:
        visual_latent = self.vision_encoder(rgb)
        visual_reconstruction = F.interpolate(
            self.vision_decoder(visual_latent), size=rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        tactile_reconstruction = self.tactile_decoder(self.encode_touch(tactile)).reshape_as(tactile)
        visual_loss = F.mse_loss(visual_reconstruction.float(), rgb.detach().float())
        tactile_loss = F.mse_loss(tactile_reconstruction.float(), tactile.detach().float())
        return {
            "loss": visual_loss + tactile_loss,
            "vision_reconstruction": visual_loss,
            "tactile_reconstruction": tactile_loss,
        }


def _decoder(cfg: TouchWorldLiteConfig, layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=cfg.width, nhead=cfg.num_heads, dim_feedforward=4 * cfg.width,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(cfg.width))


class TouchWorldLite(nn.Module):
    """Use ``forward(stage=..., **batch)`` for DDP, or the named loss methods.

    Stages are ``representation``, ``nominal``, and ``residual``. Switch stage
    BEFORE constructing its optimizer/DDP wrapper. Stage and trainability are
    restored by ``load_state_dict``. Autoencoder checkpoints must be learned on
    the training split before freezing; this class cannot verify that training
    actually occurred, and freezing random encoders is not a valid experiment.
    """

    def __init__(self, cfg: TouchWorldLiteConfig):
        super().__init__()
        self.config = cfg
        self.observation_ae = _ObservationAutoencoders(cfg)
        self.task_embedding = nn.Embedding(cfg.num_tasks, cfg.width)
        self.state_projection = nn.Linear(cfg.state_dim, cfg.width)
        self.visual_projection = nn.Linear(cfg.latent_dim, cfg.width)
        self.tactile_projection = nn.Linear(cfg.latent_dim, cfg.width)
        self.visual_positions = nn.Parameter(torch.randn(cfg.image_grid**2, cfg.width) * 0.02)
        self.finger_positions = nn.Parameter(torch.randn(cfg.num_fingers, cfg.width) * 0.02)
        self.history_positions = nn.Parameter(torch.randn(cfg.history_frames, cfg.width) * 0.02)
        self.goal_queries = nn.Parameter(torch.randn(cfg.image_grid**2 + cfg.num_fingers, cfg.width) * 0.02)
        self.goal_predictor = _decoder(cfg, cfg.goal_layers)
        self.visual_goal_head = nn.Linear(cfg.width, cfg.latent_dim)
        self.tactile_goal_head = nn.Linear(cfg.width, cfg.latent_dim)
        self.goal_visual_projection = nn.Linear(cfg.latent_dim, cfg.width)
        self.goal_tactile_projection = nn.Linear(cfg.latent_dim, cfg.width)
        self.action_queries = nn.Parameter(torch.randn(cfg.action_horizon, cfg.width) * 0.02)
        self.nominal_decoder = _decoder(cfg, cfg.nominal_layers)
        self.nominal_action_head = nn.Linear(cfg.width, cfg.action_dim)

        self.residual_tactile_projection = nn.Linear(cfg.latent_dim, cfg.width)
        self.residual_state_projection = nn.Linear(cfg.state_dim, cfg.width)
        self.residual_action_projection = nn.Linear(cfg.action_dim, cfg.width)
        self.residual_history_positions = nn.Parameter(torch.randn(cfg.history_frames, cfg.width) * 0.02)
        self.residual_finger_positions = nn.Parameter(torch.randn(cfg.num_fingers, cfg.width) * 0.02)
        self.residual_window_positions = nn.Parameter(torch.randn(cfg.window_size, cfg.width) * 0.02)
        self.residual_offset_embedding = nn.Embedding(cfg.action_horizon, cfg.width)
        self.residual_decoder = _decoder(cfg, cfg.residual_layers)
        self.residual_head = nn.Linear(cfg.width, len(cfg.residual_action_indices))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer("_training_stage", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_residual_indices", torch.tensor(cfg.residual_action_indices, dtype=torch.long))
        self._configure_trainability()

    @property
    def training_stage(self) -> str:
        return ("representation", "nominal", "residual")[int(self._training_stage)]

    def _configure_trainability(self):
        stage = int(self._training_stage)
        for name, parameter in self.named_parameters():
            if stage == 0:
                trainable = name.startswith("observation_ae.")
            elif stage == 1:
                trainable = not name.startswith(("observation_ae.", "residual_"))
            else:
                trainable = name.startswith("residual_")
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None
        if stage > 0:
            self.observation_ae.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if int(self._training_stage) > 0:
            self.observation_ae.eval()
        if int(self._training_stage) == 2:
            self.goal_predictor.eval()
            self.nominal_decoder.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._configure_trainability()
        self.train(self.training)
        return result

    def freeze_observation_encoders(self):
        self._training_stage.fill_(1)
        self._configure_trainability()
        self.train(self.training)
        return self

    def freeze_nominal(self):
        if int(self._training_stage) == 0:
            raise RuntimeError("Train/freeze observation representations before the nominal policy")
        self._training_stage.fill_(2)
        self._configure_trainability()
        self.train(self.training)
        return self

    def forward(self, *, stage: str, **kwargs) -> dict[str, Tensor]:
        if stage != self.training_stage:
            raise RuntimeError(f"Model is in {self.training_stage!r} stage, not {stage!r}")
        return {
            "representation": self.representation_loss,
            "nominal": self.nominal_loss,
            "residual": self.residual_loss,
        }[stage](**kwargs)

    def _check_rgb(self, rgb: Tensor):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("RGB must be floating [B,3,height,width] in [-1,1]")

    def _check_touch(self, tactile: Tensor):
        cfg = self.config
        expected = (cfg.history_frames, cfg.num_fingers, cfg.taxels_per_finger, cfg.force_dim)
        if tactile.ndim != 5 or tuple(tactile.shape[1:]) != expected or not tactile.is_floating_point():
            raise ValueError(f"Touch history must be floating [B,{expected}]")

    def _check_current(self, current_rgb, current_tactile, current_state, task_id):
        self._check_rgb(current_rgb)
        self._check_touch(current_tactile)
        batch = current_rgb.shape[0]
        if current_tactile.shape[0] != batch or current_state.shape != (batch, self.config.state_dim):
            raise ValueError("Current RGB, tactile and state batch/dimensions disagree")
        if task_id.shape != (batch,) or task_id.dtype != torch.long:
            raise ValueError("task_id must be long [B] from the saved task-ID mapping")

    @staticmethod
    def _check_mask(mask, shape, name):
        if mask.shape != shape or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be bool {shape}")

    def representation_loss(self, rgb: Tensor, tactile: Tensor) -> dict[str, Tensor]:
        if int(self._training_stage) != 0:
            raise RuntimeError("Observation autoencoders are already frozen")
        self._check_rgb(rgb)
        self._check_touch(tactile)
        if rgb.shape[0] != tactile.shape[0]:
            raise ValueError("RGB and tactile batch sizes disagree")
        return self.observation_ae.reconstruction_loss(rgb, tactile)

    def _predict_nominal_impl(self, *, current_rgb, current_tactile, current_state, task_id) -> NominalPlan:
        if int(self._training_stage) == 0:
            raise RuntimeError("Freeze learned observation encoders before nominal prediction")
        self._check_current(current_rgb, current_tactile, current_state, task_id)
        batch = current_rgb.shape[0]
        with torch.no_grad():
            visual = self.observation_ae.encode_rgb(current_rgb)
            tactile = self.observation_ae.encode_touch(current_tactile)
        visual_tokens = self.visual_projection(visual) + self.visual_positions[None]
        tactile_tokens = (
            self.tactile_projection(tactile)
            + self.history_positions[None, :, None]
            + self.finger_positions[None, None]
        ).flatten(1, 2)
        current_context = torch.cat(
            [self.task_embedding(task_id)[:, None], self.state_projection(current_state)[:, None],
             visual_tokens, tactile_tokens], dim=1,
        )
        goal_hidden = self.goal_predictor(self.goal_queries[None].expand(batch, -1, -1), current_context)
        vision_count = self.config.image_grid**2
        visual_goal = self.visual_goal_head(goal_hidden[:, :vision_count])
        tactile_goal = self.tactile_goal_head(goal_hidden[:, vision_count:])
        # Ground-truth future encodings are NEVER inserted into this context.
        context = torch.cat(
            [current_context, self.goal_visual_projection(visual_goal) + self.visual_positions[None],
             self.goal_tactile_projection(tactile_goal) + self.finger_positions[None]], dim=1,
        )
        actions = self.nominal_action_head(
            self.nominal_decoder(self.action_queries[None].expand(batch, -1, -1), context)
        )
        return NominalPlan(actions, context, visual_goal, tactile_goal)

    @torch.no_grad()
    def predict_nominal(self, *, current_rgb, current_tactile, current_state, task_id) -> NominalPlan:
        """Plan from observations at t; no future labels accepted by this API."""
        return self._predict_nominal_impl(
            current_rgb=current_rgb, current_tactile=current_tactile,
            current_state=current_state, task_id=task_id,
        )

    def nominal_loss(
        self, *, current_rgb: Tensor, current_tactile: Tensor, current_state: Tensor,
        task_id: Tensor, actions: Tensor, future_rgb: Tensor, future_tactile: Tensor,
        future_valid_mask: Tensor, action_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if int(self._training_stage) != 1:
            raise RuntimeError("nominal_loss requires the nominal training stage")
        plan = self._predict_nominal_impl(
            current_rgb=current_rgb, current_tactile=current_tactile,
            current_state=current_state, task_id=task_id,
        )
        batch = current_rgb.shape[0]
        cfg = self.config
        if actions.shape != plan.actions.shape or future_rgb.shape != current_rgb.shape:
            raise ValueError("Action/terminal RGB target shape mismatch")
        if future_tactile.shape != (batch, cfg.num_fingers, cfg.taxels_per_finger, cfg.force_dim):
            raise ValueError("future_tactile must be the single terminal goal frame [B,F,P,3]")
        self._check_mask(future_valid_mask, (batch,), "future_valid_mask")
        self._check_mask(action_valid_mask, (batch, cfg.action_horizon), "action_valid_mask")
        # Sanitize invalid padding BEFORE the frozen target encoders see it.
        with torch.no_grad():
            visual_target = self.observation_ae.encode_rgb(
                torch.where(future_valid_mask[:, None, None, None], future_rgb, 0.0)
            )
            tactile_target = self.observation_ae.encode_touch(
                torch.where(future_valid_mask[:, None, None, None], future_tactile, 0.0)
            )
        visual_loss = _masked_mse(plan.predicted_visual_goal, visual_target, future_valid_mask[:, None, None])
        tactile_loss = _masked_mse(plan.predicted_tactile_goal, tactile_target, future_valid_mask[:, None, None])
        action_loss = _masked_mse(plan.actions, actions, action_valid_mask[:, :, None])
        return {
            "loss": cfg.visual_goal_weight * visual_loss + cfg.tactile_goal_weight * tactile_loss
                    + cfg.action_loss_weight * action_loss,
            "visual_goal_loss": visual_loss, "tactile_goal_loss": tactile_loss,
            "action_loss": action_loss,
            "valid_goals": future_valid_mask.sum(), "valid_actions": action_valid_mask.sum(),
        }

    def _window(self, values: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        batch = values.shape[0]
        if offsets.shape != (batch,) or offsets.dtype != torch.long:
            raise ValueError("offsets must be long [B]")
        if torch.any(offsets < 0) or torch.any(offsets >= cfg.action_horizon):
            raise ValueError("offsets must be in [0, action_horizon)")
        indices = offsets[:, None] + torch.arange(cfg.window_size, device=offsets.device)[None]
        valid = indices < cfg.action_horizon
        gather = indices.clamp_max(cfg.action_horizon - 1)
        rows = torch.arange(batch, device=offsets.device)[:, None]
        return values[rows, gather], valid

    def _residual_prediction(self, *, plan, current_tactile, current_state, offsets):
        if int(self._training_stage) != 2:
            raise RuntimeError("Freeze the nominal policy before residual training/inference")
        self._check_touch(current_tactile)
        cfg = self.config
        batch = current_tactile.shape[0]
        if plan.actions.shape != (batch, cfg.action_horizon, cfg.action_dim):
            raise ValueError("Nominal plan action shape mismatch")
        if plan.context.ndim != 3 or plan.context.shape[0] != batch or plan.context.shape[2] != cfg.width:
            raise ValueError("Nominal plan context shape mismatch")
        if current_state.shape != (batch, cfg.state_dim):
            raise ValueError("Current state shape mismatch")
        base_window, valid = self._window(plan.actions.detach(), offsets)
        with torch.no_grad():
            tactile = self.observation_ae.encode_touch(current_tactile)
        fresh_tokens = (
            self.residual_tactile_projection(tactile)
            + self.residual_history_positions[None, :, None]
            + self.residual_finger_positions[None, None]
        ).flatten(1, 2)
        context = torch.cat(
            [plan.context.detach(), fresh_tokens, self.residual_state_projection(current_state.detach())[:, None]], dim=1
        )
        queries = (
            self.residual_action_projection(base_window) + self.residual_window_positions[None]
            + self.residual_offset_embedding(offsets)[:, None]
        )
        hidden = self.residual_decoder(queries, context, tgt_key_padding_mask=~valid)
        selected_delta = self.residual_head(hidden)
        selected_delta = torch.where(valid[:, :, None], selected_delta, 0.0)
        delta = torch.zeros_like(base_window).index_copy(-1, self._residual_indices, selected_delta)
        return base_window + delta, selected_delta, valid

    def residual_loss(
        self, *, plan: NominalPlan, current_tactile: Tensor, current_state: Tensor,
        offsets: Tensor, target_actions: Tensor, action_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Supervise corrections to actual frozen-model predictions, not noisy GT.

        ``plan`` must come from ``predict_nominal`` at the chunk origin. Reuse it
        at every sampled offset. ``target_actions`` stays in that original
        chunk's reference frame, including when latest state changes.
        """
        if target_actions.shape != plan.actions.shape:
            raise ValueError("target_actions must cover the original nominal chunk")
        self._check_mask(action_valid_mask, plan.actions.shape[:2], "action_valid_mask")
        corrected, delta, in_chunk = self._residual_prediction(
            plan=plan, current_tactile=current_tactile, current_state=current_state, offsets=offsets
        )
        target_window, _ = self._window(target_actions, offsets)
        valid_window, _ = self._window(action_valid_mask, offsets)
        valid = in_chunk & valid_window
        selected_prediction = corrected.index_select(-1, self._residual_indices)
        selected_target = target_window.index_select(-1, self._residual_indices)
        residual_loss = _masked_mse(selected_prediction, selected_target, valid[:, :, None])
        regularization = _masked_mse(delta, torch.zeros_like(delta), valid[:, :, None])
        return {
            "loss": residual_loss + self.config.residual_regularization * regularization,
            "residual_loss": residual_loss, "residual_regularization": regularization,
            "valid_actions": valid.sum(),
        }

    @torch.no_grad()
    def refine(
        self, *, plan: NominalPlan, current_tactile: Tensor, current_state: Tensor, offsets: Tensor,
    ) -> Tensor:
        """Return corrected [B,W,A] lookahead windows from each supplied offset.

        Only min(W,H-offset) entries belong to the chunk. A controller commits
        its chosen C entries (e.g. C=4), accounting for commands already sent
        during inference. Keep the original plan for subsequent calls: never
        accumulate residuals onto an already-corrected nominal plan.
        """
        corrected, _, _ = self._residual_prediction(
            plan=plan, current_tactile=current_tactile, current_state=current_state, offsets=offsets
        )
        return corrected
