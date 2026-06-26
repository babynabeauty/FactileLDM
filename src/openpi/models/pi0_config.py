import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import model_tavla as _model_tavla
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi.shared.effort_type import EffortType

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    
    effort_type: str | None = None
    effort_dim: int | None = None
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_effort_input: bool = None  # type: ignore

    # Inject the current tactile observation into the action-expert suffix.
    use_tactile_observation: bool = False
    tactile_num_fingers: int = 5
    tactile_dim_per_finger: int = 3
    
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.use_tactile_observation and self.pi05:
            raise ValueError("Current tactile observation tokens are only supported by Pi0, not Pi0.5.")
        if self.tactile_num_fingers <= 0 or self.tactile_dim_per_finger <= 0:
            raise ValueError("Tactile finger count and per-finger dimension must be positive.")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tactile=None,
                effort=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.tactile_num_fingers, self.tactile_dim_per_finger],
                        jnp.float32,
                    )
                    if self.use_tactile_observation
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0TaVLAConfig(Pi0Config):
    
    force_fast_tokenizer_path: str | None = None
    force_fast_vocab_size: int = 4096
    force_fast_action_dim: int = 8
    force_data_path: str | None = None # Path to force data (numpy array) for FAST tokenizer training
    force_data_repo_id: str | None = None # LeRobot repo id for force data for FAST tokenizer training

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_tavla import Pi0TaVLA
        return Pi0TaVLA(self, rngs=nnx.Rngs(rng))

    @override
    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.effort_type in {EffortType.LLM_HIS_Lang, }:
            object.__setattr__(self, "max_token_len", 300 if self.pi05 else 48)
            object.__setattr__(self, "discrete_effort_input", True)


@dataclasses.dataclass(frozen=True)
class Pi0SeerConfig(Pi0TaVLAConfig):
    foreseen_token_count_per_view: int = 10
    future_image_views: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb")
    future_rgb_step: int = 0
    image_decoder_patch_size: int = 16
    image_decoder_input_size: int = 224
    future_image_loss_weight: float = 0.1
    predict_future_image_during_inference: bool = False
    use_future_rgb_instead_of_flow: bool = True
    normalize_future_patch_targets: bool = True

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_seer import Pi0Seer

        return Pi0Seer(self, rngs=nnx.Rngs(rng))

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.foreseen_token_count_per_view <= 0:
            raise ValueError("foreseen_token_count_per_view must be positive.")
        if self.image_decoder_patch_size <= 0:
            raise ValueError("image_decoder_patch_size must be positive.")
        if self.image_decoder_input_size % self.image_decoder_patch_size != 0:
            raise ValueError(
                "image_decoder_input_size must be divisible by image_decoder_patch_size, "
                f"got {self.image_decoder_input_size} and {self.image_decoder_patch_size}."
            )
        if not 0 <= self.future_rgb_step <= self.action_horizon:
            raise ValueError(
                f"future_rgb_step must satisfy 0 <= future_rgb_step <= action_horizon={self.action_horizon}, "
                f"got {self.future_rgb_step}."
            )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model_tavla.Observation, _model_tavla.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        effort_dim = self.effort_dim if self.effort_dim is not None else self.action_dim
        effort_dim_in = getattr(self, "effort_dim_in", effort_dim)
        effort_steps = max(1, effort_dim_in // max(effort_dim, 1))

        with at.disable_typechecking():
            observation_spec = _model_tavla.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                effort=jax.ShapeDtypeStruct([batch_size, effort_steps, effort_dim], jnp.float32),
                future_rgb_img=image_spec,
                future_wrist_rgb_img=image_spec,
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec




@dataclasses.dataclass(frozen=True)
class Pi0LatentFlowConfig(Pi0Config):
    force_input_frames: int = 10
    teacher_action_loss_weight: float = 1.0
    student_action_loss_weight: float = 1.0
    distill_layer_indices: tuple = (8,12,16)
    future_force_align_loss_weight: float = 0.1
    future_flow_align_loss_weight: float = 0.1
    distill_projector_hidden_dim: int | None = None
    flow_token_count: int = 16
    flow_vae_name: str = "stabilityai/sdxl-vae"
    flow_vae_latent_channels: int = 4
    use_future_flow: bool = True
    future_flow_source: Literal["image", "scene_flow"] = "image"
    scene_flow_input_dim: int = 10
    use_future_rgb_instead_of_flow: bool = False
    future_rgb_step: int = 0
    student_future_query_noise_scale_max: float = 0
    student_future_query_noise_start_ratio: float = 0
    student_future_query_noise_end_ratio: float = 0
    structured_tactile: bool = False
    tactile_history_offsets: tuple[int, ...] = (-18, -16, -14, -12, -10, -8, -6, -4, -2, 0)
    future_tactile_segments: int = 8
    future_steps_per_segment: int = 4
    tactile_tokenizer_dim: int = 256
    tactile_points_per_finger: int = 1
    tactile_raw_contact_top_k: int = 16
    tactile_raw_contact_threshold: float = 1.0
    tactile_raw_contact_temperature: float = 5.0
    tactile_patch_tokenizer: bool = False
    tactile_patch_fingers: tuple[int, ...] = (0, 1, 2)
    tactile_num_patches: int = 5
    future_tactile_align_layer: int = 12
    tactile_sample_hz: float = 15.0
    arm_hand_mask_attention: bool = False
    arm_action_dim: int = 6
    hand_action_dim: int = 12
    tactile_refiner_enabled: bool = False
    tactile_refiner_layers: int = 2
    tactile_refiner_width: int = 256
    tactile_refiner_heads: int = 4
    tactile_refiner_mlp_dim: int = 1024
    hand_synergy_dim: int = 4
    hand_synergy_loss_weight: float = 0.03
    tactile_refiner_delta_loss_weight: float = 0.0001
    tactile_refiner_gate_bias: float = -2.0
    tactile_refiner_delta_scale: float = 0.1

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.structured_tactile:
            if self.tactile_points_per_finger <= 0:
                raise ValueError("tactile_points_per_finger must be positive.")
            expected_effort_dim = (
                self.tactile_num_fingers
                * self.tactile_points_per_finger
                * self.tactile_dim_per_finger
            )
            if self.effort_dim != expected_effort_dim:
                raise ValueError(
                    "Structured tactile effort_dim must equal "
                    "tactile_num_fingers * tactile_points_per_finger * tactile_dim_per_finger; "
                    f"got effort_dim={self.effort_dim}, expected={expected_effort_dim}."
                )
            if len(self.tactile_history_offsets) != self.force_input_frames:
                raise ValueError("tactile_history_offsets length must equal force_input_frames.")
            if self.future_tactile_segments * self.future_steps_per_segment != self.action_horizon:
                raise ValueError("Future tactile segments must exactly cover action_horizon.")
            if self.tactile_sample_hz <= 0:
                raise ValueError("tactile_sample_hz must be positive.")
            if self.tactile_raw_contact_top_k < 0:
                raise ValueError("tactile_raw_contact_top_k must be non-negative.")
            if self.tactile_raw_contact_temperature <= 0:
                raise ValueError("tactile_raw_contact_temperature must be positive.")
            if self.tactile_patch_tokenizer:
                if self.tactile_points_per_finger <= 1:
                    raise ValueError("tactile_patch_tokenizer requires raw tactile points.")
                if self.tactile_num_patches <= 0:
                    raise ValueError("tactile_num_patches must be positive.")
                if any(finger < 0 or finger >= self.tactile_num_fingers for finger in self.tactile_patch_fingers):
                    raise ValueError(
                        f"tactile_patch_fingers must be in [0, {self.tactile_num_fingers}), "
                        f"got {self.tactile_patch_fingers}."
                    )
            object.__setattr__(self, "distill_layer_indices", (self.future_tactile_align_layer,))
        if self.arm_hand_mask_attention:
            if not self.structured_tactile:
                raise ValueError("arm_hand_mask_attention currently requires structured_tactile=True.")
            if self.arm_action_dim <= 0 or self.hand_action_dim <= 0:
                raise ValueError("arm_action_dim and hand_action_dim must be positive.")
            if self.arm_action_dim + self.hand_action_dim > self.action_dim:
                raise ValueError("Arm and hand action slices exceed action_dim.")
        if self.tactile_refiner_enabled:
            if not self.arm_hand_mask_attention:
                raise ValueError("tactile_refiner_enabled requires arm_hand_mask_attention=True.")
            if self.tactile_refiner_layers <= 0:
                raise ValueError("tactile_refiner_layers must be positive.")
            if self.tactile_refiner_width <= 0 or self.tactile_refiner_mlp_dim <= 0:
                raise ValueError("tactile_refiner_width and tactile_refiner_mlp_dim must be positive.")
            if self.tactile_refiner_width % self.tactile_refiner_heads != 0:
                raise ValueError("tactile_refiner_width must be divisible by tactile_refiner_heads.")
            if self.hand_synergy_dim <= 0:
                raise ValueError("hand_synergy_dim must be positive.")
            if self.hand_synergy_loss_weight < 0 or self.tactile_refiner_delta_loss_weight < 0:
                raise ValueError("Refiner loss weights must be non-negative.")
            if self.tactile_refiner_delta_scale < 0:
                raise ValueError("tactile_refiner_delta_scale must be non-negative.")
        if not 0 <= self.future_rgb_step <= self.action_horizon:
            raise ValueError(
                f"future_rgb_step must satisfy 0 <= future_rgb_step <= action_horizon={self.action_horizon}, "
                f"got {self.future_rgb_step}."
            )
        if self.use_future_flow and self.flow_token_count <= 0:
            raise ValueError("flow_token_count must be positive when use_future_flow=True.")
        if self.future_flow_source not in ("image", "scene_flow"):
            raise ValueError(f"Unsupported future_flow_source={self.future_flow_source!r}.")
        if self.scene_flow_input_dim <= 0:
            raise ValueError("scene_flow_input_dim must be positive.")
        if self.student_future_query_noise_scale_max < 0.0:
            raise ValueError(
                "student_future_query_noise_scale_max must be non-negative, "
                f"got {self.student_future_query_noise_scale_max}."
            )
        if not 0.0 <= self.student_future_query_noise_start_ratio <= 1.0:
            raise ValueError(
                "student_future_query_noise_start_ratio must satisfy 0.0 <= p <= 1.0, "
                f"got {self.student_future_query_noise_start_ratio}."
            )
        if not 0.0 <= self.student_future_query_noise_end_ratio <= 1.0:
            raise ValueError(
                "student_future_query_noise_end_ratio must satisfy 0.0 <= p <= 1.0, "
                f"got {self.student_future_query_noise_end_ratio}."
            )
        if self.student_future_query_noise_start_ratio > self.student_future_query_noise_end_ratio:
            raise ValueError(
                "student_future_query_noise_start_ratio must be <= student_future_query_noise_end_ratio, "
                f"got start={self.student_future_query_noise_start_ratio}, "
                f"end={self.student_future_query_noise_end_ratio}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0_latent_flow import Pi0LatentFlow
        return Pi0LatentFlow(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        effort_shape = (
            (
                [
                    batch_size,
                    self.force_input_frames + self.action_horizon,
                    self.tactile_num_fingers,
                    self.tactile_points_per_finger,
                    self.tactile_dim_per_finger,
                ]
                if self.tactile_points_per_finger > 1
                else [
                    batch_size,
                    self.force_input_frames + self.action_horizon,
                    self.tactile_num_fingers,
                    self.tactile_dim_per_finger,
                ]
            )
            if self.structured_tactile
            else [batch_size, self.force_input_frames + self.action_horizon, int(self.effort_dim)]
        )
        with at.disable_typechecking():
            observation_spec = _model_tavla.Observation(
                images={key: image_spec for key in _model.IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in _model.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                effort=jax.ShapeDtypeStruct(effort_shape, jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        return observation_spec, jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )


@dataclasses.dataclass(frozen=True)
class Pi0FutureTactileConfig(Pi0Config):
    effort_type: str | None = EffortType.MOT
    effort_dim: int | None = 15
    force_input_frames: int = 10
    tactile_history_offsets: tuple[int, ...] = (-18, -16, -14, -12, -10, -8, -6, -4, -2, 0)
    future_tactile_segments: int = 8
    future_steps_per_segment: int = 4
    tactile_tokenizer_dim: int = 256
    tactile_points_per_finger: int = 1
    tactile_raw_contact_top_k: int = 16
    tactile_raw_contact_threshold: float = 1.0
    tactile_raw_contact_temperature: float = 5.0
    future_tactile_align_layer: int = 12
    tactile_sample_hz: float = 15.0
    future_tactile_latent_loss_weight: float = 0.1
    future_force_loss_weight: float = 0.2
    future_force_delta_loss_weight: float = 0.05
    future_tactile_encoder_type: Literal["dexterous", "finger_role", "raw_spatial"] = "dexterous"
    future_tactile_encoder_depth: int = 2
    future_tactile_encoder_fusion_depth: int = 0
    future_tactile_encoder_num_heads: int = 8
    future_tactile_target_width: int = 512
    future_hand_action_loss_weight: float = 0.0
    hand_action_start: int = 6
    hand_action_dim: int = 12

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.pi05:
            raise ValueError("Structured future tactile queries currently support Pi0 only.")
        if len(self.tactile_history_offsets) != self.force_input_frames:
            raise ValueError("tactile_history_offsets length must equal force_input_frames.")
        if self.future_tactile_segments * self.future_steps_per_segment != self.action_horizon:
            raise ValueError("Future tactile segments must exactly cover action_horizon.")
        if self.tactile_points_per_finger <= 0:
            raise ValueError("tactile_points_per_finger must be positive.")
        expected_effort_dim = (
            self.tactile_num_fingers * self.tactile_points_per_finger * self.tactile_dim_per_finger
        )
        if self.effort_dim != expected_effort_dim:
            raise ValueError(
                "Structured tactile effort_dim must equal "
                "tactile_num_fingers * tactile_points_per_finger * tactile_dim_per_finger; "
                f"got effort_dim={self.effort_dim}, expected={expected_effort_dim}."
            )
        if self.tactile_raw_contact_top_k < 0:
            raise ValueError("tactile_raw_contact_top_k must be non-negative.")
        if self.tactile_raw_contact_temperature <= 0:
            raise ValueError("tactile_raw_contact_temperature must be positive.")
        if self.future_tactile_encoder_type not in ("dexterous", "finger_role", "raw_spatial"):
            raise ValueError(f"Unsupported future_tactile_encoder_type={self.future_tactile_encoder_type!r}.")
        if self.tactile_points_per_finger > 1 and self.future_tactile_encoder_type != "raw_spatial":
            raise ValueError("Raw tactile input requires future_tactile_encoder_type='raw_spatial'.")
        if self.tactile_points_per_finger == 1 and self.future_tactile_encoder_type == "raw_spatial":
            raise ValueError("raw_spatial requires tactile_points_per_finger > 1.")
        if self.future_tactile_target_width <= 0:
            raise ValueError("future_tactile_target_width must be positive.")
        if self.hand_action_start + self.hand_action_dim > self.action_dim:
            raise ValueError("Hand action slice exceeds action_dim.")

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi0_future_tactile import Pi0FutureTactile

        return Pi0FutureTactile(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model_tavla.Observation(
                images={key: image_spec for key in _model.IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in _model.IMAGE_KEYS},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                effort=jax.ShapeDtypeStruct(
                    (
                        [
                            batch_size,
                            self.force_input_frames + self.action_horizon,
                            self.tactile_num_fingers,
                            self.tactile_points_per_finger,
                            self.tactile_dim_per_finger,
                        ]
                        if self.tactile_points_per_finger > 1
                        else [
                            batch_size,
                            self.force_input_frames + self.action_horizon,
                            self.tactile_num_fingers,
                            self.tactile_dim_per_finger,
                        ]
                    ),
                    jnp.float32,
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        return observation_spec, jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )


@dataclasses.dataclass(frozen=True)
class FutureTactileEncoderPretrainConfig(_model.BaseModelConfig):
    """Stage-1 action-aware future tactile encoder pretraining."""

    action_horizon: int = 32
    action_dim: int = 32
    max_token_len: int = 48
    effort_type: str | None = EffortType.MOT
    state_dim: int = 18
    tactile_num_fingers: int = 5
    tactile_dim_per_finger: int = 3
    effort_dim: int | None = 15
    force_input_frames: int = 10
    tactile_history_offsets: tuple[int, ...] = (-18, -16, -14, -12, -10, -8, -6, -4, -2, 0)
    future_tactile_segments: int = 8
    future_steps_per_segment: int = 4
    tactile_sample_hz: float = 15.0
    tactile_tokenizer_dim: int = 256
    encoder_width: int = 512
    encoder_depth: int = 2
    encoder_fusion_depth: int = 0
    encoder_num_heads: int = 8
    hand_action_start: int = 6
    hand_action_dim: int = 12
    hand_delta_loss_weight: float = 0.05
    hand_head_type: Literal["mean_repeat", "finger_flatten_4step", "flow_dit"] = "mean_repeat"
    action_decoder_depth: int = 8
    action_decoder_num_heads: int = 8

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    def __post_init__(self):
        if len(self.tactile_history_offsets) != self.force_input_frames:
            raise ValueError("tactile_history_offsets length must equal force_input_frames.")
        if self.future_tactile_segments * self.future_steps_per_segment != self.action_horizon:
            raise ValueError("Future tactile segments must exactly cover action_horizon.")
        if self.effort_dim != self.tactile_num_fingers * self.tactile_dim_per_finger:
            raise ValueError("Stage-1 currently supports per-finger 3D resultant forces only.")
        if self.hand_action_start + self.hand_action_dim > self.action_dim:
            raise ValueError("Hand action slice exceeds action_dim.")
        if self.hand_head_type not in ("mean_repeat", "finger_flatten_4step", "flow_dit"):
            raise ValueError(f"Unsupported hand_head_type={self.hand_head_type!r}.")
        if self.action_decoder_depth <= 0:
            raise ValueError("action_decoder_depth must be positive.")

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.future_tactile_pretrain import FutureTactileEncoderPretrain

        return FutureTactileEncoderPretrain(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            observation_spec = _model_tavla.Observation(
                images={key: image_spec for key in _model.IMAGE_KEYS},
                image_masks={key: image_mask_spec for key in _model.IMAGE_KEYS},
                # PadStatesAndActions presents state at the model action width.
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                effort=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.force_input_frames + self.action_horizon,
                        self.tactile_num_fingers,
                        self.tactile_dim_per_finger,
                    ],
                    jnp.float32,
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        return observation_spec, jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
