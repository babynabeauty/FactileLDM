import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.tactile_tokenizer import FingerRoleFutureTactileQFormer


class FutureTactileEncoderPretrain(nnx.Module):
    """Stage-1 pretraining for an action-aware future tactile encoder.

    This model intentionally avoids the VLM/action expert. It teaches a small
    future tactile Q-Former to extract c_future tokens that are useful for
    predicting future hand actions.
    """

    def __init__(self, config: pi0_config.FutureTactileEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.dim_per_finger = int(config.tactile_dim_per_finger)
        self.future_segments = int(config.future_tactile_segments)
        self.steps_per_segment = int(config.future_steps_per_segment)
        self.hand_action_start = int(config.hand_action_start)
        self.hand_action_dim = int(config.hand_action_dim)
        self.delta_loss_weight = float(config.hand_delta_loss_weight)
        self.encoder_width = int(config.encoder_width)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.future_encoder = FingerRoleFutureTactileQFormer(
            output_dim=self.encoder_width,
            hidden_dim=config.tactile_tokenizer_dim,
            num_fingers=self.num_fingers,
            dim_per_finger=self.dim_per_finger,
            future_segments=self.future_segments,
            future_steps_per_segment=self.steps_per_segment,
            num_layers=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            rngs=rngs,
        )
        self.state_proj = nnx.Linear(config.state_dim, self.encoder_width, rngs=rngs)
        self.history_force_proj_in = nnx.Linear(self.dim_per_finger, config.tactile_tokenizer_dim, rngs=rngs)
        self.history_force_proj_out = nnx.Linear(config.tactile_tokenizer_dim, self.encoder_width, rngs=rngs)
        self.condition_norm = nnx.LayerNorm(num_features=self.encoder_width, rngs=rngs)
        self.hand_head_in = nnx.Linear(self.encoder_width, self.encoder_width, rngs=rngs)
        self.hand_head_out = nnx.Linear(self.encoder_width, self.hand_action_dim, rngs=rngs)

    def _split_effort(self, observation: _model.Observation, dtype):
        if observation.effort is None:
            raise ValueError("Future tactile encoder pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (self.force_input_frames + self.action_horizon, self.num_fingers, self.dim_per_finger)
        if effort.ndim != 4 or effort.shape[1:] != expected:
            raise ValueError(f"Expected structured effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _history_context(self, history_effort: jax.Array, state: jax.Array) -> jax.Array:
        history = nnx.swish(self.history_force_proj_in(history_effort))
        history = self.history_force_proj_out(history)
        history = jnp.mean(history, axis=(1, 2))
        state_context = self.state_proj(state)
        return self.condition_norm(history + state_context)

    def encode_future(self, future_effort: jax.Array) -> jax.Array:
        return self.future_encoder.encode_future(
            future_effort, jnp.asarray(self.future_times, dtype=jnp.float32)
        )

    def predict_hand_action(self, c_future: jax.Array, condition: jax.Array) -> jax.Array:
        segment_tokens = einops.rearrange(
            c_future,
            "b (s f) d -> b s f d",
            s=self.future_segments,
            f=self.num_fingers,
        )
        segment_tokens = jnp.mean(segment_tokens, axis=2)
        segment_tokens = segment_tokens + condition[:, None, :]
        step_tokens = jnp.repeat(segment_tokens, repeats=self.steps_per_segment, axis=1)
        hidden = nnx.swish(self.hand_head_in(step_tokens))
        return self.hand_head_out(hidden)

    @staticmethod
    def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
        error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
        loss = jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)
        return jnp.mean(loss, axis=tuple(range(1, loss.ndim)))

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        del rng, train
        history_effort, future_effort = self._split_effort(observation, dtype=actions.dtype)
        c_future = self.encode_future(future_effort)
        condition = self._history_context(history_effort, observation.state)
        pred_hand_action = self.predict_hand_action(c_future, condition)
        target_hand_action = actions[
            :, :, self.hand_action_start : self.hand_action_start + self.hand_action_dim
        ]

        hand_loss = self._smooth_l1(pred_hand_action, target_hand_action)
        delta_loss = self._smooth_l1(
            pred_hand_action[:, 1:] - pred_hand_action[:, :-1],
            target_hand_action[:, 1:] - target_hand_action[:, :-1],
        )
        total = hand_loss + self.delta_loss_weight * delta_loss
        return total, {
            "loss/hand_action": hand_loss,
            "loss/hand_delta": delta_loss,
            "loss/total": total,
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("FutureTactileEncoderPretrain is a stage-1 training-only model.")
