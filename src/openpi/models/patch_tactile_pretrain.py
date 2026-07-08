import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.tactile_tokenizer import PatchInformedFingerTokenizer


def _sigmoid_bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    return jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)


class XHandPatchTactileEncoderPretrain(nnx.Module):
    """Stage-1 pretraining for the patch-informed XHand tactile encoder.

    This model trains only the raw tactile encoder and lightweight decoder heads.
    The encoder keeps the policy-facing token budget at one token per finger,
    while the heads force each finger token to retain local 5-patch contact
    structure.
    """

    def __init__(self, config: pi0_config.XHandPatchTactileEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.num_points = int(config.tactile_points_per_finger)
        self.dim_per_point = int(config.tactile_dim_per_finger)
        self.num_patches = int(config.tactile_num_patches)
        self.encoder_width = int(config.encoder_width)
        self.summary_dim = 2 * self.dim_per_point + 1
        self.distribution_loss_weight = float(config.patch_distribution_loss_weight)
        self.summary_loss_weight = float(config.patch_summary_loss_weight)
        self.contact_loss_weight = float(config.patch_contact_loss_weight)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.patch_encoder = PatchInformedFingerTokenizer(
            output_dim=self.encoder_width,
            hidden_dim=config.tactile_tokenizer_dim,
            num_fingers=self.num_fingers,
            num_points=self.num_points,
            dim_per_point=self.dim_per_point,
            future_segments=config.future_tactile_segments,
            future_steps_per_segment=config.future_steps_per_segment,
            contact_top_k=config.tactile_raw_contact_top_k,
            contact_threshold=config.tactile_raw_contact_threshold,
            contact_temperature=config.tactile_raw_contact_temperature,
            num_patches=self.num_patches,
            rngs=rngs,
        )
        self.patch_distribution_head = nnx.Linear(self.encoder_width, self.num_patches, rngs=rngs)
        self.patch_summary_head = nnx.Linear(
            self.encoder_width, self.num_patches * self.summary_dim, rngs=rngs
        )
        self.patch_contact_head = nnx.Linear(self.encoder_width, self.num_patches, rngs=rngs)

    def _split_effort(self, observation: _model.Observation, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
        if observation.effort is None:
            raise ValueError("XHand patch tactile encoder pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (
            self.force_input_frames + self.action_horizon,
            self.num_fingers,
            self.num_points,
            self.dim_per_point,
        )
        if effort.ndim != 5 or effort.shape[1:] != expected:
            raise ValueError(f"Expected raw tactile effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _encode_all_steps(self, history_effort: jax.Array, future_effort: jax.Array) -> jax.Array:
        history_tokens = self.patch_encoder._encode_steps(
            history_effort,
            jnp.asarray(self.history_times, dtype=jnp.float32),
            future=False,
        )
        future_tokens = self.patch_encoder._encode_steps(
            future_effort,
            jnp.asarray(self.future_times, dtype=jnp.float32),
            future=True,
        )
        return jnp.concatenate([history_tokens, future_tokens], axis=1)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        del rng, train
        history_effort, future_effort = self._split_effort(observation, dtype=actions.dtype)
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        tokens = self._encode_all_steps(history_effort, future_effort)

        target_dist, target_summary, target_contact = self.patch_encoder.patch_reconstruction_targets(effort)
        dist_logits = self.patch_distribution_head(tokens)
        summary_pred = self.patch_summary_head(tokens)
        summary_pred = einops.rearrange(
            summary_pred,
            "b t f (r c) -> b t f r c",
            r=self.num_patches,
            c=self.summary_dim,
        )
        contact_logits = self.patch_contact_head(tokens)

        dist_loss = -jnp.sum(
            jax.lax.stop_gradient(target_dist) * jax.nn.log_softmax(dist_logits.astype(jnp.float32), axis=-1),
            axis=-1,
        )
        dist_loss = jnp.mean(dist_loss, axis=(1, 2))
        summary_loss = jnp.mean(_smooth_l1(summary_pred, jax.lax.stop_gradient(target_summary)), axis=(1, 2, 3, 4))
        contact_loss = jnp.mean(
            _sigmoid_bce_with_logits(contact_logits, jax.lax.stop_gradient(target_contact)),
            axis=(1, 2, 3),
        )
        total = (
            self.distribution_loss_weight * dist_loss
            + self.summary_loss_weight * summary_loss
            + self.contact_loss_weight * contact_loss
        )

        pred_contact = jax.nn.sigmoid(contact_logits.astype(jnp.float32)) > 0.5
        target_contact_bool = target_contact > 0.5
        contact_accuracy = jnp.mean(pred_contact == target_contact_bool, axis=(1, 2, 3))
        return total, {
            "loss/patch_distribution": dist_loss,
            "loss/patch_summary": summary_loss,
            "loss/patch_contact": contact_loss,
            "loss/total": total,
            "metric/contact_accuracy": contact_accuracy,
            "metric/pred_contact_ratio": jnp.mean(pred_contact.astype(jnp.float32), axis=(1, 2, 3)),
            "metric/target_contact_ratio": jnp.mean(target_contact, axis=(1, 2, 3)),
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("XHandPatchTactileEncoderPretrain is a stage-1 training-only model.")
