"""Read-only probes for physical information in frozen future tactile latents.

The input must come from a frozen, deployable FAE inference path. This module
never accesses a policy, an HAE, actions, current touch, or future observations.
Ground-truth future patch measurements are supplied only to the loss function.

Tokens are segment-major, then finger-major. A token predicts every frame in
its segment independently; there is no averaging across its four target frames.
For the paper configuration, [B, 20, 1024] becomes contact logits [B, 16, 5, 5]
and forces [B, 16, 5, 5, 3]. The shared linear probe has 82,000 parameters.
"""

from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class FuturePatchProbe(nnx.Module):
    """Decode each frozen segment/finger token with the same small head.

    ``linear`` is a strict affine probe, without normalization, embeddings,
    attention, or additional inputs. ``mlp`` adds one hidden SiLU layer and
    should be reported separately as a nonlinear diagnostic.

    Detaching the input is a second safeguard; the caller must also keep the
    original policy out of the probe optimizer and use policy evaluation mode.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_segments: int = 4,
        num_fingers: int = 5,
        steps_per_segment: int = 4,
        num_patches: int = 5,
        decoder_type: Literal["linear", "mlp"] = "linear",
        hidden_dim: int = 128,
        rngs: nnx.Rngs,
    ):
        dimensions = {
            "input_dim": input_dim,
            "num_segments": num_segments,
            "num_fingers": num_fingers,
            "steps_per_segment": steps_per_segment,
            "num_patches": num_patches,
            "hidden_dim": hidden_dim,
        }
        for name, value in dimensions.items():
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if decoder_type not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported decoder_type={decoder_type!r}.")

        self.input_dim = int(input_dim)
        self.num_segments = int(num_segments)
        self.num_fingers = int(num_fingers)
        self.steps_per_segment = int(steps_per_segment)
        self.num_patches = int(num_patches)
        self.action_horizon = self.num_segments * self.steps_per_segment
        self.decoder_type = decoder_type
        self.hidden_layer = (
            nnx.Linear(self.input_dim, int(hidden_dim), rngs=rngs) if decoder_type == "mlp" else None
        )
        readout_width = int(hidden_dim) if decoder_type == "mlp" else self.input_dim
        contact_outputs = self.steps_per_segment * self.num_patches
        self.contact_head = nnx.Linear(readout_width, contact_outputs, rngs=rngs)
        self.force_head = nnx.Linear(readout_width, contact_outputs * 3, rngs=rngs)

    def __call__(self, future_hidden: jax.Array) -> dict[str, jax.Array]:
        expected = (self.num_segments * self.num_fingers, self.input_dim)
        if future_hidden.ndim != 3 or future_hidden.shape[1:] != expected:
            raise ValueError(f"Expected future_hidden [B,{expected[0]},{expected[1]}], got {future_hidden.shape}.")
        hidden = jax.lax.stop_gradient(jnp.asarray(future_hidden, dtype=jnp.float32))
        if self.hidden_layer is not None:
            hidden = nnx.swish(self.hidden_layer(hidden))

        batch_size = hidden.shape[0]
        contact = self.contact_head(hidden).reshape(
            batch_size, self.num_segments, self.num_fingers, self.steps_per_segment, self.num_patches
        )
        # [B, segment, finger, step, patch] -> [B, absolute future step, finger, patch].
        contact = contact.transpose(0, 1, 3, 2, 4).reshape(
            batch_size, self.action_horizon, self.num_fingers, self.num_patches
        )
        force = self.force_head(hidden).reshape(
            batch_size, self.num_segments, self.num_fingers, self.steps_per_segment, self.num_patches, 3
        )
        force = force.transpose(0, 1, 3, 2, 4, 5).reshape(
            batch_size, self.action_horizon, self.num_fingers, self.num_patches, 3
        )
        return {"contact_logits": contact, "force": force}


def make_strict_future_mask(offsets: jax.Array | int, valid_future_frames: jax.Array) -> jax.Array:
    """Mask observed and padded frames independently of the policy loss mask.

    Args:
        offsets: Integer scalar or [B]. An offset of 4 means touch through t+4
            is already visible. Callers should validate 0 <= offset <= H.
        valid_future_frames: Required boolean [B,H], derived from episode
            boundaries or frame IDs, not from whether the force is nonzero.
            Array index j describes tactile frame t+j+1.

    Returns:
        Boolean [B,H]. At offset 4, only t+5 ... t+H can remain valid.
        At offset H, every entry is false. Safe to use inside jax.jit.
    """
    valid = jnp.asarray(valid_future_frames)
    if valid.ndim != 2 or valid.dtype != jnp.bool_:
        raise ValueError("valid_future_frames must be a boolean [B,H] array.")
    offset = jnp.asarray(offsets)
    if not jnp.issubdtype(offset.dtype, jnp.integer):
        raise ValueError("offsets must have an integer dtype.")
    if offset.ndim == 0:
        offset = jnp.broadcast_to(offset, (valid.shape[0],))
    if offset.shape != (valid.shape[0],):
        raise ValueError(f"Expected offsets scalar or [{valid.shape[0]}], got {offset.shape}.")
    future_steps = jnp.arange(1, valid.shape[1] + 1, dtype=offset.dtype)
    return valid & (future_steps[None, :] > offset[:, None])


def future_patch_probe_loss(
    predictions: dict[str, jax.Array],
    target_contact: jax.Array,
    target_force: jax.Array,
    *,
    valid_mask: jax.Array,
    positive_contact_weight: float = 1.0,
    contact_loss_weight: float = 1.0,
    active_force_weight: float = 1.0,
    inactive_force_weight: float = 0.1,
    smooth_l1_beta: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Physical contact/force loss, normalized by actual valid patch counts.

    ``target_contact`` is binary [B,H,F,R]. ``target_force`` is the measured
    patch-wise three-axis force [B,H,F,R,3] in an explicitly chosen coordinate
    system and units. Inactive-patch force targets are set to zero. If force
    scaling is desired, apply scales fitted on probe-training episodes only;
    the module does not infer units, thresholds, or normalization statistics.

    ``valid_mask`` is the strict-future [B,H] mask. Masked predictions/labels
    are sanitized before arithmetic, so padding may contain NaN placeholders.
    Active and inactive force terms each average over their own patch/axis
    count. No valid targets gives zero loss with zero counts; callers should
    skip that batch. Ground-truth targets never receive gradients.
    """
    logits = jnp.asarray(predictions["contact_logits"], dtype=jnp.float32)
    force = jnp.asarray(predictions["force"], dtype=jnp.float32)
    target_contact = jnp.asarray(target_contact, dtype=jnp.float32)
    target_force = jnp.asarray(target_force, dtype=jnp.float32)
    valid_mask = jnp.asarray(valid_mask)
    if logits.ndim != 4 or target_contact.shape != logits.shape:
        raise ValueError("contact logits and targets must have matching [B,H,F,R] shapes.")
    if force.shape != (*logits.shape, 3) or target_force.shape != force.shape:
        raise ValueError("force prediction and target must have shape [B,H,F,R,3].")
    if valid_mask.shape != logits.shape[:2] or valid_mask.dtype != jnp.bool_:
        raise ValueError("valid_mask must be boolean [B,H] matching the prediction batch and horizon.")
    if smooth_l1_beta <= 0:
        raise ValueError("smooth_l1_beta must be positive.")
    if positive_contact_weight <= 0:
        raise ValueError("positive_contact_weight must be positive.")
    if min(contact_loss_weight, active_force_weight, inactive_force_weight) < 0:
        raise ValueError("Loss weights must be nonnegative.")

    patch_mask = jnp.broadcast_to(valid_mask[:, :, None, None], logits.shape)
    logits = jnp.where(patch_mask, logits, 0.0)
    contact = jax.lax.stop_gradient(jnp.where(patch_mask, target_contact, 0.0))
    force = jnp.where(patch_mask[..., None], force, 0.0)
    active = patch_mask & (contact > 0.5)
    inactive = patch_mask & ~active
    target_force = jax.lax.stop_gradient(jnp.where(active[..., None], target_force, 0.0))

    valid_count = jnp.sum(patch_mask, dtype=jnp.float32)
    active_count = jnp.sum(active, dtype=jnp.float32)
    inactive_count = jnp.sum(inactive, dtype=jnp.float32)
    # This is the weighted sigmoid cross entropy, evaluated in float32.
    bce = (1.0 - contact) * jax.nn.softplus(logits)
    bce += positive_contact_weight * contact * jax.nn.softplus(-logits)
    contact_loss = jnp.sum(jnp.where(patch_mask, bce, 0.0)) / jnp.maximum(valid_count, 1.0)

    error = jnp.abs(force - target_force)
    force_error = jnp.where(
        error < smooth_l1_beta,
        0.5 * jnp.square(error) / smooth_l1_beta,
        error - 0.5 * smooth_l1_beta,
    )
    active_loss = jnp.sum(jnp.where(active[..., None], force_error, 0.0)) / jnp.maximum(3.0 * active_count, 1.0)
    inactive_loss = jnp.sum(jnp.where(inactive[..., None], force_error, 0.0)) / jnp.maximum(
        3.0 * inactive_count, 1.0
    )
    total = (
        contact_loss_weight * contact_loss
        + active_force_weight * active_loss
        + inactive_force_weight * inactive_loss
    )
    stats = {
        "loss/contact": contact_loss,
        "loss/active_force": active_loss,
        "loss/inactive_force": inactive_loss,
        "count/valid_frames": jnp.sum(valid_mask, dtype=jnp.float32),
        "count/valid_patches": valid_count,
        "count/active_patches": active_count,
        "count/inactive_patches": inactive_count,
    }
    return total, stats
