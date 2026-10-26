"""Causal, frozen-policy features for the future physical-touch probe.

This is a batch adapter, not a dataset/training launcher. Call it with an
already restored, strictly validated Pi0LatentFlow checkpoint and the same
normalized Observation used at deployment. Future touch is separated before
calling the policy. Ground-truth actions are deliberately not an argument.

The offline replay follows recorded demonstration observations; it is not a
closed-loop rollout or an action-conditioned counterfactual world-model test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from openpi.models.future_patch_probe import make_strict_future_mask
from openpi.models.tactile_tokenizer import _patch_reconstruction_targets
from openpi.shared.effort_type import EffortType


def validate_probe_model(model: Any) -> None:
    """Reject old pooled/aggregate-force configs rather than silently adapting."""
    if not model.cached_vlm_async_ae_enabled:
        raise ValueError("Use the actual cached-VLM asynchronous policy checkpoint/config.")
    if not model.structured_tactile or model.tactile_dim_per_finger != 3 or model.effort_type != EffortType.MOT:
        raise ValueError("This adapter supports structured three-axis taxels with effort_type=MOT.")
    if not model.tactile_patch_informed_tokenizer or model.tactile_points_per_finger <= 1:
        raise ValueError("This adapter requires the patch-informed raw-taxel policy.")
    if model.disable_future_tactile or model.tactile_ttt_enabled:
        raise ValueError("This diagnostic requires future tactile query tokens, not a TTT-only policy.")
    if model.use_future_flow:
        raise ValueError("This adapter is scoped to the tactile-only future branch (use_future_flow=False).")
    if model.future_force_token_count != model.future_tactile_segments * model.tactile_num_fingers:
        raise ValueError("Expected one future token per segment and finger.")
    if model.future_tactile_segments * model.future_steps_per_segment != model.action_horizon:
        raise ValueError("Future token segments must exactly cover the action horizon.")
    # The policy's _split_effort detects a future sequence solely by its length.
    if model.force_input_frames > model.action_horizon:
        raise ValueError("History-only observations would be misread by this policy's effort splitter.")
    if not model.distill_layer_indices:
        raise ValueError("No alignment-layer index is configured.")


def probe_metadata(model: Any, *, num_steps: int = 10) -> dict[str, Any]:
    """Save alongside caches; also save checkpoint SHA/path and split IDs."""
    validate_probe_model(model)
    return {
        "feature_source": "FAE cached-VLM slow/fast deployment interfaces",
        "layer_index_zero_based": int(model.distill_layer_indices[-1]),
        "feature_stage": "last Euler velocity evaluation (not an extra evaluation at t=0)",
        "num_euler_steps": int(num_steps),
        "segments": int(model.future_tactile_segments),
        "steps_per_segment": int(model.future_steps_per_segment),
        "fingers": int(model.tactile_num_fingers),
        "patches": int(model.tactile_num_patches),
        "hidden_width": int(model.student_width),
        "future_frame_offsets": list(range(1, int(model.action_horizon) + 1)),
        "token_order": "segment-major, finger-minor",
        "tactile_sample_hz_config": float(model.tactile_sample_hz),
        "arm_hand_mask_attention": bool(model.arm_hand_mask_attention),
        "use_predicted_prefix_queries": bool(model.cached_vlm_async_use_predicted_prefix_queries),
        "evaluation_regime": "causal replay of recorded demonstration observations",
    }


def _history_only_observation(observation: Any, history: jax.Array) -> Any:
    """Remove label-only fields even if the dataset populated them."""
    return dataclasses.replace(
        observation,
        effort=history,
        tactile=None,
        flow_img=None,
        wrist_flow_img=None,
        future_rgb_img=None,
        future_wrist_rgb_img=None,
        scene_flow=None,
        is_contact=None,
    )


def _physical_targets(
    effort: jax.Array, tokenizer: Any, *, threshold: float, temperature: float
) -> tuple[jax.Array, jax.Array]:
    # This helper computes statistics directly from recorded taxels and the
    # fixed geometry; it does NOT invoke a learned reconstruction head/HAE.
    _, summary, contact = _patch_reconstruction_targets(
        effort,
        point_patch_ids=tokenizer.point_patch_ids,
        num_fingers=tokenizer.num_fingers,
        num_points=tokenizer.num_points,
        dim_per_point=tokenizer.dim_per_point,
        num_patches=tokenizer.num_patches,
        contact_threshold=threshold,
        contact_temperature=temperature,
    )
    force = jnp.where(contact[..., None] > 0, summary[..., :3], 0.0)
    return jax.lax.stop_gradient(contact), jax.lax.stop_gradient(force)


def extract_probe_batch(
    model: Any,
    rng: jax.Array,
    observation: Any,
    valid_future_frames: jax.Array,
    *,
    offsets: tuple[int, ...] | None = None,
    num_steps: int = 10,
    label_effort: jax.Array | None = None,
    label_contact_threshold: float | None = None,
    label_contact_temperature: float | None = None,
) -> dict[str, jax.Array]:
    """Return FAE features and independent future-contact/force targets.

    observation.effort: [B,L+H,F,P,3], with history t-L+1..t and
      future t+1..t+H. Only the history reaches the slow policy call.
    valid_future_frames: [B,H], MUST come from episode/timestamp metadata;
      repeated/clamped end-of-episode padding is not a real target.
    label_effort: optional unnormalized/calibrated taxels, same time/geometry
      layout. If omitted, labels use policy-normalized force units, NOT N.
      If supplied, threshold and temperature in its units are mandatory.

    At offset o, only touch through t+o is allowed into the fast policy call.
    Labels retain chunk-start coordinates, and valid_mask excludes j+1 <= o.
    Arrays returned have an offset axis O after B for features/current values.
    """
    validate_probe_model(model)
    model.eval()
    offsets = tuple(model.cached_vlm_async_offsets if offsets is None else offsets)
    horizon = int(model.action_horizon)
    history_length = int(model.force_input_frames)
    if not offsets or offsets[0] != 0 or tuple(sorted(set(offsets))) != offsets:
        raise ValueError("Offsets must be strictly increasing and start at zero.")
    if any(o < 0 or o >= horizon or o % model.future_steps_per_segment for o in offsets):
        raise ValueError("Offsets must be segment boundaries inside the action horizon.")
    # Intermediate recurrent calls cannot be skipped: that would change the
    # state whose physical information we intend to diagnose.
    if offsets != tuple(model.cached_vlm_async_offsets[: len(offsets)]):
        raise ValueError("Replay the configured recurrence in order; do not skip intermediate offsets.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    effort = jnp.asarray(observation.effort, dtype=jnp.float32)
    expected = (history_length + horizon, model.tactile_num_fingers, model.tactile_points_per_finger, 3)
    if effort.ndim != 5 or effort.shape[1:] != expected:
        raise ValueError(f"Expected merged effort [B,{expected}], got {effort.shape}.")
    valid_future_frames = jnp.asarray(valid_future_frames)
    if valid_future_frames.shape != effort.shape[:1] + (horizon,) or valid_future_frames.dtype != jnp.bool_:
        raise ValueError("valid_future_frames must be boolean [B, action_horizon].")
    tokenizer = model.student_force_tokenizer
    if label_effort is None:
        labels = effort
    else:
        if label_contact_threshold is None or label_contact_temperature is None:
            raise ValueError("Native/calibrated label forces need explicit threshold and temperature in their units.")
        labels = jnp.asarray(label_effort, dtype=jnp.float32)
        if labels.shape != effort.shape:
            raise ValueError("label_effort must have the same layout as observation.effort.")
    threshold = float(tokenizer.contact_threshold if label_contact_threshold is None else label_contact_threshold)
    temperature = float(tokenizer.contact_temperature if label_contact_temperature is None else label_contact_temperature)
    if threshold < 0 or temperature <= 0:
        raise ValueError("Use a nonnegative contact threshold and positive temperature.")

    target_contact, target_force = _physical_targets(
        labels[:, history_length:], tokenizer, threshold=threshold, temperature=temperature
    )
    history = effort[:, :history_length]
    initial_observation = _history_only_observation(observation, history)
    noise_rng, policy_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, (effort.shape[0], horizon, model.action_dim), dtype=jnp.float32)
    _, prefix_cache, prefix_mask, hidden = model.sample_actions_cached_vlm_async_ae_slow(
        policy_rng, initial_observation, noise=noise, num_steps=num_steps
    )
    first_hidden = jax.lax.stop_gradient(hidden)
    refined, history_tokens, contacts, forces, masks = [], [], [], [], []
    for offset in offsets:
        fresh_history = effort[:, offset : offset + history_length]
        if offset:
            # Keep chunk-start vision/language/proprioception; only tactile
            # history changes, exactly as in the cached deployment pathway.
            _, hidden = model.sample_actions_cached_vlm_async_ae_fast(
                policy_rng,
                initial_observation,
                prefix_kv_cache=prefix_cache,
                prefix_mask=prefix_mask,
                prefix_future_hidden=jax.lax.stop_gradient(hidden),
                async_chunk_offset=offset,
                async_fresh_effort=fresh_history,
                noise=noise,
                num_steps=num_steps,
            )
        refined.append(jax.lax.stop_gradient(hidden))
        history_tokens.append(jax.lax.stop_gradient(model._project_history_force_student(fresh_history)))
        current_contact, current_force = _physical_targets(
            labels[:, history_length + offset - 1 : history_length + offset],
            tokenizer,
            threshold=threshold,
            temperature=temperature,
        )
        contacts.append(current_contact[:, 0])
        forces.append(current_force[:, 0])
        mask = make_strict_future_mask(offset, valid_future_frames)
        # If the recorded observation needed for a refinement is missing,
        # none of that row's predictions constitutes a valid replay sample.
        observed_prefix_valid = jnp.all(valid_future_frames[:, :offset], axis=1)
        masks.append(mask & observed_prefix_valid[:, None])

    refined_hidden = jnp.stack(refined, axis=1)
    return {
        "refined_hidden": refined_hidden,
        "fixed_hidden": jnp.broadcast_to(first_hidden[:, None], refined_hidden.shape),
        "history_tokens": jnp.stack(history_tokens, axis=1),
        "target_contact": target_contact,
        "target_force": target_force,
        "current_contact": jnp.stack(contacts, axis=1),
        "current_force": jnp.stack(forces, axis=1),
        "valid_mask": jnp.stack(masks, axis=1),
        "offsets": jnp.asarray(offsets, dtype=jnp.int32),
        "future_frame_offsets": jnp.arange(1, horizon + 1, dtype=jnp.int32),
    }
