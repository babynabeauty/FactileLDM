"""Adapter contract tests; these do not execute a trained robot policy."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.extract_future_patch_probe import extract_probe_batch, probe_metadata
from openpi.shared.effort_type import EffortType


@dataclasses.dataclass
class FakeObservation:
    effort: object
    state: object
    tactile: object = "label-only"
    flow_img: object = "label-only"
    wrist_flow_img: object = "label-only"
    future_rgb_img: object = "label-only"
    future_wrist_rgb_img: object = "label-only"
    scene_flow: object = "label-only"
    is_contact: object = "label-only"


class FakePolicy:
    cached_vlm_async_ae_enabled = True
    tactile_patch_informed_tokenizer = True
    structured_tactile = True
    tactile_dim_per_finger = 3
    effort_type = EffortType.MOT
    tactile_points_per_finger = 5
    disable_future_tactile = False
    tactile_ttt_enabled = False
    use_future_flow = False
    future_force_token_count = 20
    future_tactile_segments = 4
    tactile_num_fingers = 5
    future_steps_per_segment = 4
    action_horizon = 16
    force_input_frames = 10
    action_dim = 32
    student_width = 4
    tactile_num_patches = 5
    tactile_sample_hz = 15.0
    arm_hand_mask_attention = False
    distill_layer_indices = (12,)
    cached_vlm_async_offsets = (0, 4, 8, 12)
    cached_vlm_async_use_predicted_prefix_queries = True

    def __init__(self):
        self.student_force_tokenizer = SimpleNamespace(
            point_patch_ids=tuple(tuple(range(5)) for _ in range(5)),
            num_fingers=5, num_points=5, dim_per_point=3, num_patches=5,
            contact_threshold=1.0, contact_temperature=0.5,
        )
        self.calls = []
        self.is_eval = False

    def eval(self):
        self.is_eval = True

    def _check_observation(self, observation):
        assert self.is_eval
        assert observation.effort.shape[1] == self.force_input_frames
        for name in (
            "tactile", "flow_img", "wrist_flow_img", "future_rgb_img",
            "future_wrist_rgb_img", "scene_flow", "is_contact",
        ):
            assert getattr(observation, name) is None

    def _hidden(self, effort):
        summary = jnp.mean(effort, axis=(1, 2, 3, 4))
        return jnp.broadcast_to(summary[:, None, None], (effort.shape[0], 20, 4))

    def sample_actions_cached_vlm_async_ae_slow(self, rng, observation, *, noise, num_steps):
        self._check_observation(observation)
        assert noise.shape == (observation.effort.shape[0], 16, 32)
        self.calls.append((0, observation.effort, noise, num_steps))
        return noise, "fixed-cache", jnp.ones((1, 1), dtype=bool), self._hidden(observation.effort)

    def sample_actions_cached_vlm_async_ae_fast(
        self, rng, observation, *, prefix_kv_cache, prefix_mask,
        prefix_future_hidden, async_chunk_offset, async_fresh_effort, noise, num_steps,
    ):
        self._check_observation(observation)
        assert prefix_kv_cache == "fixed-cache"
        self.calls.append((async_chunk_offset, async_fresh_effort, noise, num_steps))
        # Make skipped/interleaved recurrence observable to the tests.
        hidden = self._hidden(async_fresh_effort) + 0.1 * prefix_future_hidden
        return noise, hidden

    def _project_history_force_student(self, effort):
        return self._hidden(effort)[:, :10]


def make_observation():
    # Index L-1 is current t; index L is t+1.
    effort = jnp.broadcast_to(jnp.arange(26, dtype=jnp.float32)[None, :, None, None, None], (2, 26, 5, 5, 3))
    return FakeObservation(effort=effort, state=jnp.zeros((2, 32)))


def test_replay_layout_cutoffs_targets_and_noise():
    model = FakePolicy()
    obs = make_observation()
    records = extract_probe_batch(model, jax.random.key(0), obs, jnp.ones((2, 16), dtype=bool))
    assert records["refined_hidden"].shape == (2, 4, 20, 4)
    assert records["history_tokens"].shape == (2, 4, 10, 4)
    assert records["target_contact"].shape == (2, 16, 5, 5)
    assert records["target_force"].shape == (2, 16, 5, 5, 3)
    np.testing.assert_array_equal(records["valid_mask"].sum(-1), [[16, 12, 8, 4]] * 2)
    np.testing.assert_allclose(records["target_force"][0, :, 0, 0, 0], np.arange(10, 26))
    np.testing.assert_allclose(records["current_force"][0, :, 0, 0, 0], [9, 13, 17, 21])
    for offset_index, (offset, history, noise, steps) in enumerate(model.calls):
        np.testing.assert_array_equal(history, obs.effort[:, offset:offset + 10])
        np.testing.assert_array_equal(noise, model.calls[0][2])
        assert steps == 10
        np.testing.assert_array_equal(records["fixed_hidden"][:, offset_index], records["refined_hidden"][:, 0])
    metadata = probe_metadata(model)
    assert metadata["layer_index_zero_based"] == 12
    assert metadata["future_frame_offsets"] == list(range(1, 17))


def test_future_after_observed_cutoff_cannot_change_features():
    obs = make_observation()
    valid = jnp.ones((2, 16), dtype=bool)
    base = extract_probe_batch(FakePolicy(), jax.random.key(0), obs, valid)
    # At offset 4, the latest legal input is t+4 (merged index 13).
    changed = dataclasses.replace(obs, effort=obs.effort.at[:, 14:].add(100))
    other = extract_probe_batch(FakePolicy(), jax.random.key(0), changed, valid)
    np.testing.assert_array_equal(base["refined_hidden"][:, :2], other["refined_hidden"][:, :2])
    np.testing.assert_array_equal(base["history_tokens"][:, :2], other["history_tokens"][:, :2])
    assert not np.allclose(base["refined_hidden"][:, 2:], other["refined_hidden"][:, 2:])
    assert not np.allclose(base["target_force"], other["target_force"])


def test_label_units_never_change_policy_features():
    obs = make_observation()
    valid = jnp.ones((2, 16), dtype=bool)
    base = extract_probe_batch(FakePolicy(), jax.random.key(0), obs, valid)
    other = extract_probe_batch(
        FakePolicy(), jax.random.key(0), obs, valid,
        label_effort=obs.effort * 10,
        label_contact_threshold=10.0, label_contact_temperature=5.0,
    )
    np.testing.assert_array_equal(base["refined_hidden"], other["refined_hidden"])
    np.testing.assert_allclose(other["target_force"], base["target_force"] * 10, rtol=1e-6)
    with pytest.raises(ValueError, match="explicit threshold"):
        extract_probe_batch(FakePolicy(), jax.random.key(0), obs, valid, label_effort=obs.effort)


def test_padding_and_missing_observed_prefix_remove_invalid_rows():
    valid = jnp.ones((2, 16), dtype=bool).at[0, 10:].set(False).at[1, 2].set(False)
    records = extract_probe_batch(FakePolicy(), jax.random.key(0), make_observation(), valid)
    # Episode 0 ends at t+10; episode 1 lacks t+3, so later replay is invalid.
    np.testing.assert_array_equal(records["valid_mask"].sum(-1), [[10, 6, 2, 0], [15, 0, 0, 0]])


@pytest.mark.parametrize("offsets", [(0, 8), (4, 8), (0, 4, 4), (0, 2)])
def test_invalid_recurrence_is_rejected(offsets):
    with pytest.raises(ValueError, match="Offsets|Replay"):
        extract_probe_batch(
            FakePolicy(), jax.random.key(0), make_observation(),
            jnp.ones((2, 16), dtype=bool), offsets=offsets,
        )


def test_shape_and_config_mismatches_fail_before_policy_call():
    model = FakePolicy()
    obs = make_observation()
    with pytest.raises(ValueError, match="merged effort"):
        extract_probe_batch(model, jax.random.key(0), dataclasses.replace(obs, effort=obs.effort[:, :10]), jnp.ones((2, 16), dtype=bool))
    assert not model.calls
    model.future_force_token_count = 5
    with pytest.raises(ValueError, match="one future token"):
        extract_probe_batch(model, jax.random.key(0), obs, jnp.ones((2, 16), dtype=bool))


def test_nonboolean_mask_and_unsupported_effort_type_rejected():
    model = FakePolicy()
    with pytest.raises(ValueError, match="boolean"):
        extract_probe_batch(model, jax.random.key(0), make_observation(), jnp.ones((2, 16)))
    model.effort_type = EffortType.STATE
    with pytest.raises(ValueError, match="effort_type=MOT"):
        extract_probe_batch(model, jax.random.key(0), make_observation(), jnp.ones((2, 16), dtype=bool))
