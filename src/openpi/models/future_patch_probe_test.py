"""Temporal alignment, isolation, and masking checks for the diagnostic probe."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.future_patch_probe import FuturePatchProbe
from openpi.models.future_patch_probe import future_patch_probe_loss
from openpi.models.future_patch_probe import make_strict_future_mask


def test_segment_finger_and_within_segment_frame_order():
    probe = FuturePatchProbe(
        input_dim=1,
        num_segments=2,
        num_fingers=3,
        steps_per_segment=2,
        num_patches=2,
        rngs=nnx.Rngs(0),
    )
    probe.contact_head.kernel.value = jnp.ones((1, 4))
    probe.contact_head.bias.value = jnp.arange(4, dtype=jnp.float32)
    probe.force_head.kernel.value = jnp.ones((1, 12))
    probe.force_head.bias.value = jnp.arange(12, dtype=jnp.float32)
    hidden = jnp.asarray([[[100 * segment + 10 * finger] for segment in range(2) for finger in range(3)]])
    prediction = probe(hidden)
    assert prediction["contact_logits"].shape == (1, 4, 3, 2)
    assert prediction["force"].shape == (1, 4, 3, 2, 3)
    for segment in range(2):
        for step in range(2):
            frame = segment * 2 + step
            for finger in range(3):
                base = 100 * segment + 10 * finger
                for patch in range(2):
                    assert prediction["contact_logits"][0, frame, finger, patch] == base + 2 * step + patch
                    for axis in range(3):
                        assert prediction["force"][0, frame, finger, patch, axis] == (
                            base + 6 * step + 3 * patch + axis
                        )


def test_paper_linear_probe_has_only_82000_parameters():
    probe = FuturePatchProbe(input_dim=1024, rngs=nnx.Rngs(0))
    params = nnx.state(probe, nnx.Param)
    assert set(params.keys()) == {"contact_head", "force_head"}
    assert sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)) == 82000
    prediction = probe(jnp.zeros((2, 20, 1024)))
    assert prediction["contact_logits"].shape == (2, 16, 5, 5)
    assert prediction["force"].shape == (2, 16, 5, 5, 3)


@pytest.mark.parametrize("decoder_type", ["linear", "mlp"])
def test_input_is_detached_while_probe_parameters_receive_gradients(decoder_type):
    probe = FuturePatchProbe(input_dim=8, decoder_type=decoder_type, hidden_dim=4, rngs=nnx.Rngs(0))
    hidden = jnp.ones((1, 20, 8))

    def objective(model, inputs):
        prediction = model(inputs)
        return jnp.sum(prediction["force"]) + jnp.sum(prediction["contact_logits"])

    input_gradient = jax.grad(lambda inputs: objective(probe, inputs))(hidden)
    np.testing.assert_array_equal(input_gradient, jnp.zeros_like(hidden))
    parameter_gradient = nnx.grad(lambda model: objective(model, hidden))(probe)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree_util.tree_leaves(parameter_gradient))


def test_offsets_and_episode_padding_mask_strictly_future_frames():
    valid_frames = jnp.ones((4, 16), dtype=jnp.bool_).at[1, 14:].set(False)
    offsets = jnp.asarray([0, 4, 8, 12])
    actual = jax.jit(make_strict_future_mask)(offsets, valid_frames)
    expected = (np.arange(1, 17)[None, :] > np.asarray(offsets)[:, None]) & np.asarray(valid_frames)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.sum(axis=1), np.asarray([16, 10, 8, 4]))
    np.testing.assert_array_equal(
        make_strict_future_mask(4, jnp.ones((1, 16), dtype=jnp.bool_))[0],
        np.arange(1, 17) > 4,
    )
    assert not make_strict_future_mask(16, valid_frames).any()


def test_loss_ignores_observed_prefix_and_padded_future_even_with_nan_placeholders():
    contact = jnp.asarray([[[[0.0]], [[1.0]], [[1.0]], [[0.0]]]])
    force = jnp.ones((1, 4, 1, 1, 3))
    predictions = {"contact_logits": jnp.zeros_like(contact), "force": jnp.zeros_like(force)}
    valid_frames = jnp.asarray([[True, True, True, False]])
    valid_mask = make_strict_future_mask(2, valid_frames)
    expected_loss, expected_stats = future_patch_probe_loss(predictions, contact, force, valid_mask=valid_mask)

    invalid = jnp.broadcast_to(~valid_mask[:, :, None, None], contact.shape)
    changed_contact = jnp.where(invalid, jnp.nan, contact)
    changed_force = jnp.where(invalid[..., None], jnp.inf, force)
    changed_predictions = {
        "contact_logits": jnp.where(invalid, jnp.nan, predictions["contact_logits"]),
        "force": jnp.where(invalid[..., None], jnp.nan, predictions["force"]),
    }
    actual_loss, actual_stats = future_patch_probe_loss(
        changed_predictions, changed_contact, changed_force, valid_mask=valid_mask
    )
    assert np.isfinite(actual_loss)
    np.testing.assert_allclose(actual_loss, expected_loss)
    for name in expected_stats:
        np.testing.assert_allclose(actual_stats[name], expected_stats[name])
    assert actual_stats["count/valid_frames"] == 1
    assert actual_stats["count/valid_patches"] == 1


def test_inactive_force_targets_are_zero_and_counts_normalize_each_group():
    contact = jnp.zeros((1, 2, 2, 3))
    prediction = {"contact_logits": jnp.zeros_like(contact), "force": jnp.ones((*contact.shape, 3))}
    loss, stats = future_patch_probe_loss(
        prediction,
        contact,
        jnp.full((*contact.shape, 3), 999.0),
        valid_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        contact_loss_weight=0.0,
        inactive_force_weight=1.0,
    )
    np.testing.assert_allclose(loss, 0.5)
    assert stats["count/active_patches"] == 0
    assert stats["count/inactive_patches"] == 12
    assert stats["loss/active_force"] == 0


def test_active_force_normalization_is_not_diluted_by_inactive_patches():
    contact = jnp.asarray([[[[1.0, 0.0, 0.0]]]])
    prediction = {"contact_logits": jnp.zeros_like(contact), "force": jnp.zeros((*contact.shape, 3))}
    target = jnp.full((*contact.shape, 3), 2.0)
    loss, stats = future_patch_probe_loss(
        prediction,
        contact,
        target,
        valid_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        contact_loss_weight=0.0,
        inactive_force_weight=0.0,
    )
    np.testing.assert_allclose(loss, 1.5)
    np.testing.assert_allclose(stats["loss/active_force"], 1.5)


def test_all_masked_loss_and_gradients_are_finite_zero():
    contact = jnp.full((1, 2, 1, 1), jnp.nan)
    force = jnp.full((*contact.shape, 3), jnp.nan)
    valid_mask = jnp.zeros((1, 2), dtype=jnp.bool_)

    def objective(logits, predicted_force):
        return future_patch_probe_loss(
            {"contact_logits": logits, "force": predicted_force}, contact, force, valid_mask=valid_mask
        )[0]

    loss, stats = future_patch_probe_loss(
        {"contact_logits": contact, "force": force}, contact, force, valid_mask=valid_mask
    )
    assert loss == 0
    assert stats["count/valid_patches"] == 0
    gradients = jax.grad(objective, argnums=(0, 1))(contact, force)
    for gradient in gradients:
        assert np.isfinite(gradient).all()
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_targets_do_not_receive_gradients():
    contact = jnp.ones((1, 1, 1, 1))
    force = jnp.ones((*contact.shape, 3))
    predictions = {"contact_logits": jnp.zeros_like(contact), "force": jnp.zeros_like(force)}

    def objective(target_contact, target_force):
        return future_patch_probe_loss(
            predictions, target_contact, target_force, valid_mask=jnp.ones((1, 1), dtype=jnp.bool_)
        )[0]

    for gradient in jax.grad(objective, argnums=(0, 1))(contact, force):
        np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


def test_large_logits_use_stable_cross_entropy():
    contact = jnp.asarray([[[[1.0, 0.0]]]])
    prediction = {
        "contact_logits": jnp.asarray([[[[-1000.0, 1000.0]]]]),
        "force": jnp.zeros((*contact.shape, 3)),
    }
    loss, _ = future_patch_probe_loss(
        prediction,
        contact,
        jnp.zeros((*contact.shape, 3)),
        valid_mask=jnp.ones((1, 1), dtype=jnp.bool_),
        positive_contact_weight=2.0,
    )
    assert np.isfinite(loss)
    np.testing.assert_allclose(loss, 1500.0)


def test_invalid_layout_and_mask_fail_loudly():
    probe = FuturePatchProbe(input_dim=8, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="Expected future_hidden"):
        probe(jnp.zeros((1, 5, 8)))
    with pytest.raises(ValueError, match="boolean"):
        make_strict_future_mask(0, jnp.ones((1, 16)))
    with pytest.raises(ValueError, match="integer dtype"):
        make_strict_future_mask(4.0, jnp.ones((1, 16), dtype=jnp.bool_))
