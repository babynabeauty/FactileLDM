"""CPU safety checks and an optional four-device numerical equivalence test."""

import threading

import numpy as np
import pytest

from openpi.shared.probe_parallel import ProbeParallel
from openpi.shared.probe_parallel import pad_probe_batch
from openpi.shared.probe_parallel import prefetch_batches
from openpi.shared.probe_parallel import read_probe_cache


def test_padding_preserves_real_rows_and_masks_added_rows():
    x = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    contact = np.ones((3, 4, 1, 1), dtype=np.float32)
    force = np.ones((*contact.shape, 3), dtype=np.float32)
    mask = np.array([[True] * 4, [False, False, True, True], [False] * 4])
    padded = pad_probe_batch((x, contact, force, mask), 8)
    for original, actual in zip((x, contact, force, mask), padded, strict=True):
        np.testing.assert_array_equal(actual[:3], original)
        assert len(actual) == 8
    assert not padded[-1][3:].any()
    np.testing.assert_array_equal(mask[0], [True] * 4)


def test_prefetch_preserves_order_propagates_failure_and_joins():
    def failing():
        yield 1
        yield 2
        raise RuntimeError("reader failure")

    with prefetch_batches(range(4)) as stream:
        assert list(stream) == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="reader failure"):
        with prefetch_batches(failing()) as stream:
            assert next(stream) == 1
            assert next(stream) == 2
            next(stream)
    with prefetch_batches(range(10)) as stream:
        assert next(stream) == 0  # Early smoke-test exit.
    assert not any(t.name.startswith("probe-prefetch") for t in threading.enumerate())


def test_cache_reader_does_not_load_unused_feature_arrays(tmp_path):
    path = tmp_path / "shard.npz"
    keys = ("frame_index", "offsets", "target_contact", "target_force", "valid_mask", "fixed_hidden", "refined_hidden")
    arrays = {key: np.ones(2) for key in keys}
    # This unused object array cannot be loaded with allow_pickle=False.
    # FAE training must not touch it.
    np.savez(path, **arrays, history_tokens=np.array([object()], dtype=object))
    result = read_probe_cache(path, "fae")
    assert set(result) == set(keys)


def test_four_device_probe_loss_gradients_and_prediction_match_single_device():
    import flax.nnx as nnx
    import jax

    from openpi.models.future_patch_probe import FuturePatchProbe
    from openpi.models.future_patch_probe import future_patch_probe_loss
    from scripts.run_future_patch_probe import _prediction_fn
    from scripts.run_future_patch_probe import _validation_stats_fn

    if jax.local_device_count() < 4:
        pytest.skip("Run with four visible GPUs (or four forced CPU devices).")
    parallel = ProbeParallel(4, 8)
    with jax.default_device(jax.local_devices()[0]):
        probe = FuturePatchProbe(
            input_dim=4, num_segments=2, num_fingers=2,
            steps_per_segment=2, num_patches=2, rngs=nnx.Rngs(42),
        )
        graphdef, params = nnx.split(probe)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(5, 4, 4)).astype(np.float32)
    contact = rng.integers(0, 2, size=(5, 4, 2, 2)).astype(np.float32)
    force = rng.normal(size=(*contact.shape, 3)).astype(np.float32)
    mask = np.array([
        [True, True, True, True], [False, True, True, True],
        [False, False, True, True], [False, False, False, True],
        [False, False, False, False],
    ])
    batch = pad_probe_batch((features, contact, force, mask), 8)

    def objective(p, x, c, f, m):
        return future_patch_probe_loss(nnx.merge(graphdef, p)(x), c, f, valid_mask=m)[0]

    single = jax.jit(jax.value_and_grad(objective))
    expected_loss, expected_grad = single(params, *batch)
    distributed = jax.jit(
        jax.value_and_grad(objective),
        in_shardings=(parallel.replicated, parallel.data, parallel.data, parallel.data, parallel.data),
        out_shardings=(parallel.replicated, parallel.replicated),
    )
    replicated = parallel.replicate(params)
    loss, grad = distributed(replicated, *parallel.shard(batch))
    np.testing.assert_allclose(np.asarray(loss), np.asarray(expected_loss), rtol=2e-5, atol=2e-6)
    for actual, expected in zip(jax.tree.leaves(grad), jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-6)
    predict = _prediction_fn(graphdef, parallel)
    expected = nnx.merge(graphdef, params)(features)
    actual = predict(replicated, features)
    assert actual["force"].shape[0] == 5
    np.testing.assert_allclose(actual["force"], np.asarray(expected["force"]), rtol=2e-5, atol=2e-6)
    validate = _validation_stats_fn(graphdef, parallel, np.ones(3, dtype=np.float32))
    stats = validate(replicated, features, contact, force, mask)
    _, expected_stats = future_patch_probe_loss(expected, contact, force, valid_mask=mask)
    for name in stats:
        np.testing.assert_allclose(stats[name], np.asarray(expected_stats[name]), rtol=2e-5, atol=2e-6)
