"""Synthetic-cache integration tests; no policy checkpoint, robot, or real training data."""

import argparse
import importlib

import numpy as np
import pytest

from openpi.shared.future_patch_probe_utils import fingerprint
from openpi.shared.future_patch_probe_utils import read_json
from openpi.shared.future_patch_probe_utils import write_json
from scripts.run_future_patch_probe import _require_gpu
from scripts.run_future_patch_probe import _training_data_loader
from scripts.run_future_patch_probe import evaluate
from scripts.run_future_patch_probe import train


def test_only_missing_lerobot_api_uses_local_reader(monkeypatch):
    original = object()
    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", lambda _name: original)
        assert _training_data_loader() is original

    def missing_lerobot(_name):
        raise ModuleNotFoundError("missing", name="lerobot.common")

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", missing_lerobot)
        assert _training_data_loader() is None

    def missing_other_dependency(_name):
        raise ModuleNotFoundError("missing", name="some_other_dependency")

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", missing_other_dependency)
        with pytest.raises(ModuleNotFoundError, match="missing"):
            _training_data_loader()


def test_gpu_check_rejects_silent_cpu_fallback():
    class FakeJax:
        def default_backend(self):
            return "cpu"

        def devices(self):
            return ["cpu:0"]

    with pytest.raises(RuntimeError, match="CUDA-enabled JAX GPU"):
        _require_gpu(FakeJax())


def synthetic_cache(root, split, protocol):
    rng = np.random.default_rng(3)
    count, horizon, fingers, patches, width = 8, 4, 2, 2, 4
    directory = root / split
    directory.mkdir(parents=True)
    contact = rng.integers(0, 2, size=(count, horizon, fingers, patches)).astype(np.float32)
    force = rng.normal(size=(*contact.shape, 3)).astype(np.float32) * contact[..., None]
    valid = np.ones((count, 2, horizon), dtype=bool)
    valid[:, 1, :2] = False
    # Matching offset-0 inputs are a necessary identity check for fixed/refined.
    refined = rng.normal(size=(count, 2, 4, width)).astype(np.float16)
    ids = np.repeat([0, 1] if split != "val" else [2, 3], count // 2)
    arrays = {
        "refined_hidden": refined,
        "fixed_hidden": refined[:, 0],
        "history_tokens": refined,
        "target_contact": contact,
        "target_force": force,
        "valid_mask": valid,
        "current_contact": np.zeros((count, 2, fingers, patches), dtype=np.float32),
        "current_force": np.zeros((count, 2, fingers, patches, 3), dtype=np.float32),
        "offsets": np.array([0, 2]),
        "future_frame_offsets": np.arange(1, 5),
        "episode_id": ids,
        "frame_index": np.tile(np.arange(count // 2), 2),
        # Different dataset labels must not require cross-split task matching.
        "task_id": np.full(count, 0 if split == "test" else 1, dtype=np.int32),
    }
    np.savez(directory / "shard_00000.npz", **arrays)
    index = {
        "complete": True,
        "split": split,
        "model": {"segments": 2, "fingers": fingers, "steps_per_segment": 2, "patches": patches, "hidden_width": width},
        "task_names": ["test_toy", "train_toy"],
        "protocol_sha256": fingerprint(protocol),
        "episodes": np.unique(ids).tolist(),
        "samples": count,
        "shards": [{"file": "shard_00000.npz", "samples": count}],
    }
    write_json(directory / "index.json", index)


@pytest.mark.parametrize(("source", "decoder_type"), [("fae", "linear"), ("history", "mlp")])
def test_cached_training_checkpoint_reload_and_heldout_metrics(tmp_path, source, decoder_type):
    cache_dir = tmp_path / "cache"
    protocol = {"force_unit": "synthetic_test_units", "manifest_sha256": "toy", "policy_params": "NOT_LOADED"}
    write_json(cache_dir / "protocol.json", protocol)
    synthetic_cache(cache_dir, "train", protocol)
    synthetic_cache(cache_dir, "val", protocol)
    output = tmp_path / "probe"
    # The test cache does not exist during fitting or threshold selection.
    train(
        argparse.Namespace(
            cache_dir=cache_dir,
            output_dir=output,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            source=source,
            decoder_type=decoder_type,
            hidden_dim=8,
            seed=42,
        )
    )
    metadata = read_json(output / "best_probe.json")
    assert metadata["policy_frozen"]
    assert metadata["threshold_selected_on"].startswith("validation")
    assert set(metadata["train_episodes"]).isdisjoint(metadata["val_episodes"])
    history = read_json(output / "training_history.json")
    snapshots = sorted((output / "checkpoints").glob("epoch_*.npz"))
    assert len(snapshots) == 2
    for epoch, snapshot in enumerate(snapshots, start=1):
        saved = read_json(snapshot.with_suffix(".json"))
        assert saved["epoch"] == epoch
        assert saved["global_step"] == epoch * 2
        assert saved["epoch_steps"] == 2
        assert saved["val_loss"] == history[epoch - 1]["val_loss"]
        assert not saved["training_complete"]
    chosen = snapshots[metadata["best_epoch"] - 1]
    with np.load(chosen, allow_pickle=False) as saved, np.load(output / "best_probe.npz", allow_pickle=False) as best:
        assert set(saved.files) == set(best.files)
        for name in best.files:
            np.testing.assert_array_equal(saved[name], best[name])
    synthetic_cache(cache_dir, "test", protocol)
    # Intermediate weights have no validation-selected threshold and must not
    # be treated as final test-ready checkpoints.
    with pytest.raises(ValueError, match="threshold selection is incomplete"):
        evaluate(
            argparse.Namespace(
                cache_dir=cache_dir,
                probe_checkpoint=snapshots[0],
                output_dir=output,
                batch_size=4,
                bootstrap_samples=5,
                seed=42,
            )
        )
    evaluate(
        argparse.Namespace(
            cache_dir=cache_dir,
            probe_checkpoint=output / "best_probe.npz",
            output_dir=output,
            batch_size=4,
            bootstrap_samples=5,
            seed=42,
        )
    )
    result = read_json(output / "test_metrics.json")
    assert result["test_anchors"] == 8
    rows = result["metrics"]
    assert any(r["mode"] == "persistence" for r in rows)
    assert {r["task"] for r in rows} == {"test_toy", "__macro__"}
    assert all(r["contact_ap_tasks"] == 1 for r in rows if r["task"] == "__macro__")
    if source == "fae":
        initial = {r["mode"]: r for r in rows if r["task"] == "test_toy" and r["scope"] == "offset_0"}
        assert initial["fixed"]["contact_ap"] == initial["refined"]["contact_ap"]
        assert initial["fixed"]["active_force_rmse"] == initial["refined"]["active_force_rmse"]
    assert (output / "test_metrics.csv").is_file()
