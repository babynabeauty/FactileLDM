"""Synthetic-only tests of training causality and epoch-boundary recovery."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from openpi.baselines.touchworld_data import EpisodeData, EpisodeWindowDataset
from openpi.baselines.touchworld_lite import TouchWorldLite, TouchWorldLiteConfig
import train_touchworld_lite as trainer


def _episode(identity, length=6):
    rng = np.random.default_rng(identity)
    return EpisodeData(identity, 0, {"rgb": rng.integers(0, 256, (length, 16, 16, 3), dtype=np.uint8)},
                       rng.normal(size=(length, 5, 120, 3)).astype(np.float32),
                       rng.normal(size=(length, 18)).astype(np.float32),
                       rng.normal(size=(length, 18)).astype(np.float32))


def _model():
    torch.manual_seed(23)
    return TouchWorldLite(TouchWorldLiteConfig(num_tasks=1, action_horizon=4, window_size=2,
                                             history_frames=3, width=16, latent_dim=8, num_heads=2,
                                             image_grid=2, goal_layers=1, nominal_layers=1, residual_layers=1))


def test_offset_history_is_new_feedback_and_future_never_enters_nominal():
    episode = _episode(1)
    windows = EpisodeWindowDataset([episode], horizon=4, history=3)
    stats = trainer.fit_training_stats([episode])
    data = trainer.TrainingWindows(windows, stats, "rgb", stage="residual", seed=0, training=True)
    example = data[0]
    offset = int(example["offsets"])
    assert offset > 0
    expected = (episode.tactile[offset] - np.array(stats["tactile"]["mean"])) / np.array(stats["tactile"]["std"])
    np.testing.assert_allclose(example["offset_tactile"][-1], expected, atol=1e-6)
    assert not torch.equal(example["current_tactile"], example["offset_tactile"])

    class Spy(torch.nn.Module):
        def predict_nominal(self, *, current_rgb, current_tactile, current_state, task_id):
            assert not torch.is_grad_enabled()
            self.seen = current_tactile
            return "actual_inference_plan"

        def residual_loss(self, *, plan, current_tactile, current_state, offsets, target_actions, action_valid_mask):
            assert plan == "actual_inference_plan"
            assert not torch.equal(current_tactile, self.seen)
            assert int(offsets[0]) == offset
            return {"loss": torch.tensor(0.0)}

    batch = {k: v.unsqueeze(0) for k, v in example.items()}
    batch["future_rgb"].fill_(float("nan"))
    batch["future_tactile"].fill_(float("nan"))
    assert trainer.StageObjective(Spy(), "residual")(batch)["loss"] == 0


def test_three_stages_and_interrupted_epoch_boundary_resume_match(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    training = [_episode(1), _episode(2)]
    validation = [_episode(3)]
    train = EpisodeWindowDataset(training, horizon=4, history=3, stride=3)
    val = EpisodeWindowDataset(validation, horizon=4, history=3, stride=3)
    stats = trainer.fit_training_stats(training)
    config = trainer.TrainingConfig(epochs=(1, 1, 1), batch_size=4, device="cpu", seed=7)
    metadata = {"train_ids": [1, 2], "validation_ids": [3], "test_ids": [100],
                "model_config": asdict(_model().config)}
    uninterrupted = trainer.train_stages(_model(), train, val, stats, "rgb", tmp_path / "complete", config, metadata)
    original_save = trainer._atomic_save

    def interrupt_after_first_epoch(value, path):
        original_save(value, path)
        if path.name == "last.pt" and value["stage_index"] == 0 and value["next_epoch"] == 1:
            raise InterruptedError("simulated interruption after committed epoch")

    monkeypatch.setattr(trainer, "_atomic_save", interrupt_after_first_epoch)
    with pytest.raises(InterruptedError):
        trainer.train_stages(_model(), train, val, stats, "rgb", tmp_path / "resumed", config, metadata)
    monkeypatch.setattr(trainer, "_atomic_save", original_save)
    resumed = trainer.train_stages(_model(), train, val, stats, "rgb", tmp_path / "resumed", config, metadata,
                                   resume=str(tmp_path / "resumed" / "last.pt"))
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, atol=0, rtol=0)
    state = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert state["stage_index"] == 3
    assert state["optimizer"] is None
    assert all((tmp_path / "resumed" / f"{stage}_best.pt").is_file() for stage in trainer.STAGES)
    assert (tmp_path / "resumed" / "model_final.pt").is_file()


def test_reject_multi_process_instead_of_silent_unsynchronized_training(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="not implemented"):
        trainer.reject_distributed()


def test_local_v21_parquet_mp4_three_stage_cli_and_held_out_namespace(tmp_path):
    """Real Parquet/MP4 I/O with synthetic content; no user data or robot used."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cv2 = pytest.importorskip("cv2")
    root = tmp_path / "train_repo"
    camera = "observation.images.cam_front"
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "fps": 10, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {camera: {"dtype": "video", "shape": [16, 16, 3]},
                         "observation.state": {"dtype": "float32", "shape": [18]},
                         "action": {"dtype": "float32", "shape": [18]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    data_dir = root / "data/chunk-000"
    video_dir = root / "videos/chunk-000" / camera
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    length = 6
    for episode_id in range(4):
        episode = _episode(episode_id + 10, length=length)
        table = pa.table({"episode_index": [episode_id] * length, "frame_index": list(range(length)),
                          "timestamp": (np.arange(length) / 10).tolist(), "task_index": [3] * length,
                          "observation.state": episode.state.tolist(), "action": episode.actions.tolist(),
                          "observation.tactile": episode.tactile.tolist()})
        pq.write_table(table, data_dir / f"episode_{episode_id:06d}.parquet")
        writer = cv2.VideoWriter(str(video_dir / f"episode_{episode_id:06d}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16))
        if not writer.isOpened():
            pytest.skip("OpenCV build cannot encode a synthetic MP4 fixture")
        for frame in range(length):
            # A changing, almost uniform RGB image also exposes wrong frame indices.
            rgb = np.full((16, 16, 3), [20 + 30 * frame, 50 + episode_id, 80], dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        writer.release()

    cache = tmp_path / "rgb_cache"
    decoder = trainer.LocalVideoDecoder(root, cache, image_size=16)
    frames = np.array([0, 2, 5])
    decoded = decoder(0, camera, frames, frames / 10)
    assert isinstance(decoded, np.memmap)
    assert decoded.shape == (3, 16, 16, 3)
    np.testing.assert_allclose(decoded[:, :, :, 0].mean(axis=(1, 2)), [20, 80, 170], atol=8)
    before = sorted(cache.glob("*.npy"))
    cached = decoder(0, camera, frames, frames / 10)
    np.testing.assert_array_equal(cached, decoded)
    assert sorted(cache.glob("*.npy")) == before
    with pytest.raises(ValueError, match="timestamps"):
        decoder(0, camera, frames, frames / 10 + 0.1)
    wrong_fps = trainer.LocalVideoDecoder(root, cache, image_size=16)
    wrong_fps.fps = 20
    with pytest.raises(ValueError, match="Video FPS"):
        wrong_fps(0, camera, frames, frames / 20)

    # IDs overlap across roots. Corrupt test data ensures any accidental test load
    # fails loudly; the CLI is allowed only to record its namespace/ID manifest.
    test_root = tmp_path / "held_out_repo"
    (test_root / "data").mkdir(parents=True)
    (test_root / "data/episode_000000.parquet").write_bytes(b"DO NOT READ TEST DATA")
    train_ids, test_ids = tmp_path / "train_ids.json", tmp_path / "test_ids.json"
    train_ids.write_text("[0, 1, 2, 3]")
    test_ids.write_text("[0]")
    output = tmp_path / "checkpoint"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WORLD_SIZE="1")
    command = [sys.executable, str(Path(trainer.__file__).resolve()), "--repo-dir", str(root),
               "--test-repo-dir", str(test_root), "--train-episode-ids", str(train_ids),
               "--test-episode-ids", str(test_ids), "--image-key", camera,
               "--image-cache-dir", str(cache), "--image-size", "16", "--output-dir", str(output),
               "--epochs", "1", "1", "1", "--val-fraction", "0.25", "--device", "cpu",
               "--horizon", "4", "--window-size", "2", "--history", "3", "--stride", "3",
               "--width", "16", "--latent-dim", "8", "--batch-size", "4"]
    completed = subprocess.run(command, cwd=Path(trainer.__file__).resolve().parents[1], env=env,
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "run_manifest.json").read_text())
    metadata = manifest["metadata"]
    assert metadata["test_ids"] == [0]
    assert metadata["test_repo_dir"] == str(test_root.resolve())
    assert not set(metadata["train_ids"]) & set(metadata["validation_ids"])
    assert set(metadata["train_ids"]) | set(metadata["validation_ids"]) == {0, 1, 2, 3}
    assert metadata["task_ids"] == [3]
    assert metadata["model_config"]["num_tasks"] == 4
    assert metadata["image_loader"] == "local_v2.1_video"
    assert metadata["allow_legacy_raw_layout"] is False
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [record["stage"] for record in records] == list(trainer.STAGES)
    assert (output / "model_final.pt").is_file()
