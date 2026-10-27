"""Local data -> original XHand transforms -> real FAE smoke; tiny substitutes for expensive backbones."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import pyarrow as pa
import pyarrow.parquet as pq

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.policies import xhand_policy
from openpi.shared import normalize
from openpi.shared.effort_type import EffortType
from openpi.shared.future_patch_probe_utils import create_manifest
from openpi.shared.future_patch_probe_utils import write_json
from openpi.training import config as configs
from scripts import run_future_patch_probe as runner
from scripts.extract_future_patch_probe_real_model_test import TinyVision


class TinyTokenizer:
    def tokenize(self, prompt, state):
        assert isinstance(prompt, str)
        assert state is None
        return np.ones(4, dtype=np.int32), np.ones(4, dtype=bool)


@dataclasses.dataclass(frozen=True)
class TinyFactory:
    repo_id: str
    assets: configs.AssetsConfig
    state_delta_timestamps: tuple[int, ...]
    primary_image_key: str = "observation.images.cam_front"
    wrist_image_key: str = "observation.images.cam_right"
    extra_image_key: str = "observation.images.cam_left"

    def create(self, assets_dirs, model_config):
        del assets_dirs
        return configs.DataConfig(
            repo_id=self.repo_id,
            state_delta_timestamps=self.state_delta_timestamps,
            data_transforms=transforms.Group(
                inputs=[
                    xhand_policy.XHandTactileFlowInputs(
                        model_type=model_config.model_type,
                        tactile_mode="raw_force",
                        structured_tactile=True,
                        tactile_history_frames=10,
                        primary_image_key=self.primary_image_key,
                        wrist_image_key=self.wrist_image_key,
                        extra_image_key=self.extra_image_key,
                        future_flow_key=None,
                        future_wrist_flow_key=None,
                    ),
                ]
            ),
            model_transforms=transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(TinyTokenizer()),
                    transforms.PadStatesAndActions(model_config.action_dim),
                ]
            ),
        )


def write_tiny_dataset(root):
    (root / "meta").mkdir(parents=True)
    camera_keys = ["observation.images.cam_front", "observation.images.cam_right", "observation.images.cam_left"]
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "chunks_size": 1000,
        "total_episodes": 2,
        "features": {key: {"dtype": "video"} for key in camera_keys},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "wipe"}) + "\n")
    episodes = [{"episode_index": i, "tasks": ["wipe"], "length": 6} for i in range(2)]
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes))
    for episode in range(2):
        states = np.zeros((6, 52 + 5 * 384), dtype=np.float32)
        for finger in range(5):
            states[:, 52 + finger * 384 + 24] = np.arange(6) + 2
        table = pa.table(
            {
                "episode_index": [episode] * 6,
                "frame_index": list(range(6)),
                "timestamp": np.arange(6) / 15,
                "task_index": [0] * 6,
                "observation.state": states.tolist(),
            }
        )
        parquet = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet)
        for key in camera_keys:
            path = root / f"videos/chunk-000/{key}/episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 32))
            assert writer.isOpened()
            for frame in range(6):
                writer.write(np.full((32, 32, 3), frame * 20, dtype=np.uint8))
            writer.release()


def test_one_batch_check_from_v21_video_to_frozen_real_fae(tmp_path, monkeypatch, caplog):
    train_root, test_root = tmp_path / "train", tmp_path / "test"
    write_tiny_dataset(train_root)
    write_tiny_dataset(test_root)
    manifest_path = tmp_path / "splits.json"
    write_json(manifest_path, create_manifest(train_root, test_root, test_task_scope="common"))
    model_config = pi0_config.Pi0LatentFlowConfig(
        action_horizon=16,
        action_dim=32,
        structured_tactile=True,
        effort_type=EffortType.MOT,
        effort_dim=1800,
        tactile_points_per_finger=120,
        tactile_dim_per_finger=3,
        tactile_num_fingers=5,
        force_input_frames=10,
        future_tactile_segments=4,
        future_steps_per_segment=4,
        pool_tactile_history=True,
        cached_vlm_async_ae_enabled=True,
        cached_vlm_async_history_mode="pooled_current",
        tactile_patch_informed_tokenizer=True,
        use_future_flow=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        future_tactile_align_layer=1,
        dtype="float32",
    )
    monkeypatch.setattr(siglip, "Module", TinyVision)
    instance = model_config.create(jax.random.key(1))
    _, state = nnx.split(instance)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint / "params", {"params": state})
    norm_dir = checkpoint / "assets/toy"
    norm_dir.mkdir(parents=True)
    (norm_dir / "norm_stats.json").write_text(normalize.serialize_json({}))
    factory = TinyFactory(str(train_root), configs.AssetsConfig(), tuple(range(-9, 17)))
    monkeypatch.setattr(
        configs,
        "get_config",
        lambda _name: SimpleNamespace(data=factory, model=model_config, assets_dirs=Path("unused")),
    )
    monkeypatch.setattr(runner, "_training_data_loader", lambda: None)
    monkeypatch.setattr(runner, "_require_gpu", lambda _jax: None)  # CPU test; server requires GPU.
    args = argparse.Namespace(
        manifest=manifest_path,
        policy_params=checkpoint,
        norm_stats=None,
        config_name="tiny",
        batch_size=1,
        stride=4,
        shard_size=256,
        seed=42,
        num_steps=10,
        contact_threshold=1.0,
        contact_temperature=0.5,
        force_unit="dataset_native",
        cache_dir=tmp_path / "must_not_exist",
        smoke=True,
        splits=["train"],
        resume=False,
        num_workers=0,
    )
    caplog.set_level(logging.INFO)
    runner.extract(args)
    assert "SMOKE PASS" in caplog.text
    assert "(1, 4, 20, 64)" in caplog.text
    assert not args.cache_dir.exists()
