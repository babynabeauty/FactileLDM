"""Train the TouchWorld-inspired XHand adaptation; this is not official TouchWorld.

Three stages: observation reconstruction, nominal prediction, then tactile residual
correction. The last stage uses a frozen model's *sampled* nominal plan and the
observation actually available at the sampled execution offset. Future observations
are targets only. Train/validation splitting is by episode; test episodes are never
used for training or validation. Shared parquet partitions may still be read before
episode filtering. This implementation supports one CPU or CUDA process only.

Example (the explicit ID files contain JSON integer lists):
    PYTHONPATH=src python scripts/train_touchworld_lite.py \
      --repo-dir /data/xht --train-episode-ids /data/train_ids.json \
      --test-episode-ids /data/test_ids.json --image-key observation.images.wrist \
      --output-dir /checkpoints/touchworld_lite

For local v2.1 video data, the built-in decoder checks FPS/frame alignment and
uses a disk-backed RGB cache. --image-loader module:function overrides decoding.
There is deliberately no synthetic-data CLI fallback.
Resume with the same arguments plus --resume /checkpoints/touchworld_lite/last.pt.
Checkpoints are written at epoch boundaries; an interrupted epoch is replayed.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


STAGES = ("representation", "nominal", "residual")
IDENTITY = "TouchWorld-inspired XHand adaptation; not the official TouchWorld implementation"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: tuple[int, int, int] = (10, 50, 20)
    batch_size: int = 16
    learning_rates: tuple[float, float, float] = (1e-4, 1e-4, 1e-4)
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 0
    num_workers: int = 0
    device: str = "cuda:0"
    bf16: bool = False


def reject_distributed() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "This training entry point is verified for one process only. "
            "Do not launch it with multi-process torchrun: gradients would not be synchronized. "
            "Use one CUDA device; multi-GPU training is not implemented in this version."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_training_stats(episodes: list[Any]) -> dict[str, dict[str, list]]:
    """Fit per-coordinate statistics from real frames in TRAIN episodes only."""
    if not episodes:
        raise ValueError("No training episodes")
    result = {}
    for key in ("state", "actions", "tactile"):
        count, total, square = 0, None, None
        for episode in episodes:
            values = np.asarray(getattr(episode, key), dtype=np.float64)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Nonfinite/empty {key} in training episode {episode.episode_id}")
            summed = values.sum(axis=0)
            squared = np.square(values).sum(axis=0)
            total = summed if total is None else total + summed
            square = squared if square is None else square + squared
            count += len(values)
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 0.0)).clip(min=1e-6)
        result[key] = {"mean": mean.tolist(), "std": std.tolist()}
    return result


class LocalVideoDecoder:
    """Decode selected episode frames to a memory-mapped cache, not all-RGB RAM.

    Requires the explicit LeRobot v2.1 video_path template and episode-local
    timestamps. No nearest-timestamp replacement or fabricated frame is allowed.
    State/tactile episode arrays are still held in RAM by the data adapter.
    """

    def __init__(self, root, cache_dir, image_size=224):
        self.root, self.cache_dir = Path(root).resolve(), Path(cache_dir)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v2.1" or not self.info.get("video_path"):
            raise ValueError("Built-in video decoder requires local LeRobot v2.1 meta/info.json with video_path")
        self.fps = float(self.info["fps"])
        if self.fps <= 0 or image_size < 1:
            raise ValueError("Invalid video FPS or image size")
        self.image_size = image_size

    def __call__(self, episode_id, camera_key, frame_ids, timestamps):
        import cv2

        frames = np.asarray(frame_ids, dtype=np.int64)
        if timestamps is None or not np.allclose(timestamps, frames / self.fps, atol=1e-3, rtol=0):
            raise ValueError("Video decoder requires episode-local timestamps matching frame_index / metadata FPS")
        path = self.root / self.info["video_path"].format(
            episode_index=episode_id, episode_chunk=episode_id // int(self.info.get("chunks_size", 1000)),
            video_key=camera_key)
        stat = path.stat()
        signature = f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{self.image_size}:{self.fps}".encode() + frames.tobytes()
        key = hashlib.sha256(signature).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = self.cache_dir / f"{key}.npy"
        shape = (len(frames), self.image_size, self.image_size, 3)
        if cache.exists():
            array = np.load(cache, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != np.uint8:
                raise ValueError(f"Invalid RGB cache: {cache}")
            return array
        capture = cv2.VideoCapture(str(path))
        temporary = cache.with_suffix(".tmp.npy")
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot decode {path}")
            if not np.isclose(capture.get(cv2.CAP_PROP_FPS), self.fps, atol=0.05, rtol=0):
                raise ValueError(f"Video FPS differs from metadata: {path}")
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
            next_frame = 0
            for row, frame in enumerate(frames):
                while next_frame < frame:
                    if not capture.grab():
                        raise RuntimeError(f"Missing RGB frame {next_frame} in {path}")
                    next_frame += 1
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Missing RGB frame {frame} in {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                array[row] = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                next_frame += 1
            array.flush()
            del array
            os.replace(temporary, cache)
        finally:
            capture.release()
            if temporary.exists():
                temporary.unlink()
        return np.load(cache, mmap_mode="r", allow_pickle=False)


class TrainingWindows(Dataset):
    """Keep chunk-start nominal inputs separate from offset-time TRT inputs."""

    def __init__(self, windows, stats, image_key: str, *, stage: str, seed: int, training: bool):
        self.windows = windows
        self.stats = {k: {n: np.asarray(v, dtype=np.float32) for n, v in value.items()} for k, value in stats.items()}
        self.image_key = image_key
        self.stage = stage
        self.seed = seed
        self.training = training
        self.epoch = 0

    def __len__(self):
        return len(self.windows)

    def _normalize(self, key: str, value):
        value = np.asarray(value, dtype=np.float32)
        return (value - self.stats[key]["mean"]) / self.stats[key]["std"]

    def _image(self, images):
        image = np.asarray(images[self.image_key])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Image decoder must return RGB images [H,W,3]")
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 127.5 - 1.0
        elif np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1:
            image = image.astype(np.float32) * 2.0 - 1.0
        else:
            raise ValueError("Image decoder must return uint8 RGB or floating RGB in [0,1]")
        return torch.from_numpy(np.moveaxis(image, -1, 0).copy())

    def _history(self, sample):
        history = self._normalize("tactile", sample["tactile_history"])
        valid = np.asarray(sample["history_valid"], dtype=bool)
        history = np.where(valid[:, None, None, None], history, 0.0)
        return torch.from_numpy(history.copy())

    def __getitem__(self, index):
        start = self.windows.sample(index, offset=0)
        if not bool(start["current_valid"]):
            raise ValueError("Dataset returned an invalid chunk anchor")
        observed = start
        if self.stage == "residual":
            # Index/epoch keyed RNG is independent of DataLoader worker scheduling.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch if self.training else 0, index]))
            for offset in rng.permutation(self.windows.valid_offsets(index)):
                candidate = self.windows.sample(index, offset=int(offset))
                if bool(candidate["current_valid"]) and np.asarray(candidate["action_valid"], bool).any():
                    observed = candidate
                    break
            else:
                raise ValueError("No valid action/observation offset for residual training")
        return {
            "current_rgb": self._image(start["image"]),
            "current_tactile": self._history(start),
            "current_state": torch.from_numpy(self._normalize("state", start["chunk_state"]).copy()),
            "task_id": torch.tensor(int(start["task_id"]), dtype=torch.long),
            "actions": torch.from_numpy(self._normalize("actions", start["actions"]).copy()),
            "future_rgb": self._image(start["future_image"]),
            "future_tactile": torch.from_numpy(self._normalize("tactile", start["future_tactile"]).copy()),
            "future_valid_mask": torch.tensor(bool(start["future_valid"])),
            "action_valid_mask": torch.as_tensor(np.asarray(start["action_valid"], dtype=bool).copy()),
            "offset_tactile": self._history(observed),
            "offset_state": torch.from_numpy(self._normalize("state", observed["state"]).copy()),
            "offsets": torch.tensor(int(observed["offset"]), dtype=torch.long),
            "offset_action_valid_mask": torch.as_tensor(np.asarray(observed["action_valid"], dtype=bool).copy()),
        }


class StageObjective(nn.Module):
    def __init__(self, model: nn.Module, stage: str):
        super().__init__()
        self.model = model
        self.stage = stage

    def forward(self, batch):
        if self.stage == "representation":
            return self.model.representation_loss(batch["current_rgb"], batch["current_tactile"])
        if self.stage == "nominal":
            return self.model.nominal_loss(**{key: batch[key] for key in (
                "current_rgb", "current_tactile", "current_state", "task_id", "actions",
                "future_rgb", "future_tactile", "future_valid_mask", "action_valid_mask",
            )})
        if self.stage != "residual":
            raise ValueError(self.stage)
        # No actions, future_rgb or future_tactile may reach this call. The plan
        # is actual frozen-model inference, not teacher-forced one-step denoising.
        with torch.no_grad():
            plan = self.model.predict_nominal(
                current_rgb=batch["current_rgb"], current_tactile=batch["current_tactile"],
                current_state=batch["current_state"], task_id=batch["task_id"]
            )
        return self.model.residual_loss(
            plan=plan,
            current_tactile=batch["offset_tactile"],
            current_state=batch["offset_state"],
            offsets=batch["offsets"],
            target_actions=batch["actions"],
            action_valid_mask=batch["offset_action_valid_mask"],
        )


def _set_stage(model, stage: str) -> None:
    if stage == "nominal":
        model.freeze_observation_encoders()
    elif stage == "residual":
        model.freeze_nominal()
    elif stage != "representation":
        raise ValueError(stage)


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def _atomic_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _worker_seed(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(dataset, config, stage_index, epoch, *, training):
    dataset.epoch = epoch if training else 0
    generator = torch.Generator().manual_seed(config.seed + stage_index * 1_000_003 + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training, drop_last=False,
                      num_workers=config.num_workers, worker_init_fn=_worker_seed, generator=generator,
                      pin_memory=str(config.device).startswith("cuda"))


def _run_epoch(objective, loader, config, *, optimizer=None):
    training = optimizer is not None
    objective.train(training)
    device = torch.device(config.device)
    totals, count = {}, 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        size = len(batch["task_id"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if config.bf16 else nullcontext()
        with torch.set_grad_enabled(training), amp:
            metrics = objective(batch)
            loss = metrics["loss"]
        if loss.ndim or not torch.isfinite(loss):
            raise RuntimeError(f"{objective.stage} produced nonfinite/non-scalar loss")
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_((p for p in objective.parameters() if p.requires_grad),
                                    config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * (1 if key.startswith("valid_") else size)
        count += size
    if count == 0:
        raise ValueError("Empty train or validation DataLoader")
    return {key: value if key.startswith("valid_") else value / count for key, value in totals.items()}


def train_stages(model, train_windows, validation_windows, stats, image_key, output_dir,
                 config: TrainingConfig, metadata: dict, *, resume: str | None = None):
    """Train on explicit train/validation datasets. There is no test-data argument."""
    reject_distributed()
    if len(config.epochs) != 3 or any(x < 1 for x in config.epochs):
        raise ValueError("All three stages require at least one epoch")
    if config.batch_size < 1 or config.num_workers < 0:
        raise ValueError("Invalid batch size / worker count")
    if config.bf16 and not str(config.device).startswith("cuda"):
        raise ValueError("--bf16 is supported only on CUDA")
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu for a small verification run")
    if str(config.device).startswith("cuda"):
        requested = torch.device(config.device)
        torch.cuda.set_device(requested.index if requested.index is not None else 0)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Output directory is not empty; use --resume or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    model.to(config.device)
    # This identity checks ID splits, normalization, model shape and training plan.
    contract = {"identity": IDENTITY, "config": asdict(config), "metadata": metadata,
                "stats": stats, "image_key": image_key}
    contract_json = json.dumps(contract, sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(contract_json.encode()).hexdigest()
    saved = None
    start_stage, next_epoch, global_step = 0, 0, 0
    if resume:
        # Only load trusted checkpoints produced by this training job.
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved.get("contract_digest") != digest:
            raise ValueError("Resume contract differs (data split, stats, model or training settings)")
        if saved.get("kind") != "resume":
            raise ValueError("Resume from last.pt, not a weights-only stage checkpoint")
        model.load_state_dict(saved["model"])
        start_stage, next_epoch = saved["stage_index"], saved["next_epoch"]
        global_step = saved["global_step"]
        _restore_rng(saved["rng"])
    (output / "run_manifest.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")

    def save_resume(stage_index, epoch, optimizer, best_value):
        _atomic_save({"kind": "resume", "identity": IDENTITY, "contract_digest": digest,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                      "stage_index": stage_index, "next_epoch": epoch, "global_step": global_step,
                      "best_validation_loss": best_value, "rng": _rng_state()}, output / "last.pt")

    for stage_index in range(start_stage, len(STAGES)):
        stage = STAGES[stage_index]
        _set_stage(model, stage)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError(f"No trainable parameters in {stage}")
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rates[stage_index],
                                      weight_decay=config.weight_decay)
        epoch_start, best = 0, float("inf")
        if saved is not None and stage_index == start_stage:
            epoch_start = next_epoch
            best = saved["best_validation_loss"]
            if saved["optimizer"] is not None:
                optimizer.load_state_dict(saved["optimizer"])
        objective = StageObjective(model, stage)
        train_data = TrainingWindows(train_windows, stats, image_key, stage=stage, seed=config.seed, training=True)
        val_data = TrainingWindows(validation_windows, stats, image_key, stage=stage, seed=config.seed, training=False)
        for epoch in range(epoch_start, config.epochs[stage_index]):
            loader = _loader(train_data, config, stage_index, epoch, training=True)
            train_metrics = _run_epoch(objective, loader, config, optimizer=optimizer)
            global_step += len(loader)
            # Repeated validation uses the same inference noise and never consumes
            # the training RNG stream. Epochs remain comparable and resumable.
            devices = [torch.cuda.current_device()] if str(config.device).startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(config.seed + stage_index * 1009)
                val_metrics = _run_epoch(objective, _loader(val_data, config, stage_index, 0, training=False), config)
            record = {"stage": stage, "epoch": epoch, "global_step": global_step,
                      "loss_reduction": "sample-weighted mean of minibatch-normalized losses; not global masked MSE",
                      "train": train_metrics, "validation": val_metrics}
            print(json.dumps(record, allow_nan=False), flush=True)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                _atomic_save({"kind": "stage_best", "identity": IDENTITY, "stage": stage,
                              "contract_digest": digest, "model": model.state_dict(),
                              "validation_loss": best, "epoch": epoch}, output / f"{stage}_best.pt")
            save_resume(stage_index, epoch + 1, optimizer, best)
        best_path = output / f"{stage}_best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Stage selection checkpoint is missing: {best_path}")
        selected = torch.load(best_path, map_location=config.device, weights_only=False)
        if selected["contract_digest"] != digest:
            raise ValueError("Stage checkpoint belongs to a different run")
        model.load_state_dict(selected["model"])
        save_resume(stage_index + 1, 0, None, float("inf"))
        saved = None
    _atomic_save({"identity": IDENTITY, "model": model.state_dict(), "metadata": metadata,
                  "stats": stats, "image_key": image_key, "contract_digest": digest}, output / "model_final.pt")
    return model


def _read_ids(path):
    text = Path(path).read_text().strip()
    ids = json.loads(text) if text.startswith("[") else [int(x) for x in text.split()]
    if not ids or any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path} must contain unique integer episode IDs")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--test-repo-dir", help="Separate held-out LeRobot root, recorded for split provenance")
    parser.add_argument("--train-episode-ids", required=True)
    parser.add_argument("--test-episode-ids", required=True)
    parser.add_argument("--image-key", required=True)
    parser.add_argument("--image-loader", help="Optional module:function video decoder callback")
    parser.add_argument("--image-cache-dir", help="Disk-backed RGB cache; default is an output-directory sibling")
    parser.add_argument("--image-size", type=int, default=224, help="Built-in video decoder square RGB size")
    parser.add_argument("--allow-legacy-raw-layout", action="store_true",
                        help="Opt in to unnamed SDK raw-state offsets ONLY after confirming the original XHand layout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", nargs=3, type=int, default=(10, 50, 20))
    parser.add_argument("--learning-rates", nargs=3, type=float, default=(1e-4, 1e-4, 1e-4))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    args = parser.parse_args()
    reject_distributed()
    from openpi.baselines.touchworld_data import EpisodeWindowDataset, load_lerobot_episodes, split_training_episodes
    from openpi.baselines.touchworld_lite import TouchWorldLite, TouchWorldLiteConfig

    split = split_training_episodes(_read_ids(args.train_episode_ids), _read_ids(args.test_episode_ids),
                                    val_fraction=args.val_fraction, seed=args.seed,
                                    train_dataset_id=str(Path(args.repo_dir).resolve()),
                                    test_dataset_id=str(Path(args.test_repo_dir or args.repo_dir).resolve()))
    image_loader = None
    if args.image_loader:
        module, name = args.image_loader.split(":", 1)
        image_loader = getattr(importlib.import_module(module), name)
    else:
        info_path = Path(args.repo_dir) / "meta/info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        if info.get("features", {}).get(args.image_key, {}).get("dtype") == "video":
            cache_dir = args.image_cache_dir or str(Path(args.output_dir).with_name(Path(args.output_dir).name + "_rgb_cache"))
            image_loader = LocalVideoDecoder(args.repo_dir, cache_dir, args.image_size)
            if args.num_workers:
                raise ValueError("Use --num-workers 0 with the initial disk-backed video adapter; worker-safe lazy loading is not implemented")
            print("RGB frames use a disk-backed cache; selected state/tactile episodes are still loaded into RAM.", flush=True)
    # The held-out test IDs are recorded for provenance, never requested from the loader.
    train_episodes = load_lerobot_episodes(args.repo_dir, split.train, image_keys=(args.image_key,), image_loader=image_loader,
                                          allow_legacy_raw_layout=args.allow_legacy_raw_layout)
    val_episodes = load_lerobot_episodes(args.repo_dir, split.validation, image_keys=(args.image_key,), image_loader=image_loader,
                                        allow_legacy_raw_layout=args.allow_legacy_raw_layout)
    task_ids = sorted({episode.task_id for episode in train_episodes})
    if any(episode.task_id not in task_ids for episode in val_episodes):
        raise ValueError("Validation contains a task absent from training; choose a task-stratified ID split")
    num_tasks = max(task_ids) + 1
    model_config = TouchWorldLiteConfig(num_tasks=num_tasks, action_horizon=args.horizon, window_size=args.window_size,
                                        history_frames=args.history, width=args.width, latent_dim=args.latent_dim)
    seed_everything(args.seed)
    model = TouchWorldLite(model_config)
    stats = fit_training_stats(train_episodes)
    windows_args = {"horizon": args.horizon, "history": args.history, "stride": args.stride}
    train_windows = EpisodeWindowDataset(train_episodes, **windows_args)
    val_windows = EpisodeWindowDataset(val_episodes, **windows_args)
    metadata = {"repo_dir": str(Path(args.repo_dir).resolve()), "train_ids": list(split.train),
                "validation_ids": list(split.validation), "test_ids": list(split.test),
                "test_repo_dir": split.test_dataset_id,
                "model_config": asdict(model_config), "stride": args.stride, "task_ids": task_ids,
                "image_loader": args.image_loader or ("local_v2.1_video" if image_loader else "numeric_parquet"),
                "image_size": args.image_size, "allow_legacy_raw_layout": args.allow_legacy_raw_layout}
    config = TrainingConfig(epochs=tuple(args.epochs), batch_size=args.batch_size,
                            learning_rates=tuple(args.learning_rates), seed=args.seed,
                            num_workers=args.num_workers, device=args.device, bf16=args.bf16)
    train_stages(model, train_windows, val_windows, stats, args.image_key, args.output_dir, config, metadata,
                 resume=args.resume)


if __name__ == "__main__":
    main()
