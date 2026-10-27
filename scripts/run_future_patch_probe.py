#!/usr/bin/env python3
"""Prepare -> extract frozen FAE -> train readout -> evaluate on held-out episodes.

Only ``extract`` imports the policy/video/Parquet stack. ``train`` and ``evaluate``
consume immutable feature caches and cannot update or even load the policy.
"""

# Lazy imports isolate CPU-only preparation and cached readout fitting from the policy/data stack.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import dataclasses
import hashlib
import importlib
import logging
from pathlib import Path
import time

import numpy as np

from openpi.shared.future_patch_probe_utils import MetricGroup
from openpi.shared.future_patch_probe_utils import check_parameter_tree
from openpi.shared.future_patch_probe_utils import create_manifest
from openpi.shared.future_patch_probe_utils import dataset_metadata
from openpi.shared.future_patch_probe_utils import fingerprint
from openpi.shared.future_patch_probe_utils import future_validity
from openpi.shared.future_patch_probe_utils import paired_episode_bootstrap
from openpi.shared.future_patch_probe_utils import probe_features
from openpi.shared.future_patch_probe_utils import read_json
from openpi.shared.future_patch_probe_utils import summarize_groups
from openpi.shared.future_patch_probe_utils import training_batches
from openpi.shared.future_patch_probe_utils import training_force_scale
from openpi.shared.future_patch_probe_utils import write_json
from openpi.shared.probe_parallel import ProbeParallel
from openpi.shared.probe_parallel import pad_array
from openpi.shared.probe_parallel import pad_probe_batch
from openpi.shared.probe_parallel import prefetch_batches
from openpi.shared.probe_parallel import read_probe_cache


def prepare(args):
    manifest = create_manifest(
        args.train_root, args.test_root, val_ratio=args.val_ratio, seed=args.seed, test_task_scope=args.test_task_scope
    )
    if args.manifest.exists() and read_json(args.manifest) != manifest:
        raise FileExistsError("Manifest exists with different splits. Use a new output path.")
    write_json(args.manifest, manifest)
    for name, split in manifest["splits"].items():
        selected = set(split["episodes"])
        tasks = sorted(
            {row["task"] for row in manifest["datasets"][split["dataset"]]["episodes"] if row["episode"] in selected}
        )
        logging.info("%s: %d episodes, task labels=%s", name, len(selected), tasks)
    if manifest.get("excluded_test_episodes"):
        logging.info(
            "Excluded %d test episodes outside the shared task labels.", len(manifest["excluded_test_episodes"])
        )


class ProbeSampleDataset:
    """Keep raw label units separate from the exact policy input transforms."""

    def __init__(self, dataset, indices, episodes, frames, episode_info, tasks, data_config, horizon, history_length):
        from openpi import transforms

        self.dataset, self.indices = dataset, indices
        self.episodes, self.frames = episodes, frames
        self.episode_info, self.tasks = episode_info, tasks
        self.horizon, self.history_length = horizon, history_length
        self.before_norm = transforms.compose(
            [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
        )
        self.after_norm = transforms.compose(
            [
                transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        episode, frame = int(self.episodes[row]), int(self.frames[row])
        raw = self.dataset[row]
        if (
            int(np.asarray(raw["episode_index"]).item()) != episode
            or int(np.asarray(raw["frame_index"]).item()) != frame
        ):
            raise ValueError("LeRobot sample metadata disagrees with cache manifest.")
        info = self.episode_info[episode]
        valid = future_validity(frame, info["length"], self.horizon)
        padding = raw.get("observation.state_is_pad")
        if padding is not None:
            padding = np.asarray(padding, dtype=bool)
            if padding.shape != (self.history_length + self.horizon,):
                raise ValueError("Unexpected temporal state padding mask.")
            valid &= ~padding[self.history_length :]
        inputs = self.before_norm(raw)
        label_effort = np.asarray(inputs["effort"], dtype=np.float32).copy()
        inputs = self.after_norm(inputs)
        # Construct the Observation after collation: single-frame image masks
        # are NumPy scalars, whereas Observation expects batched arrays. Pass
        # no ground-truth actions or future auxiliary fields to the policy.
        fields = ("image", "image_mask", "state", "effort", "tokenized_prompt", "tokenized_prompt_mask")
        observation = {key: inputs[key] for key in fields if key in inputs}
        return {
            "observation": observation,
            "label_effort": label_effort,
            "valid_future_frames": valid,
            "episode_id": np.int32(episode),
            "frame_index": np.int32(frame),
            "task_id": np.int32(self.tasks.index(info["task"])),
        }


def _collate(items):
    import jax

    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)


def _checkpoint_and_norm(args):
    supplied = args.policy_params.expanduser().resolve()
    params = supplied / "params" if (supplied / "params").is_dir() else supplied
    if not params.is_dir():
        raise FileNotFoundError(f"Checkpoint params directory not found: {params}")
    root = params.parent if params.name == "params" else supplied
    norm = args.norm_stats
    if norm is None:
        candidates = list((root / "assets").glob("*/norm_stats.json"))
        if len(candidates) != 1:
            raise ValueError("Pass --norm-stats pointing to the ORIGINAL policy training norm_stats.json.")
        norm = candidates[0]
    norm = norm.expanduser().resolve()
    if not norm.is_file():
        raise FileNotFoundError(norm)
    return params, norm


def _training_data_loader():
    """Prefer the training reader, falling back only when its LeRobot API is absent."""
    try:
        return importlib.import_module("openpi.training.data_loader")
    except ModuleNotFoundError as exc:
        if exc.name == "lerobot" or (exc.name and exc.name.startswith("lerobot.common")):
            logging.warning("Training LeRobot reader unavailable (%s); using the checked v2.1 local reader.", exc.name)
            return None
        raise


def _require_gpu(jax_module):
    logging.info("JAX backend: %s; devices: %s", jax_module.default_backend(), jax_module.devices())
    if jax_module.default_backend() != "gpu":
        raise RuntimeError("FAE extraction requires a CUDA-enabled JAX GPU; check the FactileLDM environment.")


def extract(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import torch

    from openpi.models import model as policy_model
    from openpi.models import model_tavla
    from openpi.shared import normalize
    from openpi.shared.future_patch_probe_data import LocalProbeDataset
    from openpi.shared.future_patch_probe_data import require_video_decoder
    from openpi.training import config as configs
    from scripts.extract_future_patch_probe import extract_probe_batch
    from scripts.extract_future_patch_probe import probe_metadata
    from scripts.extract_future_patch_probe import validate_probe_model

    manifest = read_json(args.manifest)
    params_path, norm_path = _checkpoint_and_norm(args)
    config = configs.get_config(args.config_name)
    training_loader = _training_data_loader()
    reader_name = "training_lerobot" if training_loader is not None else "local_lerobot_v2.1_parquet_pyav_rgb"
    if training_loader is None:
        require_video_decoder()
    if args.batch_size <= 0 or args.stride <= 0 or args.shard_size <= 0:
        raise ValueError("Batch size, cache stride, and shard size must be positive.")
    expected_offsets = tuple(range(1 - config.model.force_input_frames, config.model.action_horizon + 1))
    if tuple(config.data.state_delta_timestamps) != expected_offsets:
        raise ValueError("Policy state offsets must be contiguous history through current, then t+1..t+H.")

    _require_gpu(jax)
    parallel = ProbeParallel(getattr(args, "num_devices", 1), args.batch_size)
    logging.info("Data reader: %s", reader_name)
    logging.info("Strictly restoring frozen policy from %s", params_path)
    shaped = nnx.eval_shape(config.model.create, jax.random.key(args.seed))
    graphdef, state = nnx.split(shaped)
    loaded = policy_model.restore_params(params_path, restore_type=np.ndarray)
    check_parameter_tree(state.to_pure_dict(), loaded)

    # No WeightLoader merge/initialization fallback is allowed in a readout experiment.
    def select_tree(expected, actual):
        if isinstance(expected, dict):
            return {key: select_tree(value, actual[key]) for key, value in expected.items()}
        value = actual
        dtype = jnp.bfloat16 if np.issubdtype(value.dtype, np.floating) else value.dtype
        return jnp.asarray(value, dtype=dtype)

    state.replace_by_pure_dict(select_tree(state.to_pure_dict(), loaded))
    del loaded, shaped
    model = nnx.merge(graphdef, state)
    model.eval()
    validate_probe_model(model)
    metadata = probe_metadata(model, num_steps=args.num_steps)
    graphdef, state = nnx.split(model)
    action_horizon, history_length = model.action_horizon, model.force_input_frames
    del model
    state = parallel.replicate(state)
    # Reuse the checkpoint's statistics for train/val/test; never fit test statistics.
    norm_stats = normalize.deserialize_json(norm_path.read_text())
    checkpoint_files = [
        (str(p.relative_to(params_path)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(params_path.rglob("*"))
        if p.is_file()
    ]
    protocol = {
        "config_name": args.config_name,
        "model_config": str(config.model),
        "policy_params": str(params_path),
        "checkpoint_inventory_sha256": fingerprint(checkpoint_files),
        "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "norm_stats_path": str(norm_path),
        "manifest_sha256": fingerprint(manifest),
        "contact_threshold": args.contact_threshold,
        "contact_temperature": args.contact_temperature,
        "force_unit": args.force_unit,
        "stride": args.stride,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "extraction_batch_size": args.batch_size,
        "feature_dtype": "float16",
        "dataset_reader": reader_name,
        "model": metadata,
    }
    protocol["extraction_num_devices"] = parallel.num_devices
    protocol["extraction_execution"] = "global_batch_sharded_padded_v1"
    if not args.smoke:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = args.cache_dir / "protocol.json"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Cache protocol differs. Use a new cache directory; do not mix checkpoints/units/splits.")
        write_json(protocol_path, protocol)

    def forward(policy_state, observation, labels, valid, rng):
        frozen = nnx.merge(graphdef, policy_state)
        return extract_probe_batch(
            frozen,
            rng,
            observation,
            valid,
            num_steps=args.num_steps,
            label_effort=labels,
            label_contact_threshold=args.contact_threshold,
            label_contact_temperature=args.contact_temperature,
        )

    record_shardings = {
        key: parallel.data
        for key in (
            "refined_hidden", "fixed_hidden", "history_tokens", "target_contact",
            "target_force", "current_contact", "current_force", "valid_mask",
        )
    }
    record_shardings.update(offsets=parallel.replicated, future_frame_offsets=parallel.replicated)
    forward = jax.jit(
        forward,
        in_shardings=(parallel.replicated, parallel.data, parallel.data, parallel.data, parallel.replicated),
        out_shardings=record_shardings,
    )

    for split_name in args.splits:
        split = manifest["splits"][split_name]
        dataset_info = manifest["datasets"][split["dataset"]]
        current_info = dataset_metadata(dataset_info["root"])
        if current_info["metadata_sha256"] != dataset_info["metadata_sha256"]:
            raise ValueError("Dataset metadata changed since prepare. Create a new manifest.")
        if not np.isclose(dataset_info["fps"], metadata["tactile_sample_hz_config"]):
            raise ValueError("Dataset FPS differs from the checkpoint's tactile_sample_hz.")
        destination = args.cache_dir / split_name
        index_path = destination / "index.json"
        if not args.smoke and index_path.exists():
            index = read_json(index_path)
            if index.get("complete") and args.resume:
                if not all((destination / s["file"]).is_file() for s in index["shards"]):
                    raise ValueError("Completed cache index refers to missing shards.")
                logging.info("Reusing completed %s cache", split_name)
                continue
            raise FileExistsError(f"Cache index already exists: {index_path}")
        if not args.smoke:
            destination.mkdir(parents=True, exist_ok=True)
        factory = dataclasses.replace(
            config.data,
            repo_id=dataset_info["root"],
            assets=configs.AssetsConfig(assets_dir=str(norm_path.parent.parent), asset_id=norm_path.parent.name),
        )
        data_config = factory.create(config.assets_dirs, config.model)
        data_config = dataclasses.replace(
            data_config, norm_stats=norm_stats, filter_dict_path=None, max_episodes=None, rcs_sample_enable=False
        )
        if training_loader is not None:
            dataset = training_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
            episodes, frames = training_loader._frame_metadata(dataset)  # noqa: SLF001
        else:
            features = read_json(Path(dataset_info["root"]) / "meta/info.json")["features"]
            camera_keys = [config.data.primary_image_key, config.data.wrist_image_key]
            if config.data.extra_image_key in features:
                camera_keys.append(config.data.extra_image_key)
            if any(key not in features or features[key].get("dtype") != "video" for key in camera_keys):
                raise ValueError("Required policy cameras must have per-episode video features in this v2.1 dataset.")
            dataset = LocalProbeDataset(
                dataset_info["root"],
                episodes=dataset_info["episodes"],
                state_offsets=data_config.state_delta_timestamps,
                image_keys=camera_keys,
            )
            episodes, frames = dataset.episode_ids, dataset.frame_ids
        info_by_episode = {r["episode"]: {**r, "fps": dataset_info["fps"]} for r in dataset_info["episodes"]}
        selected = []
        for episode in split["episodes"]:
            rows = np.flatnonzero(episodes == episode)
            rows = rows[np.argsort(frames[rows])]
            length = info_by_episode[episode]["length"]
            if not np.array_equal(frames[rows], np.arange(length)):
                raise ValueError(f"Episode {episode} has missing/duplicate/nonzero-based frames.")
            selected.extend(rows[(frames[rows] % args.stride == 0) & (frames[rows] + 1 < length)].tolist())
        if args.smoke:
            selected = selected[: args.batch_size]
        wrapped = ProbeSampleDataset(
            dataset,
            selected,
            episodes,
            frames,
            info_by_episode,
            manifest["task_names"],
            data_config,
            action_horizon,
            history_length,
        )
        loader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=_collate,
            **({"multiprocessing_context": "spawn"} if args.num_workers > 0 else {}),
        )
        index = {
            "version": 1,
            "split": split_name,
            "complete": False,
            "model": metadata,
            "task_names": manifest["task_names"],
            "protocol_sha256": fingerprint(protocol),
            "episodes": split["episodes"],
            "shards": [],
            "samples": 0,
        }
        pending, pending_size = [], 0

        def flush(index=index, destination=destination):
            nonlocal pending, pending_size
            if not pending:
                return
            arrays = {
                k: pending[0][k]
                if k in ("offsets", "future_frame_offsets")
                else np.concatenate([p[k] for p in pending])
                for k in pending[0]
            }
            filename = f"shard_{len(index['shards']):05d}.npz"
            path = destination / filename
            if path.exists() and not args.resume:
                raise FileExistsError(f"Partial cache exists: {path}. Use --resume or a new directory.")
            with path.with_suffix(".tmp").open("wb") as stream:
                np.savez(stream, **arrays)
            path.with_suffix(".tmp").replace(path)
            index["shards"].append({"file": filename, "samples": len(arrays["frame_index"])})
            index["samples"] += len(arrays["frame_index"])
            pending, pending_size = [], 0

        stage_start = time.perf_counter()
        batch_ready_at = stage_start
        read_wait_seconds = compute_seconds = write_seconds = 0.0
        seen = 0
        logging.info(
            "EXTRACT %s: %d anchors, %d batches; first batch includes JAX compilation.",
            split_name, len(selected), len(loader),
        )
        with prefetch_batches(loader) as batches:
            for batch_index, batch in enumerate(batches):
                now = time.perf_counter()
                read_wait_seconds += now - batch_ready_at
                real_count = len(batch["episode_id"])
                padded_observation = jax.tree.map(lambda value: pad_array(value, args.batch_size), batch["observation"])
                valid = pad_array(batch["valid_future_frames"], args.batch_size).copy()
                valid[real_count:] = False
                observation = model_tavla.Observation.from_dict(padded_observation)
                rng = jax.random.fold_in(
                    jax.random.key(args.seed + (0 if split_name == "train" else 1 if split_name == "val" else 2)),
                    batch_index,
                )
                records = jax.device_get(
                    forward(
                        state,
                        parallel.shard(observation),
                        parallel.shard(pad_array(batch["label_effort"], args.batch_size)),
                        parallel.shard(valid),
                        parallel.replicate(rng),
                    )
                )
                records = {
                    key: np.asarray(value) if key in ("offsets", "future_frame_offsets") else np.asarray(value)[:real_count]
                    for key, value in records.items()
                }
                compute_seconds += time.perf_counter() - now
                seen += real_count
                if args.smoke:
                    for key in ("refined_hidden", "target_contact", "target_force"):
                        if not np.isfinite(records[key]).all():
                            raise ValueError(f"Non-finite values in smoke-test output {key}.")
                    logging.info(
                        "SMOKE PASS: split=%s FAE=%s contact=%s force=%s valid_frames=%d; no cache written.",
                        split_name,
                        records["refined_hidden"].shape,
                        records["target_contact"].shape,
                        records["target_force"].shape,
                        np.sum(records["valid_mask"]),
                    )
                    break
                records["fixed_hidden"] = records["fixed_hidden"][:, 0]
                for key in ("fixed_hidden", "refined_hidden", "history_tokens"):
                    records[key] = records[key].astype(np.float16)
                    if not np.isfinite(records[key]).all():
                        raise ValueError(f"Non-finite frozen features: {key}")
                for key in ("episode_id", "frame_index", "task_id"):
                    records[key] = np.asarray(batch[key])
                pending.append(records)
                pending_size += len(records["frame_index"])
                if pending_size >= args.shard_size:
                    write_start = time.perf_counter()
                    flush()
                    write_seconds += time.perf_counter() - write_start
                if batch_index % 25 == 0 or batch_index + 1 == len(loader):
                    elapsed = time.perf_counter() - stage_start
                    logging.info(
                        "EXTRACT %s batch=%d/%d anchors=%d/%d rate=%.2f/s "
                        "read_wait=%.1fs inference_and_transfer=%.1fs write=%.1fs",
                        split_name, batch_index + 1, len(loader), seen, len(selected),
                        seen / max(elapsed, 1e-6), read_wait_seconds, compute_seconds, write_seconds,
                    )
                batch_ready_at = time.perf_counter()
        if args.smoke:
            if hasattr(dataset, "close"):
                dataset.close()
            return
        flush()
        if hasattr(dataset, "close"):
            dataset.close()
        if index["samples"] != len(selected) or not selected:
            raise ValueError("Cache did not cover exactly one finite pass over selected anchors.")
        index["complete"] = True
        write_json(index_path, index)
        logging.info("%s cache complete: %d anchors", split_name, index["samples"])


def _cache_index(cache_dir, split):
    index = read_json(cache_dir / split / "index.json")
    if not index.get("complete"):
        raise ValueError(f"Incomplete {split} cache.")
    if index["protocol_sha256"] != fingerprint(read_json(cache_dir / "protocol.json")):
        raise ValueError("Cache provenance mismatch.")
    return index


def _make_probe(index, source, decoder_type, hidden_dim, seed):
    import flax.nnx as nnx

    from openpi.models.future_patch_probe import FuturePatchProbe

    meta = index["model"]
    width = meta["hidden_width"] if source == "fae" else 2 * meta["hidden_width"] + meta["segments"] + 1
    return FuturePatchProbe(
        input_dim=width,
        num_segments=meta["segments"],
        num_fingers=meta["fingers"],
        steps_per_segment=meta["steps_per_segment"],
        num_patches=meta["patches"],
        decoder_type=decoder_type,
        hidden_dim=hidden_dim,
        rngs=nnx.Rngs(seed),
    )


def _save_probe(path, params, metadata):
    import jax

    paths, _ = jax.tree_util.tree_flatten_with_path(params.to_pure_dict())
    arrays = {str(i): np.asarray(jax.device_get(value)) for i, (_, value) in enumerate(paths)}
    metadata = {**metadata, "parameter_paths": [[str(k.key) for k in key] for key, _ in paths]}
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez(stream, **arrays)
    path.with_suffix(".tmp").replace(path)
    write_json(path.with_suffix(".json"), metadata)


def _load_probe(path, index):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    meta = read_json(path.with_suffix(".json"))
    if meta["protocol_sha256"] != index["protocol_sha256"]:
        raise ValueError("Decoder checkpoint was trained on a different feature-cache protocol.")
    probe = _make_probe(index, meta["source"], meta["decoder_type"], meta["hidden_dim"], meta["seed"])
    graphdef, params = nnx.split(probe)
    paths, tree = jax.tree_util.tree_flatten_with_path(params.to_pure_dict())
    if meta["parameter_paths"] != [[str(k.key) for k in key] for key, _ in paths]:
        raise ValueError("Decoder checkpoint parameter structure mismatch.")
    with np.load(path, allow_pickle=False) as checkpoint:
        values = []
        for i, (_, expected) in enumerate(paths):
            value = checkpoint[str(i)]
            if value.shape != expected.shape:
                raise ValueError("Decoder checkpoint shape mismatch.")
            values.append(jnp.asarray(value))
    params.replace_by_pure_dict(jax.tree_util.tree_unflatten(tree, values))
    return graphdef, params, meta


def _prediction_fn(graphdef, parallel=None):
    import flax.nnx as nnx
    import jax

    def function(params, x):
        return nnx.merge(graphdef, params)(x)

    if parallel is None:
        return jax.jit(function)
    compiled = jax.jit(
        function,
        in_shardings=(parallel.replicated, parallel.data),
        out_shardings=parallel.data,
    )

    def predict(params, features):
        count = len(features)
        padded = parallel.shard(pad_array(features, parallel.batch_size))
        # Transfer once, then trim on the CPU rather than compiling dynamic
        # device-side slices for every possible last-batch size.
        outputs = jax.device_get(compiled(params, padded))
        return {key: np.asarray(value)[:count] for key, value in outputs.items()}

    return predict


def _validation_stats_fn(graphdef, parallel, force_scale):
    import flax.nnx as nnx
    import jax

    from openpi.models.future_patch_probe import future_patch_probe_loss

    scale = parallel.replicate(np.asarray(force_scale, dtype=np.float32))

    def statistics(params, features, contact, force, mask):
        _, stats = future_patch_probe_loss(
            nnx.merge(graphdef, params)(features), contact, force / scale, valid_mask=mask
        )
        return stats

    compiled = jax.jit(
        statistics,
        in_shardings=(parallel.replicated, parallel.data, parallel.data, parallel.data, parallel.data),
        out_shardings=parallel.replicated,
    )

    def validate(params, features, contact, force, mask):
        batch = pad_probe_batch((features, contact, force, mask), parallel.batch_size)
        return jax.device_get(compiled(params, *parallel.shard(batch)))

    return validate


def _validation_loss(root, index, predict, params, force_scale, source, batch_size, *, statistics=None):
    from openpi.models.future_patch_probe import future_patch_probe_loss

    losses = []
    modes = (0, 1) if source == "fae" else (0,)
    for shard in index["shards"]:
        cache = read_probe_cache(root / shard["file"], source)
        for start in range(0, len(cache["frame_index"]), batch_size):
            rows = np.arange(start, min(start + batch_size, len(cache["frame_index"])))
            for offset in range(len(cache["offsets"])):
                mask = cache["valid_mask"][rows, offset]
                if not mask.any():
                    continue
                for mode in modes:
                    features = probe_features(
                        cache,
                        rows,
                        np.full(len(rows), offset),
                        np.full(len(rows), mode),
                        source=source,
                        num_segments=index["model"]["segments"],
                        num_fingers=index["model"]["fingers"],
                    )
                    if statistics is None:
                        _, stats = future_patch_probe_loss(
                            predict(params, features),
                            cache["target_contact"][rows],
                            cache["target_force"][rows] / force_scale,
                            valid_mask=mask,
                        )
                    else:
                        stats = statistics(
                            params, features, cache["target_contact"][rows], cache["target_force"][rows], mask
                        )
                    losses.append({key: float(value) for key, value in stats.items()})
    if not losses:
        raise ValueError("No valid validation samples.")

    def pooled(loss_key, count_key):
        count = sum(s[count_key] for s in losses)
        return sum(s[loss_key] * s[count_key] for s in losses) / max(count, 1)

    return (
        pooled("loss/contact", "count/valid_patches")
        + pooled("loss/active_force", "count/active_patches")
        + 0.1 * pooled("loss/inactive_force", "count/inactive_patches")
    )


def collect_metrics(root, index, predict, params, force_scale, *, source, batch_size):
    """Finite test pass: task/offset metrics plus an explicitly labelled transition subset."""
    import jax

    groups = defaultdict(MetricGroup)
    modes = (("fixed", 0), ("refined", 1)) if source == "fae" else (("history", 0),)
    for shard in index["shards"]:
        cache = read_probe_cache(root / shard["file"], source, metrics=True)
        for start in range(0, len(cache["frame_index"]), batch_size):
            rows = np.arange(start, min(start + batch_size, len(cache["frame_index"])))
            contact, target = cache["target_contact"][rows], cache["target_force"][rows]
            for oi, offset in enumerate(cache["offsets"]):
                valid = cache["valid_mask"][rows, oi]
                if not valid.any():
                    continue
                predictions = []
                for name, mode in modes:
                    features = probe_features(
                        cache,
                        rows,
                        np.full(len(rows), oi),
                        np.full(len(rows), mode),
                        source=source,
                        num_segments=index["model"]["segments"],
                        num_fingers=index["model"]["fingers"],
                    )
                    out = jax.device_get(predict(params, features))
                    predictions.append(
                        (name, np.asarray(out["contact_logits"]), np.asarray(out["force"]) * force_scale)
                    )
                current = cache["current_contact"][rows, oi]
                persistence_logits = np.broadcast_to(np.where(current[:, None] > 0.5, 20.0, -20.0), contact.shape)
                persistence_force = np.broadcast_to(cache["current_force"][rows, oi][:, None], target.shape)
                predictions.append(("persistence", persistence_logits, persistence_force))
                for task_id in np.unique(cache["task_id"][rows]):
                    selected = cache["task_id"][rows] == task_id
                    task = index["task_names"][int(task_id)]
                    ids = cache["episode_id"][rows][selected]
                    for name, logits, force in predictions:
                        for scope in ("all_future", f"offset_{int(offset)}"):
                            groups[task, name, scope].add(
                                logits[selected],
                                contact[selected],
                                force[selected],
                                target[selected],
                                valid[selected],
                                ids,
                            )
                        # Use true changes for evaluation stratification ONLY. Preserve all patches in
                        # a frame with any change; label the subset accordingly in reports.
                        changed = np.any(contact[selected] != current[selected, None], axis=(-2, -1))
                        groups[task, name, "contact_change_frames"].add(
                            logits[selected],
                            contact[selected],
                            force[selected],
                            target[selected],
                            valid[selected] & changed,
                            ids,
                        )
    return {key: group.finalize() for key, group in groups.items()}


def train(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    from openpi.models.future_patch_probe import future_patch_probe_loss

    train_index, val_index = (_cache_index(args.cache_dir, split) for split in ("train", "val"))
    if set(train_index["episodes"]) & set(val_index["episodes"]):
        raise ValueError("Train/validation episodes overlap.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_probe.npz"
    epoch_dir = args.output_dir / "checkpoints"
    if checkpoint.exists() or (epoch_dir.exists() and any(epoch_dir.iterdir())):
        raise FileExistsError("Decoder checkpoint exists; choose a new output directory.")
    epoch_dir.mkdir(parents=True, exist_ok=True)
    parallel = ProbeParallel(getattr(args, "num_devices", 1), args.batch_size)
    force_scale = training_force_scale(args.cache_dir / "train", train_index)
    probe = _make_probe(train_index, args.source, args.decoder_type, args.hidden_dim, args.seed)
    graphdef, params = nnx.split(probe)
    params = parallel.replicate(params)
    tx = optax.adam(args.learning_rate)
    opt_state = parallel.replicate(tx.init(params))
    scale = parallel.replicate(jnp.asarray(force_scale))

    def step(params, opt_state, x, contact, force, mask):
        def objective(p):
            out = nnx.merge(graphdef, p)(x)
            return future_patch_probe_loss(out, contact, force / scale, valid_mask=mask)[0]

        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    step = jax.jit(
        step,
        in_shardings=(
            parallel.replicated, parallel.replicated,
            parallel.data, parallel.data, parallel.data, parallel.data,
        ),
        out_shardings=(parallel.replicated, parallel.replicated, parallel.replicated),
        donate_argnums=(0, 1),
    )
    predict = _prediction_fn(graphdef, parallel)
    statistics = _validation_stats_fn(graphdef, parallel, force_scale)
    metadata = {
        "source": args.source,
        "decoder_type": args.decoder_type,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "force_scale": force_scale.tolist(),
        "protocol_sha256": train_index["protocol_sha256"],
        "parameter_count": sum(x.size for x in jax.tree.leaves(params)),
        "selection_metric": "validation_probe_loss",
        "threshold": 0.5,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "num_devices": parallel.num_devices,
        "global_batch_size": args.batch_size,
        "policy_frozen": True,
        "train_episodes": train_index["episodes"],
        "val_episodes": val_index["episodes"],
    }
    best, history = float("inf"), []
    global_step = 0
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        losses = []
        epoch_start = time.perf_counter()
        batch_stream = training_batches(
            args.cache_dir / "train", train_index, batch_size=args.batch_size, rng=rng, source=args.source
        )
        logging.info("TRAIN epoch=%d/%d started; first step may compile.", epoch + 1, args.epochs)
        with prefetch_batches(batch_stream) as batches:
            for batch in batches:
                batch = parallel.shard(pad_probe_batch(batch, args.batch_size))
                params, opt_state, loss = step(params, opt_state, *batch)
                losses.append(loss)
                global_step += 1
                if len(losses) % 100 == 0:
                    jax.block_until_ready(loss)
                    logging.info("TRAIN epoch=%d batch=%d step=%d", epoch + 1, len(losses), global_step)
        losses = np.asarray(jax.device_get(losses), dtype=np.float64)
        if not len(losses) or not np.isfinite(losses).all():
            raise ValueError("Empty/non-finite probe training; inspect units, masks and input cache.")
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        logging.info("VALIDATE epoch=%d started", epoch + 1)
        val_loss = _validation_loss(
            args.cache_dir / "val", val_index, predict, params, force_scale, args.source, args.batch_size,
            statistics=statistics,
        )
        validation_seconds = time.perf_counter() - validation_start
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss.")
        epoch_stats = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "epoch_steps": len(losses),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "train_batch_mean_loss": float(np.mean(losses)),
            "val_loss": val_loss,
        }
        history.append(epoch_stats)
        # Preserve every epoch for inspection; threshold selection is performed
        # only on the validation-selected best checkpoint after training.
        epoch_checkpoint = epoch_dir / f"epoch_{epoch + 1:04d}.npz"
        snapshot_metadata = {**metadata, **epoch_stats, "training_complete": False}
        _save_probe(epoch_checkpoint, params, snapshot_metadata)
        if val_loss < best:
            best = val_loss
            _save_probe(checkpoint, params, {**snapshot_metadata, "best_epoch": epoch + 1, "best_val_loss": best})
        write_json(args.output_dir / "training_history.json", history)
        logging.info(
            "epoch=%d step=%d train=%.5f val=%.5f best=%.5f checkpoint=%s train_s=%.1f val_s=%.1f",
            epoch + 1,
            global_step,
            np.mean(losses),
            val_loss,
            best,
            epoch_checkpoint,
            train_seconds,
            validation_seconds,
        )
    graphdef, params, metadata = _load_probe(checkpoint, val_index)
    params = parallel.replicate(params)
    logging.info("CALIBRATE best epoch=%d on validation data", metadata["best_epoch"])
    groups = collect_metrics(
        args.cache_dir / "val",
        val_index,
        _prediction_fn(graphdef, parallel),
        params,
        force_scale,
        source=args.source,
        batch_size=args.batch_size,
    )
    best_f1, threshold = -1.0, 0.5
    for candidate in np.linspace(0.05, 0.95, 19):
        scores = [
            row["contact_f1"]
            for row in summarize_groups(groups, float(candidate))
            if row["task"] == "__macro__" and row["scope"] == "all_future" and row["mode"] != "persistence"
        ]
        score = float(np.nanmean(scores)) if np.isfinite(scores).any() else -1.0
        if score > best_f1:
            best_f1, threshold = score, float(candidate)
    metadata["threshold"] = threshold
    metadata["threshold_selected_on"] = "validation macro F1, equally averaging fixed/refined"
    metadata["training_complete"] = True
    write_json(checkpoint.with_suffix(".json"), metadata)
    write_json(args.output_dir / "validation_metrics.json", summarize_groups(groups, threshold))
    logging.info("Saved frozen readout: %s; validation contact threshold=%.2f", checkpoint, threshold)


def evaluate(args):
    parallel = ProbeParallel(getattr(args, "num_devices", 1), args.batch_size)
    index = _cache_index(args.cache_dir, "test")
    graphdef, params, metadata = _load_probe(args.probe_checkpoint, index)
    if not metadata.get("training_complete"):
        raise ValueError(
            "Probe fitting/validation threshold selection is incomplete; finish train before final testing."
        )
    params = parallel.replicate(params)
    logging.info("EVALUATE held-out test data started")
    groups = collect_metrics(
        args.cache_dir / "test",
        index,
        _prediction_fn(graphdef, parallel),
        params,
        np.asarray(metadata["force_scale"], dtype=np.float32),
        source=metadata["source"],
        batch_size=args.batch_size,
    )
    rows = summarize_groups(groups, metadata["threshold"])
    logging.info("BOOTSTRAP %d repetitions on CPU", args.bootstrap_samples)
    ci = paired_episode_bootstrap(groups, metadata["threshold"], repetitions=args.bootstrap_samples, seed=args.seed)
    protocol = read_json(args.cache_dir / "protocol.json")
    result = {
        "probe": metadata,
        "protocol": protocol,
        "test_episodes": index["episodes"],
        "test_anchors": index["samples"],
        "metrics": rows,
        "bootstrap_all_future_macro": ci,
        "metric_notes": {
            "contact_ap": "Exact non-interpolated average precision with tied scores grouped; NOT trapezoidal PR AUC.",
            "force_errors": "Per-coordinate error in native label units, computed on true active patches without gating predictions.",
            "macro": "Equal weight per task; undefined task metrics excluded with contributing task count reported.",
            "contact_change_frames": "Frames where any patch differs from the current observed contact state; all patches retained.",
            "ci": "Task-stratified, paired episode bootstrap; conditional on this trained policy/probe, not training-seed uncertainty.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "test_metrics.json", result)
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "test_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["task"] == "__macro__" and row["scope"] == "all_future":
            logging.info(
                "%s: AP=%.4f F1=%.4f active RMSE=%.4f",
                row["mode"],
                row["contact_ap"],
                row["contact_f1"],
                row["active_force_rmse"],
            )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Lock episode-level train/validation/test manifests (no GPU).")
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument(
        "--test-task-scope",
        choices=("all", "common"),
        default="all",
        help="common evaluates only test episodes whose exact task labels occur in the training pool.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser("extract", help="Cache deployed FAE states and native-label targets; never trains the policy.")
    p.add_argument("--config-name", default="pi0_xhand_dual_patch_pretrained_f4_h16_async")
    p.add_argument("--policy-params", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    p.add_argument("--contact-threshold", type=float, required=True)
    p.add_argument("--contact-temperature", type=float, required=True)
    p.add_argument("--force-unit", default="dataset_native")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--num-devices", type=int, default=1, help="Number of visible devices for data parallelism.")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run one real batch through frozen FAE without writing cache.")
    p.set_defaults(func=extract)
    p = sub.add_parser("train", help="Optimize only the small decoder from cached, frozen features.")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--source", choices=("fae", "history"), default="fae")
    p.add_argument("--decoder-type", choices=("linear", "mlp"), default="linear")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--num-devices", type=int, default=1, help="Number of visible devices for data parallelism.")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate", help="Evaluate the validation-selected decoder on final held-out test episodes.")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--probe-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--num-devices", type=int, default=1, help="Number of visible devices for data parallelism.")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=evaluate)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    arguments.func(arguments)
