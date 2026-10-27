#!/usr/bin/env python3
"""Prepare -> extract frozen FAE -> train readout -> evaluate on held-out episodes.

Only ``extract`` imports the policy/LeRobot stack. ``train`` and ``evaluate``
consume immutable feature caches and cannot update or even load the policy.
"""

# Lazy imports isolate CPU-only preparation and cached readout fitting from the policy/LeRobot stack.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import dataclasses
import hashlib
import logging
from pathlib import Path

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


def prepare(args):
    manifest = create_manifest(args.train_root, args.test_root, val_ratio=args.val_ratio, seed=args.seed)
    if args.manifest.exists() and read_json(args.manifest) != manifest:
        raise FileExistsError("Manifest exists with different splits. Use a new output path.")
    write_json(args.manifest, manifest)
    for name, split in manifest["splits"].items():
        selected = set(split["episodes"])
        tasks = sorted(
            {row["task"] for row in manifest["datasets"][split["dataset"]]["episodes"] if row["episode"] in selected}
        )
        logging.info("%s: %d episodes, task labels=%s", name, len(selected), tasks)


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
        from openpi.models import model_tavla

        row = int(self.indices[index])
        episode, frame = int(self.episodes[row]), int(self.frames[row])
        raw = self.dataset[row]
        if (
            int(np.asarray(raw["episode_index"]).item()) != episode
            or int(np.asarray(raw["frame_index"]).item()) != frame
        ):
            raise ValueError("LeRobot sample metadata disagrees with cache manifest.")
        info = self.episode_info[episode]
        if "timestamp" in raw and not np.isclose(
            float(np.asarray(raw["timestamp"]).item()), frame / info["fps"], atol=1e-3
        ):
            raise ValueError("Nonuniform timestamps: frame offsets cannot be used as future labels.")
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
        # Observation excludes all ground-truth actions. extract_probe_batch
        # subsequently removes future tactile/auxiliary label-only fields.
        observation = model_tavla.Observation.from_dict(inputs)
        return {
            "observation": observation.to_dict(),
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


def extract(args):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import torch

    from openpi.models import model as policy_model
    from openpi.models import model_tavla
    from openpi.shared import normalize
    from openpi.training import config as configs
    from openpi.training import data_loader
    from scripts.extract_future_patch_probe import extract_probe_batch
    from scripts.extract_future_patch_probe import probe_metadata
    from scripts.extract_future_patch_probe import validate_probe_model

    manifest = read_json(args.manifest)
    params_path, norm_path = _checkpoint_and_norm(args)
    config = configs.get_config(args.config_name)
    if args.batch_size <= 0 or args.stride <= 0 or args.shard_size <= 0:
        raise ValueError("Batch size, cache stride, and shard size must be positive.")
    expected_offsets = tuple(range(1 - config.model.force_input_frames, config.model.action_horizon + 1))
    if tuple(config.data.state_delta_timestamps) != expected_offsets:
        raise ValueError("Policy state offsets must be contiguous history through current, then t+1..t+H.")

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
        "model": metadata,
    }
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

    forward = jax.jit(forward)

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
        if index_path.exists():
            index = read_json(index_path)
            if index.get("complete") and args.resume:
                if not all((destination / s["file"]).is_file() for s in index["shards"]):
                    raise ValueError("Completed cache index refers to missing shards.")
                logging.info("Reusing completed %s cache", split_name)
                continue
            raise FileExistsError(f"Cache index already exists: {index_path}")
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
        dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        episodes, frames = data_loader._frame_metadata(dataset)  # noqa: SLF001
        info_by_episode = {r["episode"]: {**r, "fps": dataset_info["fps"]} for r in dataset_info["episodes"]}
        selected = []
        for episode in split["episodes"]:
            rows = np.flatnonzero(episodes == episode)
            rows = rows[np.argsort(frames[rows])]
            length = info_by_episode[episode]["length"]
            if not np.array_equal(frames[rows], np.arange(length)):
                raise ValueError(f"Episode {episode} has missing/duplicate/nonzero-based frames.")
            selected.extend(rows[(frames[rows] % args.stride == 0) & (frames[rows] + 1 < length)].tolist())
        wrapped = ProbeSampleDataset(
            dataset,
            selected,
            episodes,
            frames,
            info_by_episode,
            manifest["task_names"],
            data_config,
            model.action_horizon,
            model.force_input_frames,
        )
        loader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=_collate,
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

        for batch_index, batch in enumerate(loader):
            observation = model_tavla.Observation.from_dict(batch["observation"])
            rng = jax.random.fold_in(
                jax.random.key(args.seed + (0 if split_name == "train" else 1 if split_name == "val" else 2)),
                batch_index,
            )
            records = jax.device_get(
                forward(state, observation, batch["label_effort"], batch["valid_future_frames"], rng)
            )
            records = {key: np.asarray(value) for key, value in records.items()}
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
                flush()
            if batch_index % 25 == 0:
                logging.info("%s: batch %d/%d", split_name, batch_index + 1, len(loader))
        flush()
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


def _prediction_fn(graphdef):
    import flax.nnx as nnx
    import jax

    return jax.jit(lambda params, x: nnx.merge(graphdef, params)(x))


def _validation_loss(root, index, predict, params, force_scale, source, batch_size):
    from openpi.models.future_patch_probe import future_patch_probe_loss

    losses = []
    modes = (0, 1) if source == "fae" else (0,)
    for shard in index["shards"]:
        with np.load(root / shard["file"], allow_pickle=False) as file:
            cache = dict(file)
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
                    _, stats = future_patch_probe_loss(
                        predict(params, features),
                        cache["target_contact"][rows],
                        cache["target_force"][rows] / force_scale,
                        valid_mask=mask,
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
        with np.load(root / shard["file"], allow_pickle=False) as file:
            cache = dict(file)
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
    if checkpoint.exists():
        raise FileExistsError("Decoder checkpoint exists; choose a new output directory.")
    force_scale = training_force_scale(args.cache_dir / "train", train_index)
    probe = _make_probe(train_index, args.source, args.decoder_type, args.hidden_dim, args.seed)
    graphdef, params = nnx.split(probe)
    tx = optax.adam(args.learning_rate)
    opt_state = tx.init(params)
    scale = jnp.asarray(force_scale)

    def step(params, opt_state, x, contact, force, mask):
        def objective(p):
            out = nnx.merge(graphdef, p)(x)
            return future_patch_probe_loss(out, contact, force / scale, valid_mask=mask)[0]

        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    step, predict = jax.jit(step), _prediction_fn(graphdef)
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
        "policy_frozen": True,
        "train_episodes": train_index["episodes"],
        "val_episodes": val_index["episodes"],
    }
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        losses = []
        for batch in training_batches(
            args.cache_dir / "train", train_index, batch_size=args.batch_size, rng=rng, source=args.source
        ):
            params, opt_state, loss = step(params, opt_state, *batch)
            losses.append(float(loss))
        if not losses or not np.isfinite(losses).all():
            raise ValueError("Empty/non-finite probe training; inspect units, masks and input cache.")
        val_loss = _validation_loss(
            args.cache_dir / "val", val_index, predict, params, force_scale, args.source, args.batch_size
        )
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss.")
        history.append({"epoch": epoch + 1, "train_batch_mean_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            _save_probe(checkpoint, params, {**metadata, "best_epoch": epoch + 1, "best_val_loss": best})
        write_json(args.output_dir / "training_history.json", history)
        logging.info("epoch=%d train=%.5f val=%.5f best=%.5f", epoch + 1, np.mean(losses), val_loss, best)
    graphdef, params, metadata = _load_probe(checkpoint, val_index)
    groups = collect_metrics(
        args.cache_dir / "val",
        val_index,
        _prediction_fn(graphdef),
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
    index = _cache_index(args.cache_dir, "test")
    graphdef, params, metadata = _load_probe(args.probe_checkpoint, index)
    if not metadata.get("training_complete"):
        raise ValueError(
            "Probe fitting/validation threshold selection is incomplete; finish train before final testing."
        )
    groups = collect_metrics(
        args.cache_dir / "test",
        index,
        _prediction_fn(graphdef),
        params,
        np.asarray(metadata["force_scale"], dtype=np.float32),
        source=metadata["source"],
        batch_size=args.batch_size,
    )
    rows = summarize_groups(groups, metadata["threshold"])
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
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
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
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate", help="Evaluate the validation-selected decoder on final held-out test episodes.")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--probe-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=evaluate)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = build_parser().parse_args()
    arguments.func(arguments)
