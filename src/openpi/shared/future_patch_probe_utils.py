"""CPU utilities for frozen-FAE probing; no policy or accelerator imports."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Write standards-compliant JSON, including undefined metric values as null."""

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def dataset_metadata(root):
    root = Path(root).expanduser().resolve()
    info = read_json(root / "meta/info.json")
    rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    episodes = []
    for row in rows:
        tasks = row.get("tasks", [row.get("task")])
        if len(tasks) != 1 or not tasks[0]:
            raise ValueError("Probe splits require one explicit task name per episode.")
        episodes.append({"episode": int(row["episode_index"]), "task": str(tasks[0]), "length": int(row["length"])})
    ids = [row["episode"] for row in episodes]
    if len(set(ids)) != len(ids) or len(ids) != int(info["total_episodes"]):
        raise ValueError(f"Duplicate episodes or inconsistent episode count in {root}.")
    if float(info["fps"]) <= 0 or any(row["length"] <= 0 for row in episodes):
        raise ValueError("Dataset FPS and episode lengths must be positive.")
    return {
        "root": str(root),
        "fps": float(info["fps"]),
        "episodes": episodes,
        "metadata_sha256": fingerprint({"info": info, "episodes": rows}),
    }


def create_manifest(train_root, test_root, *, val_ratio=0.1, seed=42):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must lie in (0, 1).")
    train, test = dataset_metadata(train_root), dataset_metadata(test_root)
    if train["root"] == test["root"]:
        raise ValueError("Training and final test datasets must be separate.")
    train_tasks = sorted({r["task"] for r in train["episodes"]})
    test_tasks = sorted({r["task"] for r in test["episodes"]})
    # Task IDs are only metric-group identifiers, not policy inputs. Train/val
    # stratification uses training labels; test macro averages use test labels.
    # Keep every name verbatim rather than guessing cross-dataset equivalences.
    task_names = sorted(set(train_tasks) | set(test_tasks))
    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for task in train_tasks:
        ids = sorted(r["episode"] for r in train["episodes"] if r["task"] == task)
        if len(ids) < 2:
            raise ValueError(f"Task {task!r} needs at least two training-pool episodes.")
        count = min(len(ids) - 1, max(1, round(len(ids) * val_ratio)))
        selected = set(map(int, rng.choice(ids, size=count, replace=False)))
        val_ids.extend(sorted(selected))
        train_ids.extend(i for i in ids if i not in selected)
    return {
        "version": 1,
        "seed": seed,
        "val_ratio": val_ratio,
        "task_names": task_names,
        "datasets": {"train_pool": train, "test": test},
        "splits": {
            "train": {"dataset": "train_pool", "episodes": sorted(train_ids)},
            "val": {"dataset": "train_pool", "episodes": sorted(val_ids)},
            "test": {"dataset": "test", "episodes": sorted(r["episode"] for r in test["episodes"])},
        },
    }


def check_parameter_tree(expected, loaded, prefix=()):
    """Fail on missing/mismatched tensors rather than retaining initialized weights."""
    if isinstance(expected, dict):
        if not isinstance(loaded, dict):
            raise ValueError(f"Checkpoint structure mismatch at {prefix}.")
        for key, value in expected.items():
            if key not in loaded:
                raise ValueError(f"Missing checkpoint parameter: {(*prefix, key)}")
            check_parameter_tree(value, loaded[key], (*prefix, key))
    elif not hasattr(loaded, "dtype") or tuple(loaded.shape) != tuple(expected.shape):
        raise ValueError(
            f"Checkpoint tensor mismatch at {prefix}: expected {expected.shape}, got {getattr(loaded, 'shape', None)}"
        )


def future_validity(frame, episode_length, horizon):
    return frame + np.arange(1, horizon + 1) < episode_length


def probe_features(cache, rows, offset_ids, mode_ids, *, source, num_segments, num_fingers):
    """Select FAE features or a separate history baseline with explicit lead time."""
    rows, offset_ids, mode_ids = map(np.asarray, (rows, offset_ids, mode_ids))
    if source == "fae":
        refined = cache["refined_hidden"][rows, offset_ids].astype(np.float32)
        fixed = cache["fixed_hidden"][rows].astype(np.float32)
        return np.where(mode_ids[:, None, None] == 0, fixed, refined)
    if source != "history":
        raise ValueError(f"Unknown probe source: {source}")
    history = cache["history_tokens"][rows, offset_ids].astype(np.float32)
    if history.shape[1] != 2 * num_fingers:
        raise ValueError("History baseline requires pooled-history + current tokens, one per finger each.")
    per_finger = np.concatenate([history[:, :num_fingers], history[:, num_fingers:]], axis=-1)
    x = np.broadcast_to(per_finger[:, None], (len(rows), num_segments, *per_finger.shape[1:]))
    # A horizon-aware baseline, not the same prediction repeated for all segments.
    segment = np.broadcast_to(np.eye(num_segments, dtype=np.float32)[None, :, None], (*x.shape[:3], num_segments))
    offset = cache["offsets"][offset_ids] / float(cache["target_contact"].shape[1])
    offset = np.broadcast_to(offset[:, None, None, None], (*x.shape[:3], 1))
    return np.concatenate([x, segment, offset], axis=-1).reshape(len(rows), num_segments * num_fingers, -1)


def training_batches(cache_root, index, *, batch_size, rng, source, shuffle=True):
    """One finite pass over cache shards; FAE mode/offset are sampled uniformly."""
    shards = list(index["shards"])
    if shuffle:
        rng.shuffle(shards)
    for shard in shards:
        with np.load(Path(cache_root) / shard["file"], allow_pickle=False) as file:
            cache = dict(file)
        rows = np.arange(len(cache["frame_index"]))
        if shuffle:
            rng.shuffle(rows)
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            offsets = rng.integers(len(cache["offsets"]), size=len(selected))
            modes = rng.integers(2, size=len(selected))
            mask = cache["valid_mask"][selected, offsets]
            keep = mask.any(axis=1)
            if not keep.any():
                continue
            selected, offsets, modes, mask = selected[keep], offsets[keep], modes[keep], mask[keep]
            yield (
                probe_features(
                    cache,
                    selected,
                    offsets,
                    modes,
                    source=source,
                    num_segments=index["model"]["segments"],
                    num_fingers=index["model"]["fingers"],
                ),
                cache["target_contact"][selected],
                cache["target_force"][selected],
                mask,
            )


def training_force_scale(cache_root, index):
    total, count = np.zeros(3, dtype=np.float64), 0
    for shard in index["shards"]:
        with np.load(Path(cache_root) / shard["file"], allow_pickle=False) as c:
            active = (c["target_contact"] > 0.5) & c["valid_mask"][:, 0, :, None, None]
            values = c["target_force"][active].astype(np.float64)
            total += np.sum(values**2, axis=0)
            count += len(values)
    if count == 0:
        raise ValueError("No active training contacts; check native label units and contact threshold.")
    return np.maximum(np.sqrt(total / count), 1e-6).astype(np.float32)


class MetricGroup:
    """Exact AP (tie-aware), global F1, and force errors, with episode-weighted resampling."""

    def __init__(self):
        self.parts = []
        self.force_by_episode = defaultdict(lambda: np.zeros(6, dtype=np.float64))

    def add(self, logits, contact, force, target_force, valid, episode_ids):
        mask = np.broadcast_to(valid[..., None, None], contact.shape)
        if not mask.any():
            return
        labels = contact[mask] > 0.5
        scores = logits[mask].astype(np.float32)
        errors = (force[mask] - target_force[mask]).astype(np.float64)
        ids = np.broadcast_to(np.asarray(episode_ids)[:, None, None, None], contact.shape)[mask]
        if not np.isfinite(scores).all() or not np.isfinite(errors).all():
            raise ValueError("Non-finite predictions or real targets in valid test frames.")
        self.parts.append((scores, labels, ids.astype(np.int32)))
        for episode in np.unique(ids):
            active, inactive = (ids == episode) & labels, (ids == episode) & ~labels
            self.force_by_episode[int(episode)] += [
                np.sum(errors[active] ** 2),
                np.sum(np.abs(errors[active])),
                int(active.sum()) * 3,
                np.sum(errors[inactive] ** 2),
                np.sum(np.abs(errors[inactive])),
                int(inactive.sum()) * 3,
            ]

    def finalize(self):
        if not self.parts:
            return self
        scores, labels, ids = (np.concatenate([part[i] for part in self.parts]) for i in range(3))
        order = np.argsort(-scores, kind="stable")
        self.scores, self.labels, self.ids = scores[order], labels[order], ids[order]
        self.ends = np.r_[np.flatnonzero(np.diff(self.scores)), len(scores) - 1]
        self.episodes = np.unique(ids)
        self.parts.clear()
        return self

    def compute(self, threshold=0.5, multiplicities=None):
        if not hasattr(self, "scores"):
            return {}
        if not 0 < threshold < 1:
            raise ValueError("Contact decision threshold must lie in (0, 1).")
        weights = np.ones(len(self.ids)) if multiplicities is None else multiplicities[self.ids].astype(np.float64)
        positives = weights * self.labels
        positive_count = positives.sum()
        cum_tp, cum_n = np.cumsum(positives)[self.ends], np.cumsum(weights)[self.ends]
        increments = np.diff(np.r_[0.0, cum_tp])
        ap = (
            (np.sum(np.divide(cum_tp, cum_n, out=np.zeros_like(cum_tp), where=cum_n > 0) * increments) / positive_count)
            if positive_count
            else np.nan
        )
        predicted = self.scores >= np.log(threshold / (1.0 - threshold))
        tp = positives[predicted].sum()
        fp = (weights * ~self.labels)[predicted].sum()
        fn = positive_count - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        force_stats = sum(
            (
                values * (1 if multiplicities is None else multiplicities[episode])
                for episode, values in self.force_by_episode.items()
            ),
            np.zeros(6),
        )
        sse, sae, n, inactive_sse, _, inactive_n = force_stats
        return {
            "contact_ap": float(ap),
            "contact_f1": float(f1),
            "active_force_rmse": float(np.sqrt(sse / n)) if n else np.nan,
            "active_force_mae": float(sae / n) if n else np.nan,
            "inactive_force_rmse": float(np.sqrt(inactive_sse / inactive_n)) if inactive_n else np.nan,
            "valid_patches": int(weights.sum()),
            "positive_patches": int(positive_count),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }


METRICS = ("contact_ap", "contact_f1", "active_force_rmse", "active_force_mae", "inactive_force_rmse")


def macro_metrics(values):
    result = {}
    for metric in METRICS:
        finite = [row[metric] for row in values if metric in row and np.isfinite(row[metric])]
        result[metric] = float(np.mean(finite)) if finite else np.nan
        result[metric + "_tasks"] = len(finite)
    return result


def summarize_groups(groups, threshold):
    rows = []
    for (task, mode, scope), group in sorted(groups.items()):
        values = group.compute(threshold=0.5 if mode == "persistence" else threshold)
        if values:
            rows.append({"task": task, "mode": mode, "scope": scope, **values})
    for mode, scope in sorted({(r["mode"], r["scope"]) for r in rows}):
        selected = [r for r in rows if r["mode"] == mode and r["scope"] == scope]
        rows.append({"task": "__macro__", "mode": mode, "scope": scope, **macro_metrics(selected)})
    return rows


def paired_episode_bootstrap(groups, threshold, *, repetitions=200, seed=42):
    """Main all-future macro CIs; shared within-task episode resamples across methods."""
    overall = {
        (task, mode): group
        for (task, mode, scope), group in groups.items()
        if scope == "all_future" and hasattr(group, "scores")
    }
    tasks = sorted({task for task, _ in overall})
    modes = sorted({mode for _, mode in overall})
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = {mode: [] for mode in modes}
    for _ in range(repetitions):
        values = {mode: [] for mode in modes}
        for task in tasks:
            task_groups = [overall[task, mode] for mode in modes]
            episodes = task_groups[0].episodes
            if any(not np.array_equal(g.episodes, episodes) for g in task_groups):
                raise ValueError("Paired bootstrap requires identical evaluated episodes across modes.")
            weights = np.bincount(
                rng.choice(episodes, size=len(episodes), replace=True), minlength=int(episodes.max()) + 1
            )
            for mode in modes:
                values[mode].append(
                    overall[task, mode].compute(
                        threshold=0.5 if mode == "persistence" else threshold, multiplicities=weights
                    )
                )
        for mode in modes:
            draws[mode].append(macro_metrics(values[mode]))
    result = {}
    for mode in modes:
        result[mode] = {}
        for metric in METRICS:
            samples = np.array([r[metric] for r in draws[mode]])
            finite = samples[np.isfinite(samples)]
            result[mode][metric] = {
                "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                "valid_resamples": len(finite),
            }
    if "fixed" in modes and "refined" in modes:
        result["delta_refined_minus_fixed"] = {}
        for metric in METRICS:
            delta = np.array([a[metric] - b[metric] for a, b in zip(draws["refined"], draws["fixed"], strict=True)])
            finite = delta[np.isfinite(delta)]
            result["delta_refined_minus_fixed"][metric] = {
                "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                "valid_resamples": len(finite),
            }
    return result
