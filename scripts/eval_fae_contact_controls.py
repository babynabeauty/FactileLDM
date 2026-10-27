#!/usr/bin/env python3
"""Read-only contact controls from completed caches and an existing best_probe.

Never loads the policy, images, videos or training environment data readers.
Only the frozen small decoder runs on GPUs; contact statistics run on the CPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import numpy as np

from openpi.shared.contact_probe_controls import METRICS
from openpi.shared.contact_probe_controls import contact_scopes
from openpi.shared.contact_probe_controls import cross_episode_donors
from openpi.shared.contact_probe_controls import fit_contact_prior
from openpi.shared.contact_probe_controls import macro
from openpi.shared.contact_probe_controls import make_groups
from openpi.shared.contact_probe_controls import paired_bootstrap
from openpi.shared.future_patch_probe_utils import fingerprint
from openpi.shared.future_patch_probe_utils import read_json
from openpi.shared.future_patch_probe_utils import write_json
from openpi.shared.probe_parallel import ProbeParallel
from scripts.run_future_patch_probe import _cache_index
from scripts.run_future_patch_probe import _load_probe
from scripts.run_future_patch_probe import _prediction_fn


LABEL_KEYS = ("target_contact", "current_contact", "valid_mask", "episode_id", "task_id", "frame_index")
CI_SCOPES = ("all_future", "contact_change_frames", "onset_candidates", "release_candidates")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_shards(root, index, *, features=False):
    seen = 0
    reference_offsets = None
    for shard in index["shards"]:
        with np.load(root / shard["file"], allow_pickle=False) as file:
            keys = (*LABEL_KEYS, "offsets") + (("fixed_hidden", "refined_hidden") if features else ())
            cache = {key: file[key] for key in keys}
        count = len(cache["episode_id"])
        if count != shard["samples"]:
            raise ValueError("Shard sample count mismatch.")
        seen += count
        labels, valid, offsets = cache["target_contact"], cache["valid_mask"], cache["offsets"]
        expected = (count, len(offsets), labels.shape[1])
        if labels.ndim != 4 or valid.shape != expected or valid.dtype != np.bool_:
            raise ValueError("Invalid contact label/validity dimensions or mask dtype.")
        if cache["current_contact"].shape != (count, len(offsets), *labels.shape[2:]):
            raise ValueError("Current contact dimensions disagree with future labels.")
        if np.any(valid & (np.arange(1, labels.shape[1] + 1)[None, None] <= offsets[None, :, None])):
            raise ValueError("Cache contains observed frames marked as future.")
        if reference_offsets is not None and not np.array_equal(offsets, reference_offsets):
            raise ValueError("Offsets differ across shards.")
        reference_offsets = offsets
        if len(np.unique(offsets)) != len(offsets) or 0 not in offsets:
            raise ValueError("Offsets must be distinct and include zero.")
        if np.any(cache["task_id"] < 0) or np.any(cache["task_id"] >= len(index["task_names"])):
            raise ValueError("Invalid task IDs.")
        if np.any(cache["episode_id"] < 0):
            raise ValueError("Negative episode IDs.")
        observed_labels = labels[valid.any(axis=1)]
        observed_current = cache["current_contact"][valid.any(axis=-1)]
        if any(not np.isin(x, (0, 1)).all() for x in (observed_labels, observed_current)):
            raise ValueError("Real contact labels must be binary and finite.")
        # Sanitize masked padding only; it must never affect priors or metrics.
        cache["target_contact"] = np.where(valid.any(axis=1)[..., None, None], labels, 0).astype(bool)
        cache["current_contact"] = np.where(valid.any(axis=-1)[..., None, None], cache["current_contact"], 0).astype(bool)
        yield cache
    if seen != index["samples"]:
        raise ValueError("Index total differs from shard counts.")


def concatenate(parts, index):
    result = {key: np.concatenate([part[key] for part in parts]) for key in LABEL_KEYS}
    result["offsets"] = parts[0]["offsets"]
    if set(result["episode_id"].tolist()) != set(index["episodes"]):
        raise ValueError("Cache episode IDs differ from index.")
    pairs = np.stack((result["episode_id"], result["frame_index"]), axis=1)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("Duplicate episode/frame anchors.")
    for episode in np.unique(result["episode_id"]):
        if len(np.unique(result["task_id"][result["episode_id"] == episode])) != 1:
            raise ValueError("An episode has more than one task.")
    return result


def prior_scores(data, probabilities, counts):
    if np.any(counts[np.unique(data["task_id"])] == 0):
        raise ValueError("Evaluation task has no training labels; prior is undefined, no silent fallback.")
    shape = (*data["valid_mask"].shape, *data["target_contact"].shape[2:])
    p = probabilities[data["task_id"]][:, None, None]
    return np.broadcast_to(np.log(p / (1 - p)), shape)


def groups_for(data, scores, threshold, mask, invert=False):
    return make_groups(scores, data["target_contact"], data["episode_id"], data["task_id"], mask, threshold, invert=invert)


def calibrate_prior(data, scores):
    # Same grid and task-macro objective as the existing probe; no test labels.
    groups = groups_for(data, scores, 0.5, data["valid_mask"][..., None, None])
    best, chosen = -np.inf, None
    for threshold in np.linspace(0.05, 0.95, 19):
        for group in groups.values():
            group.threshold = float(threshold)
        score = macro([group.compute() for group in groups.values()])["contact_f1"]
        if np.isfinite(score) and score > best:
            best, chosen = score, float(threshold)
    if chosen is None:
        raise ValueError("Prior threshold calibration has no defined validation F1.")
    return chosen


def append_rows(rows, groups, *, task_names, mode, scope, repeat=-1):
    values = []
    for task, group in sorted(groups.items()):
        value = group.compute()
        values.append(value)
        rows.append({"task": task_names[task], "mode": mode, "scope": scope, "repeat": repeat, **value})
    if values:
        rows.append({"task": "__macro__", "mode": mode, "scope": scope, "repeat": repeat, **macro(values)})


def run(args):
    if args.shuffle_repeats < 1 or args.bootstrap_samples < 0:
        raise ValueError("shuffle-repeats must be positive; bootstrap-samples must be nonnegative.")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; original results and partial runs are never overwritten.")
    indices = {split: _cache_index(args.cache_dir, split) for split in ("train", "val", "test")}
    train, val, test = (indices[s] for s in ("train", "val", "test"))
    if any(idx["task_names"] != train["task_names"] or idx["model"] != train["model"] for idx in indices.values()):
        raise ValueError("Cache split task/model metadata disagree.")
    if set(train["episodes"]) & set(val["episodes"]):
        raise ValueError("Training and validation episodes overlap.")
    meta = read_json(args.probe_checkpoint.with_suffix(".json"))
    if meta["source"] != "fae" or not meta.get("training_complete"):
        raise ValueError("Requires a completed, validation-selected FAE probe.")
    if meta["protocol_sha256"] != test["protocol_sha256"]:
        raise ValueError("Checkpoint/cache provenance mismatch.")
    for key, index in (("train_episodes", train), ("val_episodes", val)):
        if set(meta[key]) != set(index["episodes"]):
            raise ValueError("Probe train/validation split differs from cache.")
    args.output_dir.mkdir(parents=True)
    provenance = {
        "status": "running", "version": 1, "cache_dir": str(args.cache_dir.resolve()),
        "probe_checkpoint": str(args.probe_checkpoint.resolve()), "probe": meta,
        "checkpoint_sha256": sha256(args.probe_checkpoint),
        "checkpoint_metadata_sha256": sha256(args.probe_checkpoint.with_suffix(".json")),
        "cache_protocol": read_json(args.cache_dir / "protocol.json"),
        "index_fingerprints": {s: fingerprint(i) for s, i in indices.items()},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(args.output_dir / "run.json", provenance)
    logging.info("PRIOR: reading training contact labels only; no training features or policy loaded")
    probabilities, counts = fit_contact_prior(read_shards(args.cache_dir / "train", train), len(train["task_names"]))
    validation = concatenate(list(read_shards(args.cache_dir / "val", val)), val)
    prior_threshold = calibrate_prior(validation, prior_scores(validation, probabilities, counts))
    offsets = validation["offsets"].copy()
    del validation
    write_json(args.output_dir / "contact_prior.json", {
        "task_names": train["task_names"], "probabilities": probabilities, "training_frame_counts": counts,
        "smoothing": "Laplace alpha=1", "threshold": prior_threshold,
        "threshold_selection": "validation all_future task-macro F1; grid 0.05..0.95",
    })

    logging.info("PREDICT: existing frozen decoder on test cache, once per latent")
    parallel = ProbeParallel(args.num_devices, args.batch_size)
    graphdef, params, _ = _load_probe(args.probe_checkpoint, test)
    params = parallel.replicate(params)
    predict = _prediction_fn(graphdef, parallel)
    parts, fixed_parts, refined_parts = [], [], []
    for si, cache in enumerate(read_shards(args.cache_dir / "test", test, features=True)):
        count = len(cache["episode_id"])
        fixed, refined = [], []
        for start in range(0, count, args.batch_size):
            end = min(start + args.batch_size, count)
            logits = predict(params, cache["fixed_hidden"][start:end].astype(np.float32))["contact_logits"]
            fixed.append(logits)
            per_offset = []
            for oi in range(len(cache["offsets"])):
                logits = predict(params, cache["refined_hidden"][start:end, oi].astype(np.float32))["contact_logits"]
                per_offset.append(logits)
            refined.append(np.stack(per_offset, axis=1))
        fixed_parts.append(np.concatenate(fixed))
        refined_parts.append(np.concatenate(refined))
        parts.append({key: cache[key] for key in (*LABEL_KEYS, "offsets")})
        logging.info("PREDICT shard=%d/%d", si + 1, len(test["shards"]))
    data = concatenate(parts, test)
    fixed, refined = np.concatenate(fixed_parts), np.concatenate(refined_parts)
    del parts, fixed_parts, refined_parts, cache, params, predict, graphdef
    if not np.array_equal(offsets, data["offsets"]):
        raise ValueError("Validation/test offset mismatch.")
    oi_zero = int(np.flatnonzero(offsets == 0)[0])
    if not np.allclose(fixed, refined[:, oi_zero], rtol=1e-5, atol=1e-6):
        raise ValueError("Fixed/refined offset-zero predictions should agree; inspect cache consistency.")
    scores = {
        "fixed": np.broadcast_to(fixed[:, None], refined.shape), "refined": refined,
        "prior": prior_scores(data, probabilities, counts),
        "persistence": np.broadcast_to(np.where(data["current_contact"][:, :, None], 20.0, -20.0), refined.shape),
    }
    thresholds = {"fixed": meta["threshold"], "refined": meta["threshold"], "prior": prior_threshold, "persistence": 0.5}
    np.savez(args.output_dir / "contact_predictions.npz", **data, fixed_scores=fixed, refined_scores=refined)
    rows, intervals = [], {}
    # Store only the first, preselected random-map controls for paired CIs.
    # Other draws quantify map variability without pretending they are new episodes.
    first_donors = cross_episode_donors(data["task_id"], data["episode_id"], data["valid_mask"], np.random.default_rng(args.seed))
    oi = np.arange(len(offsets))[None]
    first_scores = {f"{mode}_shuffled": scores[mode][first_donors, oi] for mode in ("fixed", "refined")}
    for scope, mask, invert in contact_scopes(data["target_contact"], data["current_contact"], data["valid_mask"], offsets):
        grouped = {}
        for mode, values in scores.items():
            grouped[mode] = groups_for(data, values, thresholds[mode], mask, invert)
            append_rows(rows, grouped[mode], task_names=test["task_names"], mode=mode, scope=scope)
        if scope in CI_SCOPES:
            for mode, values in first_scores.items():
                grouped[mode] = groups_for(data, values, thresholds[mode.removesuffix("_shuffled")], mask, invert)
            logging.info("BOOTSTRAP scope=%s repetitions=%d (CPU)", scope, args.bootstrap_samples)
            intervals[scope] = paired_bootstrap(grouped, repetitions=args.bootstrap_samples, seed=args.seed + 100000)
        del grouped
    del first_scores

    # Each repeat is scored separately; averaging prediction probabilities BEFORE
    # AP would create a different, artificially smoothed baseline.
    for repeat in range(args.shuffle_repeats):
        donors = first_donors if repeat == 0 else cross_episode_donors(
            data["task_id"], data["episode_id"], data["valid_mask"], np.random.default_rng(args.seed + repeat)
        )
        np.save(args.output_dir / f"donors_{repeat:03d}.npy", donors, allow_pickle=False)
        reassigned = {mode: scores[mode][donors, oi] for mode in ("fixed", "refined")}
        for scope, mask, invert in contact_scopes(data["target_contact"], data["current_contact"], data["valid_mask"], offsets):
            for mode, values in reassigned.items():
                groups = groups_for(data, values, thresholds[mode], mask, invert)
                append_rows(rows, groups, task_names=test["task_names"], mode=f"{mode}_shuffled", scope=scope, repeat=repeat)
        logging.info("SHUFFLE repeat=%d/%d done", repeat + 1, args.shuffle_repeats)
    summary = []
    for mode in ("fixed", "refined"):
        scopes = sorted({r["scope"] for r in rows if r["mode"] == mode})
        for scope in scopes:
            original = next(r for r in rows if r["mode"] == mode and r["scope"] == scope and r["task"] == "__macro__")
            shuffled = [r for r in rows if r["mode"] == f"{mode}_shuffled" and r["scope"] == scope and r["task"] == "__macro__"]
            for metric in METRICS:
                values = np.array([r[metric] for r in shuffled])
                values = values[np.isfinite(values)]
                summary.append({"mode": mode, "scope": scope, "metric": metric,
                                "real": original[metric], "shuffle_mean": float(values.mean()) if len(values) else None,
                                "shuffle_std": float(values.std(ddof=1)) if len(values) > 1 else None,
                                "shuffle_range": [float(values.min()), float(values.max())] if len(values) else None,
                                "delta_real_minus_shuffle_mean": float(original[metric] - values.mean()) if len(values) else None,
                                "valid_repeats": len(values)})
    provenance["status"] = "complete"
    result = {
        "provenance": provenance, "thresholds": thresholds, "metrics": rows,
        "paired_episode_bootstrap": intervals, "shuffle_summary": summary,
        "notes": {
            "purpose": "Contact information decodability, not a fixed-vs-refined ranking or precise force reconstruction.",
            "prior": "Task/finger/patch training contact frequency; validation threshold only. Task identity is given to this baseline.",
            "shuffle": "Same task and update offset, different episode; donor valid frames cover recipient targets. Episode then anchor sampled uniformly with replacement. NOT an exact permutation test.",
            "shuffle_equivalence": "For this deterministic samplewise frozen probe, reassigning its outputs equals decoding reassigned latents. Original recipient labels/current contact/masks stay fixed.",
            "ci": "Paired task-stratified recipient-episode percentile bootstrap; conditional on trained policy/probe. Shuffled CI uses saved donors_000 only; random-map mean/std/range reported separately, not as confidence intervals or p-values.",
            "ap": "Tie-aware non-interpolated AP, equal-task macro. No-positive AP undefined; task counts reported. Decoder scores remain raw float32 logits, preserving the original evaluator AP ordering and ties.",
            "lead": "Target frame minus current update frame, in tactile frames; offset is update time since chunk start.",
            "change": "contact_change_frames retains every patch in frames with any change; changed_patches retains only changed patches. Both are outcome-selected diagnostics.",
            "events": "onset_candidates: currently inactive patches, future contact positive. release_candidates: currently active patches, future noncontact positive; logits negated and probability threshold complemented, no test calibration.",
            "limits": "Decodability alone does not prove causal use by actions, uniquely tactile semantics, or information beyond current touch. Persistence is retained as context. Test episodes are held out for the decoder; main-policy training exposure is not established by these caches.",
        },
    }
    write_json(args.output_dir / "contact_controls.json", result)
    columns = sorted({key for row in rows for key in row})
    with (args.output_dir / "contact_controls.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_dir / "shuffle_summary.json", summary)
    for row in rows:
        if row["task"] == "__macro__" and row["scope"] == "all_future" and row["repeat"] == -1:
            logging.info("%s AP=%.4f F1=%.4f prevalence=%.4f", row["mode"], row["contact_ap"], row["contact_f1"], row["positive_rate"])
    provenance["status"] = "complete"
    write_json(args.output_dir / "run.json", provenance)
    logging.info("CONTACT CONTROLS COMPLETE: %s", args.output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(build_parser().parse_args())
