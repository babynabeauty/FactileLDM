#!/usr/bin/env python3
"""Render the best v8 reconstruction from multiple distinct episodes.

Place this script next to:
    visualize_patch_reconstruction_episodes_v8.py
    visualize_patch_reconstruction_episodes.py

Workflow
--------
1. Scan all episodes under --repo-id.
2. Within each episode, retain a few temporally separated, contact-rich frames.
3. By default, require all five fingers to have above-threshold raw contact.
4. Load the checkpoint once and infer all retained candidates in one batched pass.
5. Select the best reconstruction frame independently for every episode.
6. Rank episodes by their best balanced reconstruction quality.
7. Render the top N distinct episodes using the original v8 visual style.

The output directory contains:
    rank_01_episode_XXXXXX_frame_XXXXXX.png
    rank_02_episode_XXXXXX_frame_XXXXXX.png
    ...
    multi_episode_selection.json

The ranking prioritizes the worst reconstructed finger, then mean reconstruction
quality. It does not force Raw and Target finger rankings to agree; alignment
diagnostics are written to the JSON so the user can make the final qualitative
selection without hiding that distinction.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

import visualize_patch_reconstruction_episodes_v8 as _v8


def _episode_task_names(repo: pathlib.Path) -> dict[int, str]:
    task_index_to_name: dict[int, str] = {}
    tasks_path = repo / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with tasks_path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "task_index" in row and "task" in row:
                    task_index_to_name[int(row["task_index"])] = str(row["task"])

    names: dict[int, str] = {}
    episodes_path = repo / "meta" / "episodes.jsonl"
    with episodes_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row.get("episode_index", row.get("index")))
            if row.get("tasks"):
                name = str(row["tasks"][0])
            elif "task" in row:
                name = str(row["task"])
            elif "task_index" in row:
                task_index = int(row["task_index"])
                name = task_index_to_name.get(task_index, f"task_{task_index}")
            else:
                name = "unknown task"
            names[episode] = name
    return names


def _candidate_positions_with_required_fingers(
    tactile: np.ndarray,
    *,
    threshold: float,
    limit: int,
    min_gap: int,
    min_active_fingers: int,
) -> tuple[list[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    active_count, weakest, geometric, total = _v8._five_finger_frame_metrics(
        tactile,
        threshold=threshold,
    )

    magnitude = np.linalg.norm(np.asarray(tactile), axis=-1)
    per_finger_energy = np.sum(
        np.where(magnitude >= threshold, magnitude, 0.0),
        axis=-1,
    )

    valid = np.flatnonzero(active_count >= min_active_fingers)
    if valid.size == 0:
        return [], (active_count, weakest, geometric, total), per_finger_energy

    # Within valid frames: favor strongest weakest finger, then balance and total.
    order = valid[
        np.lexsort(
            (
                total[valid],
                geometric[valid],
                weakest[valid],
                active_count[valid],
            )
        )[::-1]
    ]

    chosen: list[int] = []
    for raw_position in order:
        position = int(raw_position)
        if any(abs(position - previous) < min_gap for previous in chosen):
            continue
        chosen.append(position)
        if len(chosen) >= limit:
            break

    return chosen, (active_count, weakest, geometric, total), per_finger_energy


def _rank_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def _raw_target_alignment(
    raw_finger_energy: np.ndarray,
    target_strength: np.ndarray,
) -> dict[str, object]:
    raw = np.asarray(raw_finger_energy, dtype=np.float64).reshape(-1)
    target = np.sum(
        np.maximum(np.asarray(target_strength, dtype=np.float64), 0.0),
        axis=-1,
    ).reshape(-1)

    raw_top = int(np.argmax(raw))
    target_top = int(np.argmax(target))

    raw_rank = _rank_vector(raw)
    target_rank = _rank_vector(target)
    raw_rank_centered = raw_rank - np.mean(raw_rank)
    target_rank_centered = target_rank - np.mean(target_rank)
    denominator = np.linalg.norm(raw_rank_centered) * np.linalg.norm(target_rank_centered)
    rank_correlation = (
        float(np.dot(raw_rank_centered, target_rank_centered) / denominator)
        if denominator > 1e-12
        else 0.0
    )

    return {
        "raw_per_finger_energy": raw.tolist(),
        "target_per_finger_strength_sum": target.tolist(),
        "raw_strongest_finger": _v8.FINGER_TITLES[raw_top],
        "target_strongest_finger": _v8.FINGER_TITLES[target_top],
        "strongest_finger_match": bool(raw_top == target_top),
        "finger_rank_correlation": rank_correlation,
    }


def _json_safe_record(record: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in record.items():
        if key in {"selection_key", "candidate_index"}:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer,)):
            result[key] = int(value)
        elif isinstance(value, (np.floating,)):
            result[key] = float(value)
        elif isinstance(value, tuple):
            result[key] = [
                float(item) if isinstance(item, (float, np.floating))
                else int(item) if isinstance(item, (int, np.integer))
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render top v8 reconstruction examples from distinct episodes."
    )
    parser.add_argument(
        "--repo-id",
        type=pathlib.Path,
        default=pathlib.Path("data/grasp_pipette_and_press_button_106ep"),
    )
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument(
        "--config-name",
        default="xhand_patch_force_three_head_encoder_pretrain",
    )
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/pi0_xhand_tactile_structured_raw_dual_ae"),
    )
    parser.add_argument("--asset-id", default="taskall-2")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "outputs/patch_reconstruction_visualization/"
            "pipette_multi_episode_best_v8"
        ),
    )
    parser.add_argument(
        "--layout-dir",
        type=pathlib.Path,
        default=_v8._base._single_vis.DEFAULT_LAYOUT_DIR,
    )

    parser.add_argument(
        "--top-episodes",
        type=int,
        default=12,
        help="Number of distinct episode examples to render.",
    )
    parser.add_argument(
        "--best-candidates-per-episode",
        type=int,
        default=6,
        help="Contact-rich candidate frames retained per episode before inference.",
    )
    parser.add_argument(
        "--best-min-frame-gap",
        type=int,
        default=8,
        help="Minimum source-frame separation between candidates in one episode.",
    )
    parser.add_argument(
        "--min-active-fingers",
        type=int,
        default=5,
        choices=(1, 2, 3, 4, 5),
        help="Minimum fingers with above-threshold raw contact in a candidate.",
    )
    parser.add_argument("--raw-contact-threshold", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=250)

    # Original v8 visual controls.
    parser.add_argument(
        "--paper-raw-scale",
        choices=("frame", "episode"),
        default="frame",
    )
    parser.add_argument(
        "--paper-feature-scale",
        choices=("frame", "episode"),
        default="frame",
    )
    parser.add_argument("--paper-raw-gamma", type=float, default=0.55)
    parser.add_argument("--paper-feature-gamma", type=float, default=0.55)
    parser.add_argument("--paper-patch-layout-sigma", type=float, default=0.045)
    parser.add_argument(
        "--paper-patch-boundary",
        choices=("soft", "hard", "none"),
        default="soft",
    )
    parser.add_argument("--paper-patch-boundary-alpha", type=float, default=0.05)
    parser.add_argument("--paper-patch-boundary-sigma", type=float, default=0.75)
    parser.add_argument("--show-patch-labels", action="store_true")
    parser.add_argument("--patch-label-fontsize", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.top_episodes <= 0:
        raise ValueError("--top-episodes must be positive")
    if args.best_candidates_per_episode <= 0:
        raise ValueError("--best-candidates-per-episode must be positive")
    if args.best_min_frame_gap <= 0:
        raise ValueError("--best-min-frame-gap must be positive")
    if args.raw_contact_threshold < 0:
        raise ValueError("--raw-contact-threshold must be nonnegative")
    if args.batch_size <= 0 or args.dpi <= 0:
        raise ValueError("--batch-size and --dpi must be positive")
    if args.paper_raw_gamma <= 0 or args.paper_feature_gamma <= 0:
        raise ValueError("gamma values must be positive")
    if args.paper_patch_layout_sigma <= 0:
        raise ValueError("--paper-patch-layout-sigma must be positive")
    if not 0.0 <= args.paper_patch_boundary_alpha <= 1.0:
        raise ValueError("--paper-patch-boundary-alpha must be in [0, 1]")
    if args.paper_patch_boundary_sigma <= 0:
        raise ValueError("--paper-patch-boundary-sigma must be positive")

    repo = args.repo_id.expanduser().resolve()
    params = args.params.expanduser().resolve()
    assets_dir = args.assets_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply exactly the requested v8 appearance.
    _v8._PAPER_RAW_SCALE = args.paper_raw_scale
    _v8._PAPER_FEATURE_SCALE = args.paper_feature_scale
    _v8._PAPER_RAW_GAMMA = args.paper_raw_gamma
    _v8._PAPER_FEATURE_GAMMA = args.paper_feature_gamma
    _v8._PAPER_PATCH_LAYOUT_SIGMA = args.paper_patch_layout_sigma
    _v8._PAPER_PATCH_BOUNDARY_STYLE = args.paper_patch_boundary
    _v8._PAPER_PATCH_BOUNDARY_ALPHA = args.paper_patch_boundary_alpha
    _v8._PAPER_PATCH_BOUNDARY_SIGMA = args.paper_patch_boundary_sigma

    if args.paper_raw_scale != "frame" or args.paper_feature_scale != "frame":
        print(
            "[v8-multi] note: for multi-episode static comparison, frame/frame "
            "scaling is recommended. Episode mode uses the candidate-pool fallback.",
            flush=True,
        )

    episode_indices = _v8._available_episode_indices(repo)
    task_names = _episode_task_names(repo)

    candidate_records: list[dict[str, object]] = []
    candidate_tactile: list[np.ndarray] = []
    skipped_no_contact: list[int] = []

    for ordinal, episode_index in enumerate(episode_indices, start=1):
        tactile, frame_indices, timestamps = _v8._base._read_episode(
            repo,
            episode_index,
        )
        positions, metrics, per_finger_energy = (
            _candidate_positions_with_required_fingers(
                tactile,
                threshold=args.raw_contact_threshold,
                limit=args.best_candidates_per_episode,
                min_gap=args.best_min_frame_gap,
                min_active_fingers=args.min_active_fingers,
            )
        )
        active_count, weakest, geometric, total = metrics

        if not positions:
            skipped_no_contact.append(int(episode_index))
        else:
            for position in positions:
                candidate_index = len(candidate_records)
                candidate_records.append(
                    {
                        "candidate_index": candidate_index,
                        "episode_index": int(episode_index),
                        "position": int(position),
                        "frame_index": int(frame_indices[position]),
                        "timestamp": float(timestamps[position]),
                        "task_name": task_names.get(
                            int(episode_index),
                            "unknown task",
                        ),
                        "raw_key": (
                            int(active_count[position]),
                            float(weakest[position]),
                            float(geometric[position]),
                            float(total[position]),
                        ),
                        "raw_per_finger_energy": np.asarray(
                            per_finger_energy[position],
                            dtype=np.float64,
                        ),
                    }
                )
                candidate_tactile.append(
                    np.asarray(tactile[position], dtype=np.float32)
                )

        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(episode_indices):
            print(
                f"[v8-multi] scanned {ordinal}/{len(episode_indices)} episodes; "
                f"retained {len(candidate_records)} candidates",
                flush=True,
            )

    if not candidate_records:
        raise RuntimeError(
            "No candidate satisfies the requested contact criterion. "
            "Lower --min-active-fingers or --raw-contact-threshold."
        )

    effort_mean, effort_std, norm_path = _v8._base._load_effort_norm(
        assets_dir,
        args.asset_id,
    )
    stacked_tactile = np.stack(candidate_tactile, axis=0)
    normalized = _v8._base._normalize_tactile(
        stacked_tactile,
        effort_mean,
        effort_std,
    )

    print(
        f"[v8-multi] loading model and inferring "
        f"{stacked_tactile.shape[0]} candidates once",
        flush=True,
    )
    model = _v8._load_selector_model(
        config_name=args.config_name,
        params=params,
    )
    arrays = _v8._base._predict_episode(
        model,
        normalized,
        args.batch_size,
    )

    best_by_episode: dict[int, dict[str, object]] = {}
    for candidate_index, record in enumerate(candidate_records):
        reconstruction = _v8._candidate_reconstruction_metrics(
            arrays,
            candidate_index,
        )
        selection_key = (
            *reconstruction["key"],
            *record["raw_key"],
        )
        alignment = _raw_target_alignment(
            record["raw_per_finger_energy"],
            arrays["target_strength"][candidate_index],
        )
        enriched = {
            **record,
            **reconstruction,
            **alignment,
            "selection_key": selection_key,
        }

        episode_index = int(record["episode_index"])
        previous = best_by_episode.get(episode_index)
        if previous is None or selection_key > previous["selection_key"]:
            best_by_episode[episode_index] = enriched

    ranked = sorted(
        best_by_episode.values(),
        key=lambda item: item["selection_key"],
        reverse=True,
    )
    selected = ranked[: min(args.top_episodes, len(ranked))]

    # Fallback scales are only used when the user requests a non-frame mode.
    raw_magnitude = np.linalg.norm(stacked_tactile, axis=-1)
    active_raw = raw_magnitude[
        raw_magnitude >= args.raw_contact_threshold
    ]
    raw_vmax = (
        float(np.percentile(active_raw, 99.0))
        if active_raw.size
        else 1.0
    )
    combined_strength = np.concatenate(
        [
            np.maximum(arrays["target_strength"], 0.0).reshape(-1),
            np.maximum(arrays["pred_strength"], 0.0).reshape(-1),
        ]
    )
    positive_strength = combined_strength[combined_strength > 0]
    strength_vmax = (
        float(np.percentile(positive_strength, 99.0))
        if positive_strength.size
        else 1.0
    )

    summary_records: list[dict[str, object]] = []
    for rank, record in enumerate(selected, start=1):
        candidate_index = int(record["candidate_index"])
        episode_index = int(record["episode_index"])
        frame_index = int(record["frame_index"])
        output_path = output_dir / (
            f"rank_{rank:02d}_episode_{episode_index:06d}_"
            f"frame_{frame_index:06d}.png"
        )

        _v8._plot_frame(
            raw_tactile=stacked_tactile,
            arrays=arrays,
            frame_position=candidate_index,
            actual_frame_index=frame_index,
            timestamp=float(record["timestamp"]),
            task_name=str(record["task_name"]),
            episode_index=episode_index,
            layout_dir=layout_dir,
            output_path=output_path,
            raw_threshold=args.raw_contact_threshold,
            raw_vmax=raw_vmax,
            strength_vmax=strength_vmax,
            show_patch_labels=args.show_patch_labels,
            patch_label_fontsize=args.patch_label_fontsize,
            dpi=args.dpi,
        )

        record_for_json = _json_safe_record(record)
        record_for_json.update(
            {
                "rank": rank,
                "output_path": str(output_path),
                "worst_quality": float(
                    np.min(record["finger_qualities"])
                ),
                "mean_quality": float(
                    np.mean(record["finger_qualities"])
                ),
                "mean_cosine": float(
                    np.mean(record["map_cosines"])
                ),
                "mean_scale_similarity": float(
                    np.mean(record["scale_similarities"])
                ),
            }
        )
        summary_records.append(record_for_json)

        print(
            f"[v8-multi] rank {rank:02d}: episode={episode_index}, "
            f"frame={frame_index}, "
            f"worst_q={record_for_json['worst_quality']:.3f}, "
            f"mean_q={record_for_json['mean_quality']:.3f}, "
            f"raw/target strongest match="
            f"{record_for_json['strongest_finger_match']}; "
            f"saved {output_path.name}",
            flush=True,
        )

    summary = {
        "repo_id": str(repo),
        "params": str(params),
        "config_name": args.config_name,
        "normalization": str(norm_path),
        "num_episodes_scanned": len(episode_indices),
        "num_episodes_with_candidates": len(best_by_episode),
        "skipped_episode_indices": skipped_no_contact,
        "num_candidates_inferred": len(candidate_records),
        "top_episodes_requested": args.top_episodes,
        "min_active_fingers": args.min_active_fingers,
        "raw_contact_threshold": args.raw_contact_threshold,
        "best_candidates_per_episode": args.best_candidates_per_episode,
        "best_min_frame_gap": args.best_min_frame_gap,
        "visual_settings": {
            "paper_raw_scale": args.paper_raw_scale,
            "paper_feature_scale": args.paper_feature_scale,
            "paper_raw_gamma": args.paper_raw_gamma,
            "paper_feature_gamma": args.paper_feature_gamma,
            "paper_patch_layout_sigma": args.paper_patch_layout_sigma,
            "paper_patch_boundary": args.paper_patch_boundary,
            "paper_patch_boundary_alpha": args.paper_patch_boundary_alpha,
            "paper_patch_boundary_sigma": args.paper_patch_boundary_sigma,
        },
        "selected": summary_records,
    }
    summary_path = output_dir / "multi_episode_selection.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(
        f"[v8-multi] rendered {len(selected)} distinct episodes to {output_dir}",
        flush=True,
    )
    print(f"[v8-multi] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()