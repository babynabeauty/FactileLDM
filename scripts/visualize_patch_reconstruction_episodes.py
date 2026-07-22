#!/usr/bin/env python3
"""Visualize held-out XHand patch reconstruction over complete episodes.

For one selected validation episode per task, every frame is rendered as:

    Raw tactile heatmap | GT patch strength/contact | Predicted strength/contact

The raw column stays in the sensor's physical scale. Encoder inputs and patch
targets use the exact effort mean/std normalization used during Stage-1
training. This distinction is intentional and recorded in each episode's
metadata.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import subprocess
from collections import defaultdict

import einops
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy

import visualize_single_frame_patch_encoder_heads as _single_vis


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _load_episode_filter(path: pathlib.Path) -> set[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = (
            data.get("episodes")
            or data.get("episode_indices")
            or data.get("episode_index")
            or data.get("val")
        )
    else:
        raise ValueError(f"Unsupported episode filter format: {path}")
    if episodes is None:
        raise ValueError(f"No episode list found in {path}")
    return {int(episode) for episode in episodes}


def _task_metadata(repo: pathlib.Path, selected_episodes: set[int]) -> dict[object, dict]:
    task_name_to_index: dict[str, int] = {}
    task_index_to_name: dict[int, str] = {}
    tasks_path = repo / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with tasks_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "task" in row and "task_index" in row:
                    task_name = str(row["task"])
                    task_index = int(row["task_index"])
                    task_name_to_index[task_name] = task_index
                    task_index_to_name[task_index] = task_name

    grouped: dict[object, dict] = {}
    episodes_path = repo / "meta" / "episodes.jsonl"
    with episodes_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row.get("episode_index", row.get("index")))
            if episode not in selected_episodes:
                continue
            if "task_index" in row:
                task_key: object = int(row["task_index"])
            elif row.get("tasks"):
                task_name = str(row["tasks"][0])
                task_key = task_name_to_index.get(task_name, task_name)
            elif "task" in row:
                task_name = str(row["task"])
                task_key = task_name_to_index.get(task_name, task_name)
            else:
                task_key = 0
            task_name = task_index_to_name.get(task_key, None) if isinstance(task_key, int) else str(task_key)
            if task_name is None:
                if row.get("tasks"):
                    task_name = str(row["tasks"][0])
                elif "task" in row:
                    task_name = str(row["task"])
                else:
                    task_name = f"task_{task_key}"
            item = grouped.setdefault(task_key, {"task_name": task_name, "episodes": []})
            item["episodes"].append(episode)

    for item in grouped.values():
        item["episodes"] = sorted(set(item["episodes"]))
    return grouped


def _episode_file(repo: pathlib.Path, episode_index: int) -> pathlib.Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"episode_{episode_index:06d}.parquet not found under {repo}")
    return candidates[0]


def _read_episode(repo: pathlib.Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _episode_file(repo, episode_index)
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = ["observation.state"]
    if "frame_index" in available:
        columns.append("frame_index")
    if "timestamp" in available:
        columns.append("timestamp")
    table = parquet.read(columns=columns)
    states = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
    tactile = np.stack([_single_vis._extract_raw_tactile(state) for state in states], axis=0)
    frame_indices = (
        np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if "frame_index" in table.column_names
        else np.arange(states.shape[0], dtype=np.int64)
    )
    timestamps = (
        np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
        if "timestamp" in table.column_names
        else frame_indices.astype(np.float32) / 15.0
    )
    return tactile, frame_indices, timestamps


def _contact_richness(tactile: np.ndarray, threshold: float) -> float:
    magnitude = np.linalg.norm(tactile, axis=-1)
    frame_energy = np.sum(np.where(magnitude > threshold, magnitude, 0.0), axis=(1, 2))
    keep = max(1, int(np.ceil(0.1 * frame_energy.size)))
    return float(np.mean(np.partition(frame_energy, -keep)[-keep:]))


def _select_episodes(
    repo: pathlib.Path,
    grouped: dict[object, dict],
    *,
    selection: str,
    seed: int,
    raw_contact_threshold: float,
) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for task_key, item in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        candidates = list(item["episodes"])
        if not candidates:
            continue
        scores = {}
        if selection == "first":
            episode = candidates[0]
        elif selection == "random":
            episode = rng.choice(candidates)
        elif selection == "max_contact":
            for candidate in candidates:
                tactile, _, _ = _read_episode(repo, candidate)
                scores[candidate] = _contact_richness(tactile, raw_contact_threshold)
            episode = max(candidates, key=lambda candidate: (scores[candidate], -candidate))
        else:
            raise ValueError(f"Unsupported selection mode: {selection}")
        selected.append(
            {
                "task_key": task_key,
                "task_name": item["task_name"],
                "episode_index": episode,
                "candidate_episodes": candidates,
                "contact_richness": scores.get(episode),
            }
        )
    return selected


def _normalize_tactile(tactile: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    feature_shape = tactile.shape[1:]
    if mean.size == feature_shape[-1]:
        return ((tactile - mean) / (std + 1e-6)).astype(np.float32)
    if mean.size == feature_shape[-2] * feature_shape[-1]:
        flat = tactile.reshape(tactile.shape[0], feature_shape[0], -1)
        return ((flat - mean) / (std + 1e-6)).reshape(tactile.shape).astype(np.float32)
    if mean.size == int(np.prod(feature_shape)):
        flat = tactile.reshape(tactile.shape[0], -1)
        return ((flat - mean) / (std + 1e-6)).reshape(tactile.shape).astype(np.float32)
    raise ValueError(f"Cannot apply effort stats of length {mean.size} to tactile shape {feature_shape}")


def _load_effort_norm(assets_dir: pathlib.Path, asset_id: str) -> tuple[np.ndarray, np.ndarray, pathlib.Path]:
    norm_dir = assets_dir.resolve() / asset_id
    norm_stats = _normalize.load(norm_dir)
    if "effort" not in norm_stats:
        raise KeyError(f"No effort statistics in {norm_dir / 'norm_stats.json'}")
    effort = norm_stats["effort"]
    return np.asarray(effort.mean), np.asarray(effort.std), norm_dir / "norm_stats.json"


def _load_model(config_name: str, params: pathlib.Path):
    config = _config.get_config(config_name)
    restored = _model.restore_params(params, restore_type=np.ndarray)
    model = config.model.load(restored)
    model.eval()
    return model


def _model_batch(model, normalized_tactile: jax.Array) -> dict[str, jax.Array]:
    effort = normalized_tactile[:, None, ...].astype(jnp.float32)
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)
    target_dist, target_summary, target_contact = model.patch_encoder.patch_reconstruction_targets(effort)

    pred_dist = jax.nn.softmax(model.patch_distribution_head(tokens).astype(jnp.float32), axis=-1)
    pred_summary = einops.rearrange(
        model.patch_summary_head(tokens),
        "b t f (r c) -> b t f r c",
        r=model.num_patches,
        c=model.summary_dim,
    ).astype(jnp.float32)
    pred_contact = jax.nn.sigmoid(model.patch_contact_head(tokens).astype(jnp.float32))
    return {
        "target_dist": target_dist[:, 0].astype(jnp.float32),
        "target_summary": target_summary[:, 0].astype(jnp.float32),
        "target_contact": target_contact[:, 0].astype(jnp.float32),
        "target_strength": target_summary[:, 0, ..., -1].astype(jnp.float32),
        "pred_dist": pred_dist[:, 0],
        "pred_summary": pred_summary[:, 0],
        "pred_contact": pred_contact[:, 0],
        "pred_strength": pred_summary[:, 0, ..., -1],
    }


def _predict_episode(model, normalized_tactile: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    predict = jax.jit(lambda batch: _model_batch(model, batch))
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, normalized_tactile.shape[0], batch_size):
        batch = jnp.asarray(normalized_tactile[start : start + batch_size], dtype=jnp.float32)
        output = jax.device_get(predict(batch))
        for key, value in output.items():
            chunks[key].append(np.asarray(value, dtype=np.float32))
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _patch_centers(layout_dir: pathlib.Path, finger_idx: int) -> tuple[np.ndarray, np.ndarray]:
    coords, _ = _single_vis._taxel_coords(layout_dir, finger_idx)
    patch_ids = np.asarray(
        _single_vis.AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120)[finger_idx],
        dtype=np.int32,
    )
    centers = np.stack([np.mean(coords[patch_ids == patch_id], axis=0) for patch_id in range(5)], axis=0)
    return centers, patch_ids


def _annotate_patch_panel(
    ax,
    *,
    layout_dir: pathlib.Path,
    finger_idx: int,
    strength: np.ndarray,
    contact: np.ndarray,
    predicted: bool,
) -> None:
    centers, _ = _patch_centers(layout_dir, finger_idx)
    for patch_id, center in enumerate(centers):
        if predicted:
            text = f"p={float(contact[patch_id]):.2f}\ns={float(strength[patch_id]):.2f}"
        else:
            symbol = "C" if float(contact[patch_id]) >= 0.5 else "-"
            text = f"{symbol}\ns={float(strength[patch_id]):.2f}"
        ax.text(
            float(center[0]),
            float(center[1]),
            text,
            ha="center",
            va="center",
            fontsize=5.4,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "edgecolor": "none", "alpha": 0.45},
            zorder=8,
        )


def _plot_frame(
    *,
    raw_tactile: np.ndarray,
    arrays: dict[str, np.ndarray],
    frame_position: int,
    actual_frame_index: int,
    timestamp: float,
    task_name: str,
    episode_index: int,
    layout_dir: pathlib.Path,
    output_path: pathlib.Path,
    raw_threshold: float,
    raw_vmax: float,
    strength_vmax: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(5, 3, figsize=(8.8, 12.4), constrained_layout=True)
    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        _single_vis._draw_raw_taxel_force(
            axes[finger_idx, 0],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            tactile=raw_tactile[frame_position],
            threshold=raw_threshold,
            cmap_name="turbo",
            vmax=raw_vmax,
            title="",
        )
        target_strength = arrays["target_strength"][frame_position, finger_idx]
        target_contact = arrays["target_contact"][frame_position, finger_idx]
        pred_strength = arrays["pred_strength"][frame_position, finger_idx]
        pred_contact = arrays["pred_contact"][frame_position, finger_idx]

        _single_vis._draw_patch_values(
            axes[finger_idx, 1],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=np.maximum(target_strength, 0.0),
            cmap_name="turbo",
            vmin=0.0,
            vmax=strength_vmax,
            title="",
        )
        _annotate_patch_panel(
            axes[finger_idx, 1],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            strength=target_strength,
            contact=target_contact,
            predicted=False,
        )

        _single_vis._draw_patch_values(
            axes[finger_idx, 2],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=np.maximum(pred_strength, 0.0),
            cmap_name="turbo",
            vmin=0.0,
            vmax=strength_vmax,
            title="",
        )
        _annotate_patch_panel(
            axes[finger_idx, 2],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            strength=pred_strength,
            contact=pred_contact,
            predicted=True,
        )

        axes[finger_idx, 0].text(
            -0.08,
            0.5,
            finger_name,
            transform=axes[finger_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    axes[0, 0].set_title("Raw tactile\nheatmap", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("GT patch\nstrength/contact", fontsize=10, fontweight="bold")
    axes[0, 2].set_title("Predicted patch\nstrength/contact", fontsize=10, fontweight="bold")

    raw_map = matplotlib.cm.ScalarMappable(
        cmap="turbo", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(raw_vmax, 1e-6))
    )
    strength_map = matplotlib.cm.ScalarMappable(
        cmap="turbo", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(strength_vmax, 1e-6))
    )
    raw_map.set_array([])
    strength_map.set_array([])
    raw_bar = fig.colorbar(raw_map, ax=axes[:, 0], fraction=0.018, pad=0.01)
    raw_bar.set_label("raw force magnitude", fontsize=8)
    patch_bar = fig.colorbar(strength_map, ax=axes[:, 1:], fraction=0.012, pad=0.01)
    patch_bar.set_label("normalized patch strength", fontsize=8)

    fig.suptitle(
        f"{task_name}\nepisode {episode_index}, frame {actual_frame_index}, t={timestamp:.2f}s",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.002,
        "GT: C = contact, - = no contact. Prediction: p = contact probability, s = patch strength.",
        ha="center",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _make_video(frames_dir: pathlib.Path, output_path: pathlib.Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; skipping video generation", flush=True)
        return False
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=pathlib.Path, default=pathlib.Path("data/taskall-2"))
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="xhand_patch_tactile_encoder_pretrain")
    parser.add_argument(
        "--filter-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/episode_splits/taskall-2_encoder_final_10pct_seed42/val_episodes.json"),
    )
    parser.add_argument("--assets-dir", type=pathlib.Path, default=pathlib.Path("assets/pi0_xhand_tactile_structured_raw_dual_ae"))
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/patch_reconstruction_visualization"))
    parser.add_argument("--layout-dir", type=pathlib.Path, default=_single_vis.DEFAULT_LAYOUT_DIR)
    parser.add_argument("--selection", choices=("first", "random", "max_contact"), default="max_contact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--raw-contact-threshold", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--make-video", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.frame_stride <= 0:
        raise ValueError("batch-size and frame-stride must be positive")
    repo = args.repo_id.expanduser().resolve()
    filter_path = args.filter_path.expanduser().resolve()
    params = args.params.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    asset_id = args.asset_id or repo.name

    held_out = _load_episode_filter(filter_path)
    grouped = _task_metadata(repo, held_out)
    selected = _select_episodes(
        repo,
        grouped,
        selection=args.selection,
        seed=args.seed,
        raw_contact_threshold=args.raw_contact_threshold,
    )
    if not selected:
        raise RuntimeError("No validation episodes were selected")

    effort_mean, effort_std, norm_path = _load_effort_norm(args.assets_dir, asset_id)
    model = _load_model(args.config_name, params)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": str(repo),
        "params": str(params),
        "config_name": args.config_name,
        "filter_path": str(filter_path),
        "normalization": str(norm_path),
        "selection": args.selection,
        "seed": args.seed,
        "episodes": selected,
    }
    (output_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for selected_item in selected:
        episode_index = int(selected_item["episode_index"])
        task_name = str(selected_item["task_name"])
        task_key = selected_item["task_key"]
        print(f"Processing task={task_key} episode={episode_index}: {task_name}", flush=True)
        raw_tactile, frame_indices, timestamps = _read_episode(repo, episode_index)
        normalized_tactile = _normalize_tactile(raw_tactile, effort_mean, effort_std)
        arrays = _predict_episode(model, normalized_tactile, args.batch_size)

        episode_dir = output_root / f"task_{str(task_key).replace('/', '_')}_episode_{episode_index:06d}"
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episode_dir / "patch_reconstruction_data.npz",
            raw_tactile=raw_tactile,
            normalized_tactile=normalized_tactile,
            frame_indices=frame_indices,
            timestamps=timestamps,
            **arrays,
        )

        raw_magnitude = np.linalg.norm(raw_tactile, axis=-1)
        active_raw = raw_magnitude[raw_magnitude > args.raw_contact_threshold]
        raw_vmax = float(np.percentile(active_raw, 99.0)) if active_raw.size else 1.0
        combined_strength = np.concatenate(
            [np.maximum(arrays["target_strength"], 0.0).reshape(-1), np.maximum(arrays["pred_strength"], 0.0).reshape(-1)]
        )
        positive_strength = combined_strength[combined_strength > 0]
        strength_vmax = float(np.percentile(positive_strength, 99.0)) if positive_strength.size else 1.0

        rendered = []
        for output_frame, frame_position in enumerate(range(0, raw_tactile.shape[0], args.frame_stride)):
            output_path = frames_dir / f"frame_{output_frame:06d}.png"
            _plot_frame(
                raw_tactile=raw_tactile,
                arrays=arrays,
                frame_position=frame_position,
                actual_frame_index=int(frame_indices[frame_position]),
                timestamp=float(timestamps[frame_position]),
                task_name=task_name,
                episode_index=episode_index,
                layout_dir=layout_dir,
                output_path=output_path,
                raw_threshold=args.raw_contact_threshold,
                raw_vmax=raw_vmax,
                strength_vmax=strength_vmax,
                dpi=args.dpi,
            )
            rendered.append(
                {
                    "output_frame": output_frame,
                    "source_position": frame_position,
                    "frame_index": int(frame_indices[frame_position]),
                    "timestamp": float(timestamps[frame_position]),
                    "path": str(output_path),
                }
            )
            if output_frame == 0 or (output_frame + 1) % 50 == 0:
                print(f"  rendered {output_frame + 1} frames", flush=True)

        video_path = episode_dir / "patch_reconstruction.mp4"
        video_created = args.make_video and _make_video(frames_dir, video_path, args.fps / args.frame_stride)
        metadata = {
            **selected_item,
            "num_source_frames": int(raw_tactile.shape[0]),
            "num_rendered_frames": len(rendered),
            "frame_stride": args.frame_stride,
            "raw_contact_threshold": args.raw_contact_threshold,
            "raw_vmax": raw_vmax,
            "normalized_strength_vmax": strength_vmax,
            "normalization": str(norm_path),
            "frames": rendered,
            "video": str(video_path) if video_created else None,
        }
        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"Saved patch reconstruction visualizations to {output_root}", flush=True)


if __name__ == "__main__":
    main()
