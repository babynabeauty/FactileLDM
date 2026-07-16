#!/usr/bin/env python3
"""Visualize pretrained XHand patch tactile encoder heads.

This script loads the Stage-1 `xhand_patch_tactile_encoder_pretrain` checkpoint,
reads the last N seconds of each local LeRobot episode, and plots the three
decoder heads against their reconstruction targets:

  1. patch distribution: where contact is concentrated among 5 patches
  2. patch contact: whether each patch is in contact
  3. patch strength: max force magnitude in each patch

Example:
  env/.venv/bin/python scripts/visualize_patch_encoder_heads.py \
    --dataset data/grasp_pipette_and_press_w_force_w_depth_0614_good_tactile_7ep \
    --params checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/<step>/params \
    --output-dir outputs/patch_encoder_heads_7ep
"""

from __future__ import annotations

import argparse
import json
import pathlib

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.tri as mtri
import numpy as np
import pyarrow.parquet as pq
import scipy.ndimage
import scipy.spatial

from openpi.models import model as _model
from openpi.models.tactile_tokenizer import AdaptiveFingertipPatchTokenizer
from openpi.policies import xhand_policy
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
PATCH_NAMES = ("tip", "center", "base", "left", "right")
DEFAULT_TACTILE_LAYOUT_DIR = pathlib.Path("Xhand1交付资料-带触觉/触觉传感器")


def _find_latest_params() -> pathlib.Path:
    roots = sorted(
        pathlib.Path("checkpoints/xhand_patch_tactile_encoder_pretrain").glob("*/[0-9]*/params"),
        key=lambda p: (p.parent.parent.name, int(p.parent.name)),
    )
    if not roots:
        raise FileNotFoundError(
            "Could not auto-find Stage-1 patch encoder params under "
            "checkpoints/xhand_patch_tactile_encoder_pretrain/*/*/params. "
            "Pass --params explicitly."
        )
    return roots[-1]


def _load_model(params_path: pathlib.Path, config_name: str):
    train_config = _config.get_config(config_name)
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    return train_config.model.load(params)


def _episode_files(dataset: pathlib.Path) -> list[pathlib.Path]:
    info_path = dataset / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    data_pattern = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    glob_pattern = data_pattern.replace("{episode_chunk:03d}", "*").replace("{episode_index:06d}", "*")
    files = sorted(dataset.glob(glob_pattern))
    if not files:
        files = sorted((dataset / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset}")
    return files


def _load_episode_raw_tactile(parquet_file: pathlib.Path) -> tuple[int, np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_file, columns=["observation.state", "episode_index", "timestamp"])
    state = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
    episode_values = _arrow_column_to_numpy(table["episode_index"])
    episode_index = int(np.asarray(episode_values[0]).item())
    if "timestamp" in table.column_names:
        timestamps = _arrow_column_to_numpy(table["timestamp"]).astype(np.float32).reshape(-1)
    else:
        timestamps = np.arange(state.shape[0], dtype=np.float32) / 15.0

    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(state[:, start:end].reshape(state.shape[0], xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    tactile = np.stack(chunks, axis=1).astype(np.float32)  # [T, 5, 120, 3]
    return episode_index, timestamps, tactile


def _forward_heads(model, tactile: np.ndarray):
    effort = jnp.asarray(tactile[None], dtype=jnp.float32)
    times = jnp.zeros((tactile.shape[0],), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(
        effort,
        times,
        future=False,
        include_temporal=False,
    )

    dist_logits = model.patch_distribution_head(tokens)
    pred_dist = jax.nn.softmax(dist_logits.astype(jnp.float32), axis=-1)

    pred_summary = model.patch_summary_head(tokens)
    pred_summary = einops.rearrange(
        pred_summary,
        "b t f (r c) -> b t f r c",
        r=model.num_patches,
        c=model.summary_dim,
    )

    contact_logits = model.patch_contact_head(tokens)
    pred_contact = jax.nn.sigmoid(contact_logits.astype(jnp.float32))

    target_dist, target_summary, target_contact = model.patch_encoder.patch_reconstruction_targets(effort)

    return {
        "pred_dist": np.asarray(pred_dist[0]),
        "target_dist": np.asarray(target_dist[0]),
        "pred_contact": np.asarray(pred_contact[0]),
        "target_contact": np.asarray(target_contact[0]),
        "pred_strength": np.asarray(pred_summary[0, ..., -1]),
        "target_strength": np.asarray(target_summary[0, ..., -1]),
    }


def _gt_patch_targets(
    tactile: np.ndarray,
    *,
    contact_threshold: float = 0.5,
    contact_temperature: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute the same GT patch targets as PatchInformedFingerTokenizer, in numpy."""
    forces = tactile.astype(np.float32)  # [T, F, P, 3]
    if forces.ndim != 4 or forces.shape[1:] != (5, 120, 3):
        raise ValueError(f"Expected tactile [T,5,120,3], got {forces.shape}")

    magnitude = np.linalg.norm(forces, axis=-1)  # [T,F,P]
    gate = 1.0 / (1.0 + np.exp(-((magnitude - contact_threshold) / max(contact_temperature, 1e-6))))

    patch_ids = np.asarray(AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120), dtype=np.int32)
    patch_masks = np.eye(5, dtype=np.float32)[patch_ids]  # [F,P,R]
    patch_masks = np.transpose(patch_masks, (0, 2, 1))  # [F,R,P]

    masked_gate = gate[:, :, None, :] * patch_masks[None, :, :, :]  # [T,F,R,P]
    gate_sum = np.sum(masked_gate, axis=-1)
    gate_denom = np.maximum(gate_sum, 1e-6)
    gated_force_mean = np.einsum("tfrp,tfpc->tfrc", masked_gate, forces) / gate_denom[..., None]

    abs_forces = np.abs(forces)
    patch_abs_max = np.max(
        np.where(patch_masks[None, :, :, :, None] > 0, abs_forces[:, :, None, :, :], 0.0),
        axis=-2,
    )
    patch_strength = np.max(
        np.where(patch_masks[None, :, :, :] > 0, magnitude[:, :, None, :], 0.0),
        axis=-1,
    )

    contact_mask = patch_strength > contact_threshold
    active_strength = np.where(contact_mask, patch_strength, 0.0)
    strength_sum = np.sum(active_strength, axis=-1, keepdims=True)
    uniform = np.full_like(active_strength, 1.0 / 5.0)
    distribution = np.where(strength_sum > 1e-6, active_strength / np.maximum(strength_sum, 1e-6), uniform)
    summary = np.concatenate([gated_force_mean, patch_abs_max, patch_strength[..., None]], axis=-1)
    return {
        "target_dist": distribution.astype(np.float32),
        "target_contact": contact_mask.astype(np.float32),
        "target_strength": summary[..., -1].astype(np.float32),
    }


def _flatten_patch_matrix(values: np.ndarray) -> np.ndarray:
    # [T, F, R] -> [F*R, T]
    return values.transpose(1, 2, 0).reshape(values.shape[1] * values.shape[2], values.shape[0])


def _plot_episode(
    *,
    episode_index: int,
    timestamps: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_dir: pathlib.Path,
):
    labels = [f"{finger}:{patch}" for finger in FINGER_NAMES for patch in PATCH_NAMES]
    rel_time = timestamps - timestamps[0]
    extent = [float(rel_time[0]), float(rel_time[-1]), len(labels) - 0.5, -0.5]

    panels = [
        ("Patch distribution", "pred_dist", "target_dist", "viridis", 0.0, 1.0),
        ("Patch contact prob/mask", "pred_contact", "target_contact", "magma", 0.0, 1.0),
        (
            "Patch strength",
            "pred_strength",
            "target_strength",
            "plasma",
            0.0,
            float(max(np.max(arrays["pred_strength"]), np.max(arrays["target_strength"]), 1e-6)),
        ),
    ]

    fig, axes = plt.subplots(len(panels), 2, figsize=(16, 11), constrained_layout=True)
    for row, (title, pred_key, target_key, cmap, vmin, vmax) in enumerate(panels):
        for col, (kind, key) in enumerate((("Pred", pred_key), ("GT", target_key))):
            ax = axes[row, col]
            matrix = _flatten_patch_matrix(arrays[key])
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
            ax.set_title(f"{title} - {kind}")
            ax.set_xlabel("time in last window (s)")
            ax.set_yticks(np.arange(len(labels)))
            if col == 0:
                ax.set_yticklabels(labels, fontsize=7)
            else:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.suptitle(f"Patch encoder decoder heads, episode {episode_index:06d}", fontsize=14)
    output_path = output_dir / f"episode_{episode_index:06d}_patch_encoder_heads.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_episode_gt_only(
    *,
    episode_index: int,
    timestamps: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_dir: pathlib.Path,
):
    labels = [f"{finger}:{patch}" for finger in FINGER_NAMES for patch in PATCH_NAMES]
    rel_time = timestamps - timestamps[0]
    extent = [float(rel_time[0]), float(rel_time[-1]), len(labels) - 0.5, -0.5]
    panels = [
        ("Patch contact distribution", "target_dist", "viridis", 0.0, 1.0),
        ("Patch contact mask", "target_contact", "magma", 0.0, 1.0),
        (
            "Patch contact strength",
            "target_strength",
            "plasma",
            0.0,
            float(max(np.max(arrays["target_strength"]), 1e-6)),
        ),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 10), constrained_layout=True)
    for ax, (title, key, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        matrix = _flatten_patch_matrix(arrays[key])
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title)
        ax.set_xlabel("time in last window (s)")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.suptitle(f"GT XHand patch tactile signals, episode {episode_index:06d}", fontsize=14)
    output_path = output_dir / f"episode_{episode_index:06d}_gt_patch_tactile.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _draw_patch_finger(ax, values: np.ndarray, *, title: str, cmap_name: str, vmin: float, vmax: float) -> None:
    """Draw one finger as five colored tactile patches.

    Patch order follows PATCH_NAMES:
      0=tip, 1=center, 2=base, 3=left, 4=right.
    """
    import matplotlib.patches as patches

    cmap = plt.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))

    # Simple anatomy-inspired schematic. It is not a geometric taxel layout;
    # it is a clean visual symbol for the five patch values.
    patch_shapes = {
        2: patches.Rectangle((0.28, 0.05), 0.44, 0.26),  # base
        1: patches.Rectangle((0.28, 0.31), 0.44, 0.36),  # center
        0: patches.FancyBboxPatch((0.28, 0.67), 0.44, 0.25, boxstyle="round,pad=0.02,rounding_size=0.12"),  # tip
        3: patches.Rectangle((0.07, 0.26), 0.21, 0.45),  # left
        4: patches.Rectangle((0.72, 0.26), 0.21, 0.45),  # right
    }
    for patch_id, shape in patch_shapes.items():
        shape.set_facecolor(cmap(norm(float(values[patch_id]))))
        shape.set_edgecolor("white")
        shape.set_linewidth(1.6)
        ax.add_patch(shape)

    for patch_id, (x, y) in {
        0: (0.50, 0.80),
        1: (0.50, 0.49),
        2: (0.50, 0.18),
        3: (0.18, 0.49),
        4: (0.82, 0.49),
    }.items():
        ax.text(x, y, PATCH_NAMES[patch_id], ha="center", va="center", fontsize=7, color="white")

    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def _read_measurement_points(path: pathlib.Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    points = data.get("measurement_points")
    if points is None:
        points = data.get("points")
    if points is None:
        raise ValueError(f"Could not find measurement_points/points in {path}")
    points = sorted(points, key=lambda item: int(item["point"]))
    coords = np.asarray([[float(p["x"]), float(p["y"]), float(p["z"])] for p in points], dtype=np.float32)
    if coords.shape != (120, 3):
        raise ValueError(f"Expected 120 points in {path}, got {coords.shape}")
    return coords


_TAXEL_LAYOUT_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _taxel_layout_for_finger(layout_dir: pathlib.Path, finger_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return official 2D taxel coordinates and patch ids for one finger."""
    key = (str(layout_dir.resolve()), finger_idx)
    if key in _TAXEL_LAYOUT_CACHE:
        return _TAXEL_LAYOUT_CACHE[key]

    patch_ids = np.asarray(AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120)[finger_idx], dtype=np.int32)
    if finger_idx == 0:
        path = layout_dir / "points_t30_right_hand_transformed.json"
        if not path.exists():
            path = layout_dir / "points.t30 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [0, 1]]  # T30: lateral x, distal y.
    else:
        path = layout_dir / "points_t16_transformed (1).json"
        if not path.exists():
            path = layout_dir / "points_t16_transformed (2).json"
        if not path.exists():
            path = layout_dir / "points.t16 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [1, 2]]  # T16 transformed: lateral y, distal z.

    _TAXEL_LAYOUT_CACHE[key] = (coords_2d.astype(np.float32), patch_ids)
    return _TAXEL_LAYOUT_CACHE[key]


def _draw_taxel_patch_finger(
    ax,
    values: np.ndarray,
    *,
    finger_idx: int,
    title: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    layout_dir: pathlib.Path,
) -> None:
    """Draw patch values on the official XHand taxel pad shape.

    Each taxel is assigned the value of its patch. The dense image is generated
    from nearest official taxel assignment and masked by the taxel convex hull,
    so the image shows a continuous pad-shaped patch heatmap instead of a
    scattered taxel plot.
    """
    coords, patch_ids = _taxel_layout_for_finger(layout_dir, finger_idx)
    point_values = values[patch_ids].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]
    pad_x = max(0.5, 0.08 * float(np.ptp(x)))
    pad_y = max(0.5, 0.08 * float(np.ptp(y)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(float(x.min() - pad_x), float(x.max() + pad_x), 180),
        np.linspace(float(y.min() - pad_y), float(y.max() + pad_y), 220),
    )

    tree = scipy.spatial.cKDTree(coords)
    _, nearest = tree.query(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    grid_values = point_values[nearest].reshape(grid_x.shape)

    hull = scipy.spatial.Delaunay(coords)
    inside = hull.find_simplex(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)) >= 0
    inside = inside.reshape(grid_x.shape)
    alpha = scipy.ndimage.gaussian_filter(inside.astype(np.float32), sigma=1.0)
    grid_values = np.ma.array(grid_values, mask=alpha < 0.2)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    ax.imshow(
        grid_values,
        origin="lower",
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-6),
        interpolation="bilinear",
    )
    tri = mtri.Triangulation(x, y)
    try:
        ax.tricontour(tri, np.ones_like(x), levels=[0.5], colors="white", linewidths=0.0)
    except Exception:
        pass

    for patch_id in range(5):
        patch_coords = coords[patch_ids == patch_id]
        center = np.mean(patch_coords, axis=0)
        ax.text(
            float(center[0]),
            float(center[1]),
            PATCH_NAMES[patch_id].capitalize() if patch_id == 0 else PATCH_NAMES[patch_id],
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            weight="bold",
            alpha=0.9,
            clip_on=False,
            path_effects=[patheffects.withStroke(linewidth=1.2, foreground="black", alpha=0.45)],
        )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def _plot_frame_patch_maps(
    *,
    episode_index: int,
    frame_idx: int,
    timestamp: float,
    rel_time: float,
    arrays: dict[str, np.ndarray],
    output_dir: pathlib.Path,
    source: str,
    layout: str,
    layout_dir: pathlib.Path,
) -> pathlib.Path:
    prefix = "target" if source == "target" else "pred"
    dist = arrays[f"{prefix}_dist"][frame_idx]  # [F,R]
    contact = arrays[f"{prefix}_contact"][frame_idx]
    strength = arrays[f"{prefix}_strength"][frame_idx]
    strength_vmax = float(max(np.max(arrays[f"{prefix}_strength"]), 1e-6))

    rows = [
        ("Distribution", dist, "viridis", 0.0, 1.0),
        ("Contact", contact, "magma", 0.0, 1.0),
        ("Strength", strength, "plasma", 0.0, strength_vmax),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.5), constrained_layout=True)
    for row_idx, (row_name, values, cmap, vmin, vmax) in enumerate(rows):
        for finger_idx, finger_name in enumerate(FINGER_NAMES):
            title = finger_name if row_idx == 0 else ""
            if layout == "taxel":
                _draw_taxel_patch_finger(
                    axes[row_idx, finger_idx],
                    values[finger_idx],
                    finger_idx=finger_idx,
                    title=title,
                    cmap_name=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    layout_dir=layout_dir,
                )
            else:
                _draw_patch_finger(
                    axes[row_idx, finger_idx],
                    values[finger_idx],
                    title=title,
                    cmap_name=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            if finger_idx == 0:
                axes[row_idx, finger_idx].text(
                    -0.25,
                    0.5,
                    row_name,
                    transform=axes[row_idx, finger_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    # Add one colorbar per row.
    for row_idx, (_, _, cmap, vmin, vmax) in enumerate(rows):
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row_idx, :], fraction=0.025, pad=0.01)

    fig.suptitle(
        f"Episode {episode_index:06d}, frame {frame_idx:04d}, t={rel_time:.2f}s in last window",
        fontsize=14,
    )
    episode_dir = output_dir / f"episode_{episode_index:06d}_frames"
    episode_dir.mkdir(parents=True, exist_ok=True)
    output_path = episode_dir / f"frame_{frame_idx:04d}_patch_map.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=pathlib.Path("data/grasp_pipette_and_press_w_force_w_depth_0614_good_tactile_7ep"),
    )
    parser.add_argument("--params", type=pathlib.Path, default=None)
    parser.add_argument("--config-name", default="xhand_patch_tactile_encoder_pretrain")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/patch_encoder_heads_7ep"))
    parser.add_argument("--last-seconds", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--framewise-patch-maps",
        action="store_true",
        help="Save one figure per frame: 3 rows of patch info x 5 finger subplots.",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Only used with --framewise-patch-maps.")
    parser.add_argument("--max-frames", type=int, default=None, help="Only used with --framewise-patch-maps.")
    parser.add_argument(
        "--framewise-layout",
        choices=("taxel", "schematic"),
        default="taxel",
        help="Framewise map style: official taxel-shaped heatmap or simple schematic patches.",
    )
    parser.add_argument(
        "--tactile-layout-dir",
        type=pathlib.Path,
        default=DEFAULT_TACTILE_LAYOUT_DIR,
        help="Directory containing official XHand T16/T30 tactile point JSON files.",
    )
    parser.add_argument(
        "--framewise-source",
        choices=("target", "pred"),
        default="target",
        help="Use target/GT values or model predictions for framewise maps.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Visualize GT patch targets directly without loading pretrained encoder params.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[visualize] dataset={dataset}")
    if args.gt_only:
        params = None
        print("[visualize] mode=gt-only")
    else:
        params = args.params.resolve() if args.params is not None else _find_latest_params().resolve()
        print(f"[visualize] params={params}")
    print(f"[visualize] output_dir={output_dir}")

    with (dataset / "meta" / "info.json").open() as f:
        info = json.load(f)
    fps = float(args.fps if args.fps is not None else info.get("fps", 15.0))
    window_frames = max(1, int(round(args.last_seconds * fps)))

    model = None if args.gt_only else _load_model(params, args.config_name)
    files = _episode_files(dataset)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]

    saved = []
    for parquet_file in files:
        episode_index, timestamps, tactile = _load_episode_raw_tactile(parquet_file)
        tactile = tactile[-window_frames:]
        timestamps = timestamps[-window_frames:]
        if args.gt_only:
            gt_arrays = _gt_patch_targets(tactile)
            arrays = {
                "pred_dist": gt_arrays["target_dist"],
                "target_dist": gt_arrays["target_dist"],
                "pred_contact": gt_arrays["target_contact"],
                "target_contact": gt_arrays["target_contact"],
                "pred_strength": gt_arrays["target_strength"],
                "target_strength": gt_arrays["target_strength"],
            }
        else:
            arrays = _forward_heads(model, tactile)
        np.savez_compressed(output_dir / f"episode_{episode_index:06d}_patch_encoder_heads.npz", **arrays, timestamps=timestamps)
        if args.framewise_patch_maps:
            frame_indices = np.arange(tactile.shape[0])
            frame_indices = frame_indices[:: max(1, args.frame_stride)]
            if args.max_frames is not None:
                frame_indices = frame_indices[: args.max_frames]
            episode_saved = []
            rel_times = timestamps - timestamps[0]
            for frame_idx in frame_indices:
                saved_path = _plot_frame_patch_maps(
                    episode_index=episode_index,
                    frame_idx=int(frame_idx),
                    timestamp=float(timestamps[frame_idx]),
                    rel_time=float(rel_times[frame_idx]),
                    arrays=arrays,
                    output_dir=output_dir,
                    source=args.framewise_source,
                    layout=args.framewise_layout,
                    layout_dir=args.tactile_layout_dir,
                )
                episode_saved.append(saved_path)
            saved.extend(episode_saved)
            print(f"[visualize] saved {len(episode_saved)} frame maps for episode {episode_index:06d}")
            continue

        if args.gt_only:
            saved_path = _plot_episode_gt_only(
                episode_index=episode_index,
                timestamps=timestamps,
                arrays=arrays,
                output_dir=output_dir,
            )
        else:
            saved_path = _plot_episode(
                episode_index=episode_index,
                timestamps=timestamps,
                arrays=arrays,
                output_dir=output_dir,
            )
        saved.append(saved_path)
        print(f"[visualize] saved {saved_path}")

    print(f"[visualize] done, saved {len(saved)} episode figures")


if __name__ == "__main__":
    main()
