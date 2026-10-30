#!/usr/bin/env python3
"""Paper-style 5-finger tactile reconstruction visualization (v7).

Place this file in ``scripts/`` beside:
    visualize_patch_reconstruction_episodes.py

This wrapper reuses the original data/model pipeline and replaces only the
per-frame renderer.  Compared with v1 it deliberately does *not* mask the
Target/Predicted maps by the U-shaped projected taxel support.

Visual semantics
----------------
Raw tactile:
    - square panel;
    - gray outside a contextual fingertip silhouette;
    - dark blue low-force fingertip background;
    - blue/green/yellow/red contact glow generated only from real taxel forces.

Target / Predicted:
    - square, low-resolution feature maps like the paper mock-up;
    - each finger is divided into the five official XHand patch regions;
    - every pixel inside one patch has exactly the same scalar strength;
    - no interpolation is applied to the Target/Predicted scalar values;
    - only the visual separator between regions is softly feathered;
    - GT and prediction share the same patch partition and color scale.

Extra command-line options consumed by this wrapper before the original parser:
    --auto-contact-frame
        With --episode-index, select the frame with the largest total tactile
        energy. This can be dominated by one finger.

    --auto-five-finger-frame
        With --episode-index, select the frame that prioritizes: (1) number of
        active fingers, (2) the weakest finger's contact energy, (3) geometric
        mean across fingers, and (4) total contact energy.

    --auto-five-finger-global
        Scan every episode under --repo-id and automatically select the best
        episode/frame using the same five-finger criterion. This is useful for
        choosing a publication example from a single-task dataset.

    --auto-best-reconstruction-frame
        With --episode-index, first keep contact-rich five-finger candidate
        frames, run the encoder/decoder on those candidates, and select the
        frame whose five displayed patch-magnitude reconstructions are best.

    --auto-best-reconstruction-global
        Scan the whole dataset, retain a small number of balanced five-finger
        candidates per episode, run inference once on the candidate batch, and
        select the strongest reconstruction example. Selection prioritizes the
        weakest finger, then the five-finger mean, so one excellent finger
        cannot hide four poor reconstructions.

    --best-candidates-per-episode INT
        Number of temporally separated raw-contact candidates retained from
        each episode before model inference (default: 4).

    --best-min-frame-gap INT
        Minimum source-frame separation between candidates from the same
        episode (default: 8).

    --paper-feature-sigma FLOAT
        Deprecated compatibility option. It is accepted but ignored because
        v7 renders piecewise-constant official patch values.

    --paper-patch-layout-sigma FLOAT
        Smoothness used only to turn official taxel patch IDs into clean region
        geometry (default: 0.060). It never changes the five patch values.

    --paper-patch-boundary {soft,hard,none}
        Visual treatment of internal patch separators (default: soft).

    --paper-patch-boundary-alpha FLOAT
        Maximum opacity of the soft separator (default: 0.18).

    --paper-patch-boundary-sigma FLOAT
        Feather width in rendered pixels for the soft separator (default: 1.50).

    --paper-feature-scale {frame,episode}
        Feature color scaling.  ``frame`` (default) gives the clearest static
        paper figure; ``episode`` is temporally consistent for videos.

    --paper-raw-scale {frame,episode}
        Raw-force color scaling. ``frame`` (default) maps the current frame's
        true maximum force to red; ``episode`` preserves a constant scale.

    --paper-raw-gamma FLOAT
        Power-law color normalization for raw tactile force (default 0.45).
        Values below 1 expand low/mid-force contrast while keeping 0 exactly 0
        and the selected maximum exactly at the red endpoint.

    --paper-feature-gamma FLOAT
        Power-law normalization shared by Target and Predicted (default 0.70).
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import visualize_patch_reconstruction_episodes as _base


FINGER_TITLES = ("Thumb", "Index", "Middle", "Ring", "Little")

RAW_CONTACT_CMAP = LinearSegmentedColormap.from_list(
    "xhand_contact_glow",
    (
        (0.00, "#06182d"),
        (0.16, "#0a3562"),
        (0.34, "#006d77"),
        (0.52, "#16a66a"),
        (0.70, "#9bd93c"),
        (0.84, "#ffe234"),
        (0.93, "#ff7a18"),
        (1.00, "#d7191c"),
    ),
    N=256,
)
FEATURE_CMAP = matplotlib.colormaps["viridis"]

# Set by the wrapper-only command-line parser at the bottom of this file.
_PAPER_FEATURE_SCALE = "frame"
_PAPER_RAW_SCALE = "frame"
_PAPER_RAW_GAMMA = 0.45
_PAPER_FEATURE_GAMMA = 0.70
_PAPER_FEATURE_SIGMA = 0.145  # deprecated compatibility setting
_PAPER_PATCH_LAYOUT_SIGMA = 0.060
_PAPER_PATCH_BOUNDARY_STYLE = "soft"
_PAPER_PATCH_BOUNDARY_ALPHA = 0.18
_PAPER_PATCH_BOUNDARY_SIGMA = 1.50

# Reuse the model loaded by the best-reconstruction selector when base.main()
# runs, avoiding a second checkpoint restore.
_CACHED_MODEL = None
_CACHED_MODEL_CONFIG = None
_CACHED_MODEL_PARAMS = None

# Cache key includes the layout smoothing because it changes only region geometry.
_OFFICIAL_PATCH_RASTER_CACHE: dict[tuple[str, int, int, float], np.ndarray] = {}


def _finger_force(raw_frame: np.ndarray, finger_idx: int) -> np.ndarray:
    frame = np.asarray(raw_frame)
    if frame.ndim == 3:
        return np.asarray(frame[finger_idx], dtype=np.float32)
    if frame.ndim == 2 and frame.shape[-1] == 3:
        if frame.shape[0] % len(FINGER_TITLES) != 0:
            raise ValueError(f"Cannot split raw tactile shape {frame.shape} into five fingers")
        per_finger = frame.shape[0] // len(FINGER_TITLES)
        start = finger_idx * per_finger
        return np.asarray(frame[start : start + per_finger], dtype=np.float32)
    raise ValueError(f"Unsupported raw tactile frame shape: {frame.shape}")


def _normalize_points(
    points: np.ndarray,
    *,
    x_range: tuple[float, float] = (0.18, 0.82),
    y_range: tuple[float, float] = (0.10, 0.90),
) -> np.ndarray:
    """Map arbitrary 2-D layout coordinates into one canonical square."""
    points = np.asarray(points, dtype=np.float64)
    mins = np.min(points, axis=0)
    spans = np.ptp(points, axis=0)
    spans = np.where(spans < 1e-9, 1.0, spans)
    uv = (points - mins) / spans
    out = np.empty_like(uv)
    out[:, 0] = x_range[0] + uv[:, 0] * (x_range[1] - x_range[0])
    out[:, 1] = y_range[0] + uv[:, 1] * (y_range[1] - y_range[0])
    return out


def _finger_silhouette(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """A contextual rounded fingertip silhouette, not a sensor-coverage mask."""
    cx = 0.5
    bottom = 0.07
    shoulder = 0.61
    top = 0.94
    half_width = 0.355

    body = (yy >= bottom) & (yy <= shoulder) & (np.abs(xx - cx) <= half_width)
    radius_y = top - shoulder
    cap = (
        (yy > shoulder)
        & (yy <= top)
        & (((xx - cx) / half_width) ** 2 + ((yy - shoulder) / radius_y) ** 2 <= 1.0)
    )
    return body | cap


def _gaussian_peak_field(
    xx: np.ndarray,
    yy: np.ndarray,
    centers: np.ndarray,
    values: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Create a smooth field whose peak preserves the original maximum value.

    Each taxel/patch contributes a Gaussian with its own physical magnitude.
    Taking the pointwise maximum avoids the previous overlap normalization,
    which could make a 50 N taxel appear as only a small fraction of the
    colorbar range when surrounded by many zero-valued taxels.
    """
    centers = np.asarray(centers, dtype=np.float64)
    values = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if centers.shape[0] != values.size:
        raise ValueError(f"Center/value mismatch: {centers.shape} vs {values.shape}")
    if values.size == 0 or not np.any(values > 0):
        return np.zeros_like(xx, dtype=np.float64)

    positive = values > 0
    centers = centers[positive]
    values = values[positive]

    dx = xx[..., None] - centers[None, None, :, 0]
    dy = yy[..., None] - centers[None, None, :, 1]
    weight = np.exp(-(dx * dx + dy * dy) / (2.0 * max(sigma, 1e-8) ** 2))
    field = np.max(weight * values[None, None, :], axis=-1)

    # The regular rendering grid may not land exactly on a taxel center.
    # Rescale by a tiny numerical correction so the strongest visible point
    # reaches the true maximum and therefore maps to the colorbar endpoint.
    field_max = float(np.max(field))
    value_max = float(np.max(values))
    if field_max > 0:
        field *= value_max / field_max

    # Eliminate negligible Gaussian tails: no-force background is exactly zero.
    field[field < 1e-4 * max(value_max, 1e-12)] = 0.0
    return field


def _style_panel(ax: plt.Axes, *, facecolor: str, border: str = "#c8cdd2") -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.62)
        spine.set_edgecolor(border)


def _draw_raw_panel(
    ax: plt.Axes,
    *,
    layout_dir: pathlib.Path,
    finger_idx: int,
    raw_frame: np.ndarray,
    threshold: float,
    vmax: float,
) -> None:
    coords, _ = _base._single_vis._taxel_coords(layout_dir, finger_idx)
    coords = _normalize_points(np.asarray(coords, dtype=np.float64))
    magnitude = np.linalg.norm(_finger_force(raw_frame, finger_idx), axis=-1)
    if magnitude.size != coords.shape[0]:
        raise ValueError(
            f"Finger {finger_idx}: {magnitude.size} force values but {coords.shape[0]} coordinates"
        )

    # Threshold is used only to suppress sensor noise, not to alter real forces.
    active = np.where(magnitude >= threshold, magnitude, 0.0)

    n = 240
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    mask = _finger_silhouette(xx, yy)

    field = _gaussian_peak_field(xx, yy, coords, active, sigma=0.040)
    vmax = max(float(vmax), 1e-6)

    # Zero remains physically and visually zero (dark-blue endpoint).
    shown = np.ma.array(field, mask=~mask)

    ax.imshow(
        shown,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=RAW_CONTACT_CMAP,
        norm=PowerNorm(gamma=_PAPER_RAW_GAMMA, vmin=0.0, vmax=vmax),
        interpolation="bicubic",
        zorder=1,
    )
    ax.contour(
        xx,
        yy,
        mask.astype(np.float32),
        levels=[0.5],
        colors=["white"],
        linewidths=0.72,
        linestyles="--",
        alpha=0.82,
        zorder=3,
    )
    _style_panel(ax, facecolor="#55585d", border="#8f9499")


def _official_patch_raster(
    *,
    layout_dir: pathlib.Path,
    finger_idx: int,
    resolution: int = 320,
) -> np.ndarray:
    """Rasterize the official 120-taxel -> 5-patch assignment.

    The official tokenizer IDs remain the source of truth. To avoid the
    staircase-like Voronoi edges produced by assigning every output pixel to
    one nearest taxel, each official patch casts a smooth spatial vote from its
    member taxels. The winning patch ID defines the region.

    Important: this smooths only the *geometry of the separator*. The displayed
    Target/Predicted value remains exactly constant throughout each winning
    patch region.
    """
    layout_sigma = float(_PAPER_PATCH_LAYOUT_SIGMA)
    key = (
        str(layout_dir.expanduser().resolve()),
        int(finger_idx),
        int(resolution),
        round(layout_sigma, 8),
    )
    cached = _OFFICIAL_PATCH_RASTER_CACHE.get(key)
    if cached is not None:
        return cached

    coords, _ = _base._single_vis._taxel_coords(layout_dir, finger_idx)
    coords = _normalize_points(
        np.asarray(coords, dtype=np.float64),
        x_range=(0.18, 0.82),
        y_range=(0.10, 0.90),
    )
    patch_ids = np.asarray(
        _base._single_vis.AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(
            5,
            120,
        )[finger_idx],
        dtype=np.int16,
    ).reshape(-1)

    if coords.shape[0] != patch_ids.size:
        raise ValueError(
            f"Finger {finger_idx}: {coords.shape[0]} taxel coordinates but "
            f"{patch_ids.size} official patch IDs"
        )
    unique_ids = np.unique(patch_ids)
    if not np.array_equal(unique_ids, np.arange(5, dtype=unique_ids.dtype)):
        raise ValueError(
            f"Finger {finger_idx}: expected official patch IDs [0,1,2,3,4], "
            f"got {unique_ids.tolist()}"
        )

    axis = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    silhouette = _finger_silhouette(xx, yy)

    region_ids = np.full((resolution, resolution), -1, dtype=np.int16)
    sigma_sq = max(layout_sigma, 1e-6) ** 2
    rows_per_chunk = 24

    for row_start in range(0, resolution, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, resolution)
        chunk_x = xx[row_start:row_end]
        chunk_y = yy[row_start:row_end]
        patch_scores = np.empty(
            (row_end - row_start, resolution, 5),
            dtype=np.float64,
        )

        for patch_id in range(5):
            patch_coords = coords[patch_ids == patch_id]
            distance_sq = (
                (chunk_x[..., None] - patch_coords[None, None, :, 0]) ** 2
                + (chunk_y[..., None] - patch_coords[None, None, :, 1]) ** 2
            )
            # Mean voting avoids giving large patches an artificial advantage.
            patch_scores[..., patch_id] = np.mean(
                np.exp(-distance_sq / (2.0 * sigma_sq)),
                axis=-1,
            )

        region_ids[row_start:row_end] = np.argmax(
            patch_scores,
            axis=-1,
        ).astype(np.int16)

    region_ids[~silhouette] = -1
    _OFFICIAL_PATCH_RASTER_CACHE[key] = region_ids
    return region_ids


def _patch_region_centers(region_ids: np.ndarray) -> np.ndarray:
    """Return visual centroids [5, 2] in normalized panel coordinates."""
    height, width = region_ids.shape
    centers = np.zeros((5, 2), dtype=np.float64)
    for patch_id in range(5):
        rows, cols = np.where(region_ids == patch_id)
        if rows.size == 0:
            raise ValueError(f"Official patch {patch_id} has no rendered pixels")
        centers[patch_id, 0] = float(np.mean(cols) / max(width - 1, 1))
        centers[patch_id, 1] = float(np.mean(rows) / max(height - 1, 1))
    return centers



def _internal_patch_boundary(region_ids: np.ndarray) -> np.ndarray:
    """Return a one-pixel mask for boundaries between valid patch IDs."""
    region_ids = np.asarray(region_ids)
    valid = region_ids >= 0
    boundary = np.zeros(region_ids.shape, dtype=np.float64)

    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (region_ids[:, :-1] != region_ids[:, 1:])
    )
    boundary[:, :-1] = np.maximum(boundary[:, :-1], horizontal)
    boundary[:, 1:] = np.maximum(boundary[:, 1:], horizontal)

    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (region_ids[:-1, :] != region_ids[1:, :])
    )
    boundary[:-1, :] = np.maximum(boundary[:-1, :], vertical)
    boundary[1:, :] = np.maximum(boundary[1:, :], vertical)
    return boundary


def _gaussian_blur_2d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Small dependency-free separable Gaussian blur."""
    values = np.asarray(values, dtype=np.float64)
    sigma = float(sigma)
    if sigma <= 0:
        return values.copy()

    radius = max(1, int(np.ceil(3.0 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(positions * positions) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)

    padded_x = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    blurred_x = np.empty_like(values)
    for row in range(values.shape[0]):
        blurred_x[row] = np.convolve(
            padded_x[row],
            kernel,
            mode="valid",
        )

    padded_y = np.pad(blurred_x, ((radius, radius), (0, 0)), mode="edge")
    blurred = np.empty_like(values)
    for col in range(values.shape[1]):
        blurred[:, col] = np.convolve(
            padded_y[:, col],
            kernel,
            mode="valid",
        )
    return blurred


def _draw_soft_patch_boundaries(
    ax: plt.Axes,
    region_ids: np.ndarray,
) -> None:
    """Overlay a feathered separator without blending patch data values."""
    boundary = _internal_patch_boundary(region_ids)
    if not np.any(boundary):
        return

    feather = _gaussian_blur_2d(
        boundary,
        sigma=_PAPER_PATCH_BOUNDARY_SIGMA,
    )
    maximum = float(np.max(feather))
    if maximum > 0:
        feather /= maximum
    feather[region_ids < 0] = 0.0

    rgba = np.ones((*region_ids.shape, 4), dtype=np.float64)
    # Slightly warm neutral white is less visually harsh than pure white.
    rgba[..., 0] = 0.96
    rgba[..., 1] = 0.97
    rgba[..., 2] = 0.98
    rgba[..., 3] = np.clip(
        feather * _PAPER_PATCH_BOUNDARY_ALPHA,
        0.0,
        1.0,
    )

    ax.imshow(
        rgba,
        origin="lower",
        extent=(0, 1, 0, 1),
        interpolation="bilinear",
        zorder=3,
    )



def _draw_feature_panel(
    ax: plt.Axes,
    *,
    layout_dir: pathlib.Path,
    finger_idx: int,
    values: np.ndarray,
    vmax: float,
    annotate_values: bool,
) -> None:
    """Draw five official patch strengths with no within-patch interpolation."""
    values = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if values.size != 5:
        raise ValueError(
            f"Finger {finger_idx}: expected five patch strengths, got {values.shape}"
        )

    region_ids = _official_patch_raster(
        layout_dir=layout_dir,
        finger_idx=finger_idx,
        resolution=320,
    )

    field = np.full(region_ids.shape, np.nan, dtype=np.float64)
    for patch_id in range(5):
        field[region_ids == patch_id] = float(values[patch_id])
    shown = np.ma.masked_invalid(field)

    vmax = max(float(vmax), 1e-6)
    ax.imshow(
        shown,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap=FEATURE_CMAP,
        norm=PowerNorm(
            gamma=_PAPER_FEATURE_GAMMA,
            vmin=0.0,
            vmax=vmax,
        ),
        interpolation="nearest",
        zorder=1,
    )

    axis = np.linspace(0.0, 1.0, region_ids.shape[0], dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)

    if _PAPER_PATCH_BOUNDARY_STYLE == "soft":
        _draw_soft_patch_boundaries(ax, region_ids)
    elif _PAPER_PATCH_BOUNDARY_STYLE == "hard":
        for patch_id in range(5):
            ax.contour(
                xx,
                yy,
                (region_ids == patch_id).astype(np.float32),
                levels=[0.5],
                colors=["white"],
                linewidths=0.42,
                alpha=0.46,
                zorder=3,
            )
    elif _PAPER_PATCH_BOUNDARY_STYLE != "none":
        raise ValueError(
            f"Unsupported patch boundary style: {_PAPER_PATCH_BOUNDARY_STYLE}"
        )

    # Keep only a subtle fingertip outline; the internal boundaries carry the
    # five-patch structure without dominating the heatmap.
    ax.contour(
        xx,
        yy,
        (region_ids >= 0).astype(np.float32),
        levels=[0.5],
        colors=["white"],
        linewidths=0.58,
        alpha=0.52,
        zorder=4,
    )

    if annotate_values:
        centers = _patch_region_centers(region_ids)
        for patch_id, (center, value) in enumerate(
            zip(centers, values, strict=True)
        ):
            ax.text(
                float(center[0]),
                float(center[1]),
                f"P{patch_id}\n{float(value):.2f}",
                ha="center",
                va="center",
                fontsize=5.6,
                color="white",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.38,
                },
                zorder=5,
            )

    _style_panel(ax, facecolor="#eef0f2", border="#d0d2d5")


def _robust_positive_vmax(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return max(float(fallback), 1e-6)
    return max(float(np.percentile(positive, 99.0)), 1e-6)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return None
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


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
    show_patch_labels: bool,
    patch_label_fontsize: float,
    dpi: int,
) -> None:
    del patch_label_fontsize

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    target_strength = np.maximum(arrays["target_strength"][frame_position], 0.0)
    predicted_strength = np.maximum(arrays["pred_strength"][frame_position], 0.0)
    target_contact = np.clip(arrays["target_contact"][frame_position], 0.0, 1.0)
    predicted_contact = np.clip(arrays["pred_contact"][frame_position], 0.0, 1.0)

    # The model returns exactly five scalar strengths per finger. For the
    # force-three-head configuration, each strength is the L2 magnitude of
    # that patch's reconstructed 3-D force vector. The renderer must therefore
    # keep one constant value throughout each official patch region.
    target_display = target_strength
    predicted_display = predicted_strength

    frame_raw_magnitude = np.linalg.norm(raw_tactile[frame_position], axis=-1)
    frame_raw_active = frame_raw_magnitude[frame_raw_magnitude >= raw_threshold]
    frame_raw_vmax = (
        float(np.max(frame_raw_active)) if frame_raw_active.size else max(float(raw_vmax), 1e-6)
    )
    raw_plot_vmax = frame_raw_vmax if _PAPER_RAW_SCALE == "frame" else max(raw_vmax, 1e-6)

    combined_feature = np.concatenate([target_display.reshape(-1), predicted_display.reshape(-1)])
    positive_feature = combined_feature[np.isfinite(combined_feature) & (combined_feature > 0)]
    frame_feature_vmax = (
        float(np.max(positive_feature))
        if positive_feature.size
        else max(float(strength_vmax), 1e-6)
    )
    feature_plot_vmax = (
        frame_feature_vmax if _PAPER_FEATURE_SCALE == "frame" else max(strength_vmax, 1e-6)
    )

    fig = plt.figure(figsize=(11.45, 7.05), facecolor="white")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=6,
        width_ratios=(1, 1, 1, 1, 1, 0.055),
        left=0.115,
        right=0.935,
        bottom=0.105,
        top=0.900,
        wspace=0.092,
        hspace=0.105,
    )
    axes = np.empty((3, 5), dtype=object)
    for row in range(3):
        for col in range(5):
            axes[row, col] = fig.add_subplot(grid[row, col])

    for finger_idx, title in enumerate(FINGER_TITLES):
        _draw_raw_panel(
            axes[0, finger_idx],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            raw_frame=raw_tactile[frame_position],
            threshold=raw_threshold,
            vmax=raw_plot_vmax,
        )
        _draw_feature_panel(
            axes[1, finger_idx],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=target_display[finger_idx],
            vmax=feature_plot_vmax,
            annotate_values=show_patch_labels,
        )
        _draw_feature_panel(
            axes[2, finger_idx],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=predicted_display[finger_idx],
            vmax=feature_plot_vmax,
            annotate_values=show_patch_labels,
        )
        axes[0, finger_idx].set_title(title, fontsize=11.0, fontweight="bold", pad=6.5)

        # The visible panels contain five scalar patch magnitudes, therefore
        # the printed similarity is computed on those same five values.
        # Force-vector cosine is retained in the selector diagnostics, but is
        # not shown under a scalar-magnitude heatmap.
        sim = _cosine_similarity(
            target_display[finger_idx],
            predicted_display[finger_idx],
        )

        label = "no contact" if sim is None else f"cos. sim. {sim:.2f}"
        axes[2, finger_idx].text(
            0.5,
            -0.085,
            label,
            transform=axes[2, finger_idx].transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
            color="#6f767d" if sim is None else "#187a43",
            fontweight="normal" if sim is None else "bold",
        )

    # Shared colorbars.
    raw_cax = fig.add_subplot(grid[0, 5])
    raw_map = matplotlib.cm.ScalarMappable(
        cmap=RAW_CONTACT_CMAP,
        norm=PowerNorm(
            gamma=_PAPER_RAW_GAMMA,
            vmin=0.0,
            vmax=raw_plot_vmax,
        ),
    )
    raw_map.set_array([])
    raw_bar = fig.colorbar(raw_map, cax=raw_cax)
    raw_bar.outline.set_linewidth(0.5)
    raw_bar.ax.tick_params(labelsize=7.2, length=2.0, width=0.5)
    raw_bar.set_ticks(np.linspace(0.0, raw_plot_vmax, 5))
    raw_bar.set_label(r"$\|\mathbf{F}\|$ (N)", fontsize=8.0, labelpad=5)

    feature_cax = fig.add_subplot(grid[1:, 5])
    feature_map = matplotlib.cm.ScalarMappable(
        cmap=FEATURE_CMAP,
        norm=PowerNorm(
            gamma=_PAPER_FEATURE_GAMMA,
            vmin=0.0,
            vmax=feature_plot_vmax,
        ),
    )
    feature_map.set_array([])
    feature_bar = fig.colorbar(feature_map, cax=feature_cax)
    feature_bar.outline.set_linewidth(0.5)
    feature_bar.ax.tick_params(labelsize=7.2, length=2.0, width=0.5)
    feature_bar.set_ticks(np.linspace(0.0, feature_plot_vmax, 5))
    feature_bar.set_label("Patch feature magnitude (a.u.)", fontsize=8.0, labelpad=5)

    row_labels = (
        ("Raw tactile", "3D force magnitude"),
        ("Target", "encoder target"),
        ("Predicted", "reconstruction"),
    )
    for row, (primary, secondary) in enumerate(row_labels):
        bbox = axes[row, 0].get_position()
        y = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.086,
            y + 0.014,
            primary,
            ha="right",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color="#101214",
        )
        fig.text(
            0.086,
            y - 0.018,
            secondary,
            ha="right",
            va="center",
            fontsize=7.9,
            color="#666d73",
        )

    raw_bottom = axes[0, 0].get_position().y0
    target_top = axes[1, 0].get_position().y1
    separator_y = 0.5 * (raw_bottom + target_top)
    fig.add_artist(
        Line2D(
            [0.115, 0.918],
            [separator_y, separator_y],
            transform=fig.transFigure,
            color="#d8dde2",
            linewidth=0.62,
        )
    )

    # Compact legend for the gray area in the raw row.
    legend_y = 0.055
    fig.add_artist(
        Rectangle(
            (0.115, legend_y - 0.008),
            0.012,
            0.016,
            transform=fig.transFigure,
            facecolor="#55585d",
            edgecolor="#888d92",
            linewidth=0.45,
        )
    )
    fig.text(
        0.132,
        legend_y,
        "outside contextual fingertip silhouette",
        ha="left",
        va="center",
        fontsize=7.4,
        color="#666d73",
    )
    fig.text(
        0.918,
        legend_y,
        "Official 5-patch regions; piecewise-constant strength; "
        "Target/Predicted share the same scale.",
        ha="right",
        va="center",
        fontsize=7.4,
        color="#666d73",
    )

    fig.text(
        0.115,
        0.952,
        f"{task_name}  |  episode {episode_index}  |  frame {actual_frame_index}  |  t={timestamp:.2f}s",
        ha="left",
        va="center",
        fontsize=8.0,
        color="#737980",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor="white",
    )
    plt.close(fig)


def _pop_flag(argv: list[str], flag: str) -> bool:
    found = False
    while flag in argv:
        argv.remove(flag)
        found = True
    return found


def _pop_choice(argv: list[str], option: str, default: str, allowed: set[str]) -> str:
    if option not in argv:
        return default
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires one of: {sorted(allowed)}")
    value = argv[index + 1]
    del argv[index : index + 2]
    if value not in allowed:
        raise SystemExit(f"Invalid {option}={value!r}; choose one of {sorted(allowed)}")
    return value



def _pop_float(
    argv: list[str],
    option: str,
    default: float,
    *,
    minimum_exclusive: float | None = None,
) -> float:
    if option not in argv:
        return float(default)
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires a numeric value")
    raw = argv[index + 1]
    del argv[index : index + 2]
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{option} requires a numeric value, got {raw!r}") from exc
    if minimum_exclusive is not None and value <= minimum_exclusive:
        raise SystemExit(f"{option} must be > {minimum_exclusive}, got {value}")
    return value


def _option_value(argv: list[str], option: str, default: str | None = None) -> str | None:
    if option not in argv:
        return default
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires a value")
    return argv[index + 1]



def _five_finger_frame_metrics(
    tactile: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame metrics designed to favor simultaneous 5-finger contact.

    Parameters
    ----------
    tactile:
        Raw tactile array shaped [frames, fingers, taxels, xyz].
    threshold:
        Per-taxel force-magnitude threshold. Values below it contribute zero.

    Returns
    -------
    active_count:
        Number of fingers with nonzero thresholded contact energy.
    weakest_energy:
        Minimum thresholded energy across the five fingers.
    geometric_mean:
        Geometric mean of per-finger energy, with a tiny numerical floor.
    total_energy:
        Total thresholded energy across all fingers.
    """
    magnitude = np.linalg.norm(np.asarray(tactile), axis=-1)
    if magnitude.ndim != 3:
        raise ValueError(
            "Expected raw tactile magnitude shaped [frames, fingers, taxels], "
            f"got {magnitude.shape}"
        )
    active_magnitude = np.where(magnitude >= threshold, magnitude, 0.0)
    per_finger_energy = np.sum(active_magnitude, axis=-1)
    active_count = np.sum(per_finger_energy > 0.0, axis=-1).astype(np.int32)
    weakest_energy = np.min(per_finger_energy, axis=-1)
    geometric_mean = np.exp(
        np.mean(np.log(np.maximum(per_finger_energy, 1e-12)), axis=-1)
    )
    total_energy = np.sum(per_finger_energy, axis=-1)
    return active_count, weakest_energy, geometric_mean, total_energy


def _best_five_finger_position(
    tactile: np.ndarray,
    *,
    threshold: float,
) -> tuple[int, tuple[int, float, float, float], np.ndarray]:
    """Select the lexicographically best simultaneous-contact frame."""
    active_count, weakest, geometric, total = _five_finger_frame_metrics(
        tactile,
        threshold=threshold,
    )
    # np.lexsort uses the final key as primary. We prioritize active_count first,
    # then weakest finger, geometric mean, and finally total energy.
    order = np.lexsort((total, geometric, weakest, active_count))
    position = int(order[-1])

    magnitude = np.linalg.norm(np.asarray(tactile[position]), axis=-1)
    active_magnitude = np.where(magnitude >= threshold, magnitude, 0.0)
    per_finger_energy = np.sum(active_magnitude, axis=-1)
    metrics = (
        int(active_count[position]),
        float(weakest[position]),
        float(geometric[position]),
        float(total[position]),
    )
    return position, metrics, np.asarray(per_finger_energy, dtype=np.float64)


def _available_episode_indices(repo: pathlib.Path) -> list[int]:
    episodes_path = repo / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episode metadata not found: {episodes_path}")
    indices: list[int] = []
    with episodes_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            indices.append(int(row.get("episode_index", row.get("index"))))
    indices = sorted(set(indices))
    if not indices:
        raise RuntimeError(f"No episodes found in {episodes_path}")
    return indices


def _install_auto_five_finger_frame(argv: list[str]) -> None:
    if "--frame-index" in argv:
        raise SystemExit("Use either --auto-five-finger-frame or --frame-index, not both")

    repo_value = _option_value(argv, "--repo-id", "data/taskall-2")
    episode_value = _option_value(argv, "--episode-index")
    threshold_value = _option_value(argv, "--raw-contact-threshold", "1.0")
    if episode_value is None:
        raise SystemExit("--auto-five-finger-frame requires --episode-index")

    repo = pathlib.Path(str(repo_value)).expanduser().resolve()
    episode_index = int(str(episode_value))
    threshold = float(str(threshold_value))
    tactile, frame_indices, timestamps = _base._read_episode(repo, episode_index)
    best_position, metrics, per_finger_energy = _best_five_finger_position(
        tactile,
        threshold=threshold,
    )
    best_frame = int(frame_indices[best_position])
    active_count, weakest, geometric, total = metrics
    energies_text = ", ".join(
        f"{name.lower()}={energy:.2f}"
        for name, energy in zip(FINGER_TITLES, per_finger_energy, strict=True)
    )
    print(
        "[paper-v7] auto-selected five-finger frame: "
        f"episode={episode_index}, frame={best_frame}, "
        f"t={float(timestamps[best_position]):.2f}s, "
        f"active_fingers={active_count}/5, weakest={weakest:.3f}, "
        f"geomean={geometric:.3f}, total={total:.3f}; {energies_text}",
        flush=True,
    )
    argv.extend(["--frame-index", str(best_frame)])


def _install_auto_five_finger_global(argv: list[str]) -> None:
    if "--episode-index" in argv or "--frame-index" in argv:
        raise SystemExit(
            "--auto-five-finger-global chooses both episode and frame; "
            "do not also pass --episode-index or --frame-index"
        )

    repo_value = _option_value(argv, "--repo-id", "data/taskall-2")
    threshold_value = _option_value(argv, "--raw-contact-threshold", "1.0")
    repo = pathlib.Path(str(repo_value)).expanduser().resolve()
    threshold = float(str(threshold_value))

    best_record = None
    episode_indices = _available_episode_indices(repo)
    for ordinal, episode_index in enumerate(episode_indices, start=1):
        tactile, frame_indices, timestamps = _base._read_episode(repo, episode_index)
        position, metrics, per_finger_energy = _best_five_finger_position(
            tactile,
            threshold=threshold,
        )
        active_count, weakest, geometric, total = metrics
        key = (active_count, weakest, geometric, total)
        record = {
            "key": key,
            "episode_index": episode_index,
            "position": position,
            "frame_index": int(frame_indices[position]),
            "timestamp": float(timestamps[position]),
            "per_finger_energy": per_finger_energy,
        }
        if best_record is None or record["key"] > best_record["key"]:
            best_record = record
        if ordinal == 1 or ordinal % 20 == 0 or ordinal == len(episode_indices):
            print(
                f"[paper-v7] scanned {ordinal}/{len(episode_indices)} episodes",
                flush=True,
            )

    assert best_record is not None
    active_count, weakest, geometric, total = best_record["key"]
    energies_text = ", ".join(
        f"{name.lower()}={energy:.2f}"
        for name, energy in zip(
            FINGER_TITLES,
            best_record["per_finger_energy"],
            strict=True,
        )
    )
    print(
        "[paper-v7] globally selected five-finger example: "
        f"episode={best_record['episode_index']}, "
        f"frame={best_record['frame_index']}, "
        f"t={best_record['timestamp']:.2f}s, "
        f"active_fingers={active_count}/5, weakest={weakest:.3f}, "
        f"geomean={geometric:.3f}, total={total:.3f}; {energies_text}",
        flush=True,
    )
    argv.extend(
        [
            "--episode-index",
            str(best_record["episode_index"]),
            "--frame-index",
            str(best_record["frame_index"]),
        ]
    )


def _pop_int(
    argv: list[str],
    option: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    if option not in argv:
        return int(default)
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"{option} requires an integer value")
    raw = argv[index + 1]
    del argv[index : index + 2]
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{option} requires an integer value, got {raw!r}") from exc
    if value < minimum:
        raise SystemExit(f"{option} must be >= {minimum}, got {value}")
    return value


def _candidate_positions(
    tactile: np.ndarray,
    *,
    threshold: float,
    limit: int,
    min_gap: int,
) -> list[int]:
    """Choose balanced-contact candidates without near-duplicate neighboring frames."""
    active_count, weakest, geometric, total = _five_finger_frame_metrics(
        tactile,
        threshold=threshold,
    )
    # Primary order: all five fingers, then weakest finger, then balance/total.
    order = np.lexsort((total, geometric, weakest, active_count))[::-1]
    chosen: list[int] = []
    for raw_position in order:
        position = int(raw_position)
        if chosen and any(abs(position - previous) < min_gap for previous in chosen):
            continue
        chosen.append(position)
        if len(chosen) >= limit:
            break
    return chosen


def _symmetric_scale_similarity(target: np.ndarray, predicted: np.ndarray) -> float:
    """Return 1 for equal vector norms and approach 0 as their scales diverge."""
    target_norm = float(np.linalg.norm(np.asarray(target, dtype=np.float64).reshape(-1)))
    predicted_norm = float(np.linalg.norm(np.asarray(predicted, dtype=np.float64).reshape(-1)))
    denominator = target_norm + predicted_norm
    if denominator < 1e-10:
        return 0.0
    return float(2.0 * min(target_norm, predicted_norm) / denominator)


def _candidate_reconstruction_metrics(
    arrays: dict[str, np.ndarray],
    candidate_index: int,
) -> dict[str, object]:
    """Score the same five-patch magnitudes that are shown in the paper panels."""
    target = np.maximum(np.asarray(arrays["target_strength"][candidate_index]), 0.0)
    predicted = np.maximum(np.asarray(arrays["pred_strength"][candidate_index]), 0.0)

    map_cosines: list[float] = []
    scale_similarities: list[float] = []
    finger_qualities: list[float] = []
    relative_errors: list[float] = []
    force_cosines: list[float] = []

    for finger_idx in range(len(FINGER_TITLES)):
        target_finger = target[finger_idx]
        predicted_finger = predicted[finger_idx]
        cosine = _cosine_similarity(target_finger, predicted_finger)
        cosine_value = 0.0 if cosine is None else max(float(cosine), 0.0)
        scale_value = _symmetric_scale_similarity(target_finger, predicted_finger)

        target_norm = float(np.linalg.norm(target_finger))
        error = float(np.linalg.norm(predicted_finger - target_finger))
        relative_error = error / max(target_norm, 1e-8)

        # Shape and absolute scale are both necessary for a convincing panel.
        quality = cosine_value * scale_value

        map_cosines.append(cosine_value)
        scale_similarities.append(scale_value)
        finger_qualities.append(quality)
        relative_errors.append(relative_error)

        if "target_force" in arrays and "pred_force" in arrays:
            force_cos = _cosine_similarity(
                arrays["target_force"][candidate_index, finger_idx],
                arrays["pred_force"][candidate_index, finger_idx],
            )
            force_cosines.append(0.0 if force_cos is None else float(force_cos))

    qualities = np.asarray(finger_qualities, dtype=np.float64)
    map_cosines_array = np.asarray(map_cosines, dtype=np.float64)
    scales = np.asarray(scale_similarities, dtype=np.float64)
    errors = np.asarray(relative_errors, dtype=np.float64)

    # Lexicographic selection prevents cherry-picking a frame with one excellent
    # finger and several weak fingers. The worst finger is primary.
    key = (
        float(np.min(qualities)),
        float(np.mean(qualities)),
        float(np.min(map_cosines_array)),
        float(np.mean(map_cosines_array)),
        float(np.mean(scales)),
        -float(np.mean(errors)),
    )
    return {
        "key": key,
        "map_cosines": map_cosines_array,
        "scale_similarities": scales,
        "finger_qualities": qualities,
        "relative_errors": errors,
        "force_cosines": np.asarray(force_cosines, dtype=np.float64),
    }


def _load_selector_model(
    *,
    config_name: str,
    params: pathlib.Path,
):
    global _CACHED_MODEL, _CACHED_MODEL_CONFIG, _CACHED_MODEL_PARAMS
    model = _base._load_model(config_name, params)
    _CACHED_MODEL = model
    _CACHED_MODEL_CONFIG = str(config_name)
    _CACHED_MODEL_PARAMS = str(params.resolve())

    original_loader = _base._load_model

    def _reuse_model(requested_config: str, requested_params: pathlib.Path):
        if (
            _CACHED_MODEL is not None
            and str(requested_config) == _CACHED_MODEL_CONFIG
            and str(pathlib.Path(requested_params).resolve()) == _CACHED_MODEL_PARAMS
        ):
            print("[paper-v7] reusing checkpoint model loaded during selection", flush=True)
            return _CACHED_MODEL
        return original_loader(requested_config, requested_params)

    _base._load_model = _reuse_model
    return model


def _best_reconstruction_selection(
    argv: list[str],
    *,
    global_search: bool,
    candidates_per_episode: int,
    min_frame_gap: int,
) -> None:
    if "--frame-index" in argv:
        raise SystemExit(
            "Best-reconstruction selection chooses the frame; do not also pass --frame-index"
        )
    if global_search and "--episode-index" in argv:
        raise SystemExit(
            "--auto-best-reconstruction-global chooses the episode; "
            "do not also pass --episode-index"
        )

    repo = pathlib.Path(
        str(_option_value(argv, "--repo-id", "data/taskall-2"))
    ).expanduser().resolve()
    threshold = float(str(_option_value(argv, "--raw-contact-threshold", "1.0")))
    params_value = _option_value(argv, "--params")
    if params_value is None or not str(params_value).strip():
        raise SystemExit("--params is required for best-reconstruction selection")
    params = pathlib.Path(str(params_value)).expanduser().resolve()
    config_name = str(
        _option_value(
            argv,
            "--config-name",
            "xhand_patch_tactile_encoder_pretrain",
        )
    )
    assets_dir = pathlib.Path(
        str(
            _option_value(
                argv,
                "--assets-dir",
                "assets/pi0_xhand_tactile_structured_raw_dual_ae",
            )
        )
    ).expanduser().resolve()
    asset_id = _option_value(argv, "--asset-id", repo.name)
    asset_id = repo.name if asset_id is None else str(asset_id)
    batch_size = int(str(_option_value(argv, "--batch-size", "256")))

    if global_search:
        episode_indices = _available_episode_indices(repo)
    else:
        episode_value = _option_value(argv, "--episode-index")
        if episode_value is None:
            raise SystemExit(
                "--auto-best-reconstruction-frame requires --episode-index"
            )
        episode_indices = [int(str(episode_value))]

    candidate_records: list[dict[str, object]] = []
    candidate_tactile: list[np.ndarray] = []

    for ordinal, episode_index in enumerate(episode_indices, start=1):
        tactile, frame_indices, timestamps = _base._read_episode(repo, episode_index)
        positions = _candidate_positions(
            tactile,
            threshold=threshold,
            limit=candidates_per_episode,
            min_gap=min_frame_gap,
        )
        active_count, weakest, geometric, total = _five_finger_frame_metrics(
            tactile,
            threshold=threshold,
        )
        for position in positions:
            candidate_records.append(
                {
                    "episode_index": int(episode_index),
                    "position": int(position),
                    "frame_index": int(frame_indices[position]),
                    "timestamp": float(timestamps[position]),
                    "raw_key": (
                        int(active_count[position]),
                        float(weakest[position]),
                        float(geometric[position]),
                        float(total[position]),
                    ),
                }
            )
            candidate_tactile.append(np.asarray(tactile[position], dtype=np.float32))

        if global_search and (
            ordinal == 1 or ordinal % 20 == 0 or ordinal == len(episode_indices)
        ):
            print(
                f"[paper-v7] collected candidates from "
                f"{ordinal}/{len(episode_indices)} episodes",
                flush=True,
            )

    if not candidate_records:
        raise RuntimeError("No reconstruction-selection candidates were found")

    effort_mean, effort_std, _ = _base._load_effort_norm(assets_dir, asset_id)
    stacked_tactile = np.stack(candidate_tactile, axis=0)
    normalized = _base._normalize_tactile(stacked_tactile, effort_mean, effort_std)
    model = _load_selector_model(config_name=config_name, params=params)
    arrays = _base._predict_episode(model, normalized, batch_size)

    best_record = None
    for candidate_index, record in enumerate(candidate_records):
        reconstruction = _candidate_reconstruction_metrics(arrays, candidate_index)
        # Reconstruction quality is primary; balanced five-finger raw contact
        # is used only as a tie-breaker.
        selection_key = (*reconstruction["key"], *record["raw_key"])
        enriched = {
            **record,
            **reconstruction,
            "selection_key": selection_key,
        }
        if best_record is None or enriched["selection_key"] > best_record["selection_key"]:
            best_record = enriched

    assert best_record is not None
    per_finger_text = ", ".join(
        (
            f"{name.lower()}: cos={cosine:.3f}, "
            f"scale={scale:.3f}, q={quality:.3f}"
        )
        for name, cosine, scale, quality in zip(
            FINGER_TITLES,
            best_record["map_cosines"],
            best_record["scale_similarities"],
            best_record["finger_qualities"],
            strict=True,
        )
    )
    print(
        "[paper-v7] selected best balanced reconstruction: "
        f"episode={best_record['episode_index']}, "
        f"frame={best_record['frame_index']}, "
        f"t={best_record['timestamp']:.2f}s; "
        f"worst_q={float(np.min(best_record['finger_qualities'])):.3f}, "
        f"mean_q={float(np.mean(best_record['finger_qualities'])):.3f}, "
        f"mean_cos={float(np.mean(best_record['map_cosines'])):.3f}, "
        f"mean_scale={float(np.mean(best_record['scale_similarities'])):.3f}; "
        f"{per_finger_text}",
        flush=True,
    )

    if global_search:
        argv.extend(
            [
                "--episode-index",
                str(best_record["episode_index"]),
                "--frame-index",
                str(best_record["frame_index"]),
            ]
        )
    else:
        argv.extend(["--frame-index", str(best_record["frame_index"])])



def _install_auto_contact_frame(argv: list[str]) -> None:
    if "--frame-index" in argv:
        raise SystemExit("Use either --auto-contact-frame or --frame-index, not both")

    repo_value = _option_value(argv, "--repo-id", "data/taskall-2")
    episode_value = _option_value(argv, "--episode-index")
    threshold_value = _option_value(argv, "--raw-contact-threshold", "1.0")
    if episode_value is None:
        raise SystemExit("--auto-contact-frame requires --episode-index")

    repo = pathlib.Path(str(repo_value)).expanduser().resolve()
    episode_index = int(str(episode_value))
    threshold = float(str(threshold_value))
    tactile, frame_indices, timestamps = _base._read_episode(repo, episode_index)
    magnitude = np.linalg.norm(tactile, axis=-1)
    energy = np.sum(np.where(magnitude >= threshold, magnitude, 0.0), axis=tuple(range(1, magnitude.ndim)))
    best_position = int(np.argmax(energy))
    best_frame = int(frame_indices[best_position])
    print(
        "[paper-v7] auto-selected contact-rich frame: "
        f"episode={episode_index}, frame={best_frame}, "
        f"t={float(timestamps[best_position]):.2f}s, contact_energy={float(energy[best_position]):.3f}",
        flush=True,
    )
    argv.extend(["--frame-index", str(best_frame)])


# Patch only the renderer; all model/data logic remains in the original script.
_base._plot_frame = _plot_frame


if __name__ == "__main__":
    auto_contact = _pop_flag(sys.argv, "--auto-contact-frame")
    auto_five_finger = _pop_flag(sys.argv, "--auto-five-finger-frame")
    auto_five_finger_global = _pop_flag(sys.argv, "--auto-five-finger-global")
    auto_best_frame = _pop_flag(sys.argv, "--auto-best-reconstruction-frame")
    auto_best_global = _pop_flag(sys.argv, "--auto-best-reconstruction-global")
    selectors = (
        auto_contact,
        auto_five_finger,
        auto_five_finger_global,
        auto_best_frame,
        auto_best_global,
    )
    if sum(selectors) > 1:
        raise SystemExit(
            "Choose only one automatic selector: --auto-contact-frame, "
            "--auto-five-finger-frame, --auto-five-finger-global, "
            "--auto-best-reconstruction-frame, or "
            "--auto-best-reconstruction-global"
        )

    best_candidates_per_episode = _pop_int(
        sys.argv,
        "--best-candidates-per-episode",
        4,
        minimum=1,
    )
    best_min_frame_gap = _pop_int(
        sys.argv,
        "--best-min-frame-gap",
        8,
        minimum=1,
    )
    _deprecated_feature_sigma = _pop_float(
        sys.argv,
        "--paper-feature-sigma",
        0.145,
        minimum_exclusive=0.0,
    )
    if "--paper-feature-sigma" in " ".join(sys.argv):
        # This branch is normally unreachable because _pop_float removes the
        # option; kept only for clarity.
        pass
    if abs(_deprecated_feature_sigma - 0.145) > 1e-12:
        print(
            "[paper-v7] warning: --paper-feature-sigma is ignored; "
            "Target/Predicted are rendered as exact piecewise-constant patches.",
            flush=True,
        )
    _PAPER_PATCH_LAYOUT_SIGMA = _pop_float(
        sys.argv,
        "--paper-patch-layout-sigma",
        0.060,
        minimum_exclusive=0.0,
    )
    _PAPER_PATCH_BOUNDARY_STYLE = _pop_choice(
        sys.argv,
        "--paper-patch-boundary",
        "soft",
        {"soft", "hard", "none"},
    )
    _PAPER_PATCH_BOUNDARY_ALPHA = _pop_float(
        sys.argv,
        "--paper-patch-boundary-alpha",
        0.18,
        minimum_exclusive=-1e-12,
    )
    if not 0.0 <= _PAPER_PATCH_BOUNDARY_ALPHA <= 1.0:
        raise SystemExit("--paper-patch-boundary-alpha must be in [0, 1]")
    _PAPER_PATCH_BOUNDARY_SIGMA = _pop_float(
        sys.argv,
        "--paper-patch-boundary-sigma",
        1.50,
        minimum_exclusive=0.0,
    )
    _PAPER_RAW_GAMMA = _pop_float(
        sys.argv,
        "--paper-raw-gamma",
        0.45,
        minimum_exclusive=0.0,
    )
    _PAPER_FEATURE_GAMMA = _pop_float(
        sys.argv,
        "--paper-feature-gamma",
        0.70,
        minimum_exclusive=0.0,
    )
    _PAPER_FEATURE_SCALE = _pop_choice(
        sys.argv,
        "--paper-feature-scale",
        "frame",
        {"frame", "episode"},
    )
    _PAPER_RAW_SCALE = _pop_choice(
        sys.argv,
        "--paper-raw-scale",
        "frame",
        {"frame", "episode"},
    )
    if auto_contact:
        _install_auto_contact_frame(sys.argv)
    elif auto_five_finger:
        _install_auto_five_finger_frame(sys.argv)
    elif auto_five_finger_global:
        _install_auto_five_finger_global(sys.argv)
    elif auto_best_frame:
        _best_reconstruction_selection(
            sys.argv,
            global_search=False,
            candidates_per_episode=best_candidates_per_episode,
            min_frame_gap=best_min_frame_gap,
        )
    elif auto_best_global:
        _best_reconstruction_selection(
            sys.argv,
            global_search=True,
            candidates_per_episode=best_candidates_per_episode,
            min_frame_gap=best_min_frame_gap,
        )
    _base.main()