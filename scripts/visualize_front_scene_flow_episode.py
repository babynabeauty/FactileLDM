"""Visualize single-view front-camera 3D pseudo-flow for one LeRobot episode.

This is a lightweight validation tool for the 3D-flow idea.  It keeps all
visible depth points, colors background gray, robot yellow, object cyan, and
writes line segments from frame t to t + future_step using nearest-neighbor
matching inside robot/object masks.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from scipy import ndimage
from scipy.spatial import cKDTree
from torchcodec.decoders import VideoDecoder


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


FRONT_DEPTH_INTRINSICS = CameraIntrinsics(
    fx=387.7893371582031,
    fy=387.7893371582031,
    cx=321.7603454589844,
    cy=241.91043090820312,
)

GRAY = np.array([130, 130, 130], dtype=np.uint8)
YELLOW = np.array([245, 191, 35], dtype=np.uint8)
CYAN = np.array([0, 210, 210], dtype=np.uint8)
RED = np.array([255, 40, 40], dtype=np.uint8)


def read_rgb(video_path: pathlib.Path, frame_index: int) -> np.ndarray:
    decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cpu")
    frame = decoder.get_frame_at(frame_index).data.numpy()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def read_depth(parquet_path: pathlib.Path, frame_index: int, key: str) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=[key]).to_pandas()
    depth = np.asarray(table[key].iloc[frame_index].tolist(), dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got {depth.shape}.")
    return depth


def rgb_to_hsv_opencv_like(rgb: np.ndarray) -> np.ndarray:
    # Small dependency-free HSV conversion with OpenCV-like ranges:
    # H in [0, 179], S/V in [0, 255].
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = arr.max(axis=-1)
    cmin = arr.min(axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta > 1e-6
    rmax = mask & (cmax == r)
    gmax = mask & (cmax == g)
    bmax = mask & (cmax == b)
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4
    hue = hue * 30.0

    sat = np.zeros_like(cmax)
    nonzero = cmax > 1e-6
    sat[nonzero] = delta[nonzero] / cmax[nonzero]
    return np.stack([hue, sat * 255.0, cmax * 255.0], axis=-1).astype(np.float32)


def clean_mask(
    mask: np.ndarray,
    close_iters: int = 2,
    open_iters: int = 1,
    *,
    fill_holes: bool = False,
    min_area: int = 0,
) -> np.ndarray:
    mask = ndimage.binary_closing(mask, iterations=close_iters)
    mask = ndimage.binary_opening(mask, iterations=open_iters)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    if min_area > 0:
        labels, count = ndimage.label(mask)
        if count:
            areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
            keep_labels = np.flatnonzero(areas >= min_area) + 1
            mask = np.isin(labels, keep_labels)
    return mask.astype(bool)


def object_mask_from_points(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    points: list[tuple[int, int]],
    *,
    extra_down_px: int = 95,
    half_width_px: int = 11,
    depth_tolerance_m: float = 0.32,
) -> np.ndarray:
    height, width = depth_mm.shape
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    x1 = max(0, int(xs.min()) - half_width_px)
    x2 = min(width, int(xs.max()) + half_width_px + 1)
    y1 = max(0, int(ys.min()) - 20)
    y2 = min(height, int(ys.max()) + extra_down_px + 1)

    prompt_depths = []
    for x, y in points:
        patch = depth_mm[max(0, y - 2) : min(height, y + 3), max(0, x - 2) : min(width, x + 3)]
        valid = patch[patch > 0]
        if valid.size:
            prompt_depths.append(float(np.median(valid)) * 0.001)
    prompt_z = float(np.median(prompt_depths)) if prompt_depths else 1.0

    yy, xx = np.mgrid[:height, :width]
    roi = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    z = depth_mm.astype(np.float32) * 0.001
    valid_depth = (z > 0.2) & (np.abs(z - prompt_z) < depth_tolerance_m)

    hsv = rgb_to_hsv_opencv_like(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_or_yellow_top = ((h < 20) | ((h > 20) & (h < 45))) & (s > 70) & (v > 80)
    dark_or_light_stem = ((v < 95) | ((s < 80) & (v > 95))) & (np.abs(xx - float(np.median(xs))) <= half_width_px)

    mask = roi & valid_depth & (red_or_yellow_top | dark_or_light_stem)
    # Keep the mask as a narrow connected component seeded by the prompt points.
    labels, count = ndimage.label(clean_mask(mask, close_iters=1, open_iters=0, fill_holes=False, min_area=8))
    if count == 0:
        return roi & valid_depth
    seed_labels = set()
    for x, y in points:
        if 0 <= x < width and 0 <= y < height and labels[y, x] != 0:
            seed_labels.add(int(labels[y, x]))
    if not seed_labels:
        # Fall back to the largest object-like component in the ROI.
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        seed_labels.add(int(np.argmax(sizes)) + 1)
    return np.isin(labels, list(seed_labels))


def robot_mask_heuristic(rgb: np.ndarray, depth_mm: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    height, width = depth_mm.shape
    yy, xx = np.mgrid[:height, :width]
    hsv = rgb_to_hsv_opencv_like(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    z = depth_mm.astype(np.float32) * 0.001
    valid_depth = (z > 0.45) & (z < 2.20)

    left_roi = (xx < 335) & (yy < 205)
    right_roi = (xx > 355) & (xx < 625) & (yy < 390)
    roi = left_roi | right_roi

    blue_link = (h > 80) & (h < 115) & (s > 35) & (v > 60)
    green_led = (h > 45) & (h < 95) & (s > 50) & (v > 90)
    metal_or_white = (s < 70) & (v > 135)
    plausible_metal_area = (
        (left_roi & (yy < 165))
        | (right_roi & ((yy < 280) | ((xx > 420) & (xx < 570) & (yy < 340))))
    )

    mask = roi & valid_depth & (blue_link | green_led | (metal_or_white & plausible_metal_area))
    # Avoid labeling the bright work table as robot.
    table_like = (yy > 300) & (s < 65) & (v > 100)
    mask &= ~table_like
    mask &= ~ndimage.binary_dilation(object_mask, iterations=3)
    return clean_mask(mask, close_iters=1, open_iters=1, fill_holes=False, min_area=35)


def masks_for_frame(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    object_points: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    object_mask = object_mask_from_points(rgb, depth_mm, object_points)
    robot_mask = robot_mask_heuristic(rgb, depth_mm, object_mask)
    return robot_mask, object_mask


def _union_sam3_masks(output: dict) -> tuple[np.ndarray, int, list[float]]:
    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or masks.numel() == 0:
        return np.zeros((480, 640), dtype=bool), 0, []
    masks_np = masks.detach().cpu().numpy().astype(bool)
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    union = np.any(masks_np, axis=0)
    scores_list = scores.detach().float().cpu().tolist() if scores is not None else []
    return union.astype(bool), int(masks_np.shape[0]), scores_list


class Sam3FrameMasker:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str,
        confidence_threshold: float,
        robot_prompts: tuple[str, ...],
        object_box_padding_x: int,
        object_box_padding_up: int,
        object_box_padding_down: int,
        object_negative_radius: int = 14,
        object_negative_y_margin: int = 8,
        object_negative_mode: str = "local",
    ):
        self.device = device
        self.robot_prompts = robot_prompts
        self.object_box_padding_x = object_box_padding_x
        self.object_box_padding_up = object_box_padding_up
        self.object_box_padding_down = object_box_padding_down
        self.object_negative_radius = object_negative_radius
        self.object_negative_y_margin = object_negative_y_margin
        self.object_negative_mode = object_negative_mode
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.last_info: dict[str, object] = {}

    def _amp_context(self):
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.inference_mode()

    def _object_box_from_points(self, points: list[tuple[int, int]], width: int, height: int) -> list[float]:
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        x0 = max(0.0, float(xs.min() - self.object_box_padding_x))
        x1 = min(float(width - 1), float(xs.max() + self.object_box_padding_x))
        y0 = max(0.0, float(ys.min() - self.object_box_padding_up))
        y1 = min(float(height - 1), float(ys.max() + self.object_box_padding_down))
        return [
            ((x0 + x1) * 0.5) / width,
            ((y0 + y1) * 0.5) / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
        ]

    def _object_roi_from_points(self, points: list[tuple[int, int]], width: int, height: int) -> np.ndarray:
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        x0 = max(0, int(np.floor(xs.min() - self.object_box_padding_x)))
        x1 = min(width, int(np.ceil(xs.max() + self.object_box_padding_x + 1)))
        y0 = max(0, int(np.floor(ys.min() - self.object_box_padding_up)))
        y1 = min(height, int(np.ceil(ys.max() + self.object_box_padding_down + 1)))
        roi = np.zeros((height, width), dtype=bool)
        roi[y0:y1, x0:x1] = True
        return roi

    def _remove_negative_point_disks(
        self,
        mask: np.ndarray,
        negative_points: list[tuple[int, int]] | None,
    ) -> np.ndarray:
        if not negative_points:
            return mask
        height, width = mask.shape
        yy, xx = np.mgrid[:height, :width]
        refined = mask.copy()
        for x, y in negative_points:
            if 0 <= x < width and 0 <= y < height:
                disk = (xx - x) ** 2 + (yy - y) ** 2 <= self.object_negative_radius**2
                refined[disk] = False
        return refined

    def _apply_object_negative_points(
        self,
        mask: np.ndarray,
        positive_points: list[tuple[int, int]],
        negative_points: list[tuple[int, int]] | None,
    ) -> np.ndarray:
        if not negative_points:
            return mask

        height, width = mask.shape
        refined = self._remove_negative_point_disks(mask, negative_points)

        if self.object_negative_mode == "below":
            # Stronger mode for the fixed black tube: remove the local vertical
            # strip under negative clicks. This works early, but can overcut if
            # the object later moves into that image region.
            neg_x = np.array([p[0] for p in negative_points], dtype=np.float32)
            neg_y = np.array([p[1] for p in negative_points], dtype=np.float32)
            x0 = max(0, int(np.floor(neg_x.min() - self.object_box_padding_x)))
            x1 = min(width, int(np.ceil(neg_x.max() + self.object_box_padding_x + 1)))
            y0 = max(0, int(np.floor(neg_y.min() - self.object_negative_y_margin)))
            refined[y0:, x0:x1] = False
        elif self.object_negative_mode != "local":
            raise ValueError(f"Unknown object negative mode: {self.object_negative_mode}")

        # Keep only components that are supported by positive clicks. This prevents
        # unrelated SAM3 instances in the same box from leaking into object points.
        labels, count = ndimage.label(refined)
        if count == 0:
            return refined

        keep_labels = set()
        for x, y in positive_points:
            if 0 <= x < width and 0 <= y < height and labels[y, x] != 0:
                keep_labels.add(int(labels[y, x]))

        if not keep_labels:
            # If the click lands on a small hole after thresholding, keep the
            # component whose pixels are closest to any positive point.
            points = np.array(positive_points, dtype=np.float32)
            best_label = None
            best_dist = float("inf")
            for label in range(1, count + 1):
                ys, xs = np.nonzero(labels == label)
                if len(xs) == 0:
                    continue
                coords = np.stack([xs, ys], axis=1).astype(np.float32)
                dist = np.min(((coords[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
                if dist < best_dist:
                    best_dist = float(dist)
                    best_label = label
            if best_label is not None:
                keep_labels.add(int(best_label))

        return np.isin(labels, list(keep_labels))

    def masks_for_frame(
        self,
        rgb: np.ndarray,
        object_points: list[tuple[int, int]],
        object_negative_points: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = Image.fromarray(rgb)
        width, height = image.size
        robot_mask = np.zeros((height, width), dtype=bool)
        object_mask = np.zeros((height, width), dtype=bool)
        info: dict[str, object] = {
            "robot": {},
            "object": {},
        }

        with self._amp_context():
            state = self.processor.set_image(image)
        for prompt in self.robot_prompts:
            with self._amp_context():
                output = self.processor.set_text_prompt(prompt, state=state)
            mask, count, scores = _union_sam3_masks(output)
            robot_mask |= mask
            info["robot"][prompt] = {"count": count, "scores": scores}
            self.processor.reset_all_prompts(state)

        object_box = self._object_box_from_points(object_points, width, height)
        object_roi = self._object_roi_from_points(object_points, width, height)
        robot_mask = self._remove_negative_point_disks(robot_mask, object_negative_points)
        with self._amp_context():
            output = self.processor.add_geometric_prompt(object_box, True, state=state)
        object_mask, count, scores = _union_sam3_masks(output)
        object_mask &= object_roi
        object_mask = self._apply_object_negative_points(object_mask, object_points, object_negative_points)
        info["object"] = {"box_cxcywh_norm": object_box, "count": count, "scores": scores}
        self.processor.reset_all_prompts(state)
        self.last_info = info
        return robot_mask, object_mask


def depth_to_points(
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
    stride: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_m = depth_mm.astype(np.float32) * 0.001
    yy, xx = np.mgrid[: depth_mm.shape[0] : stride, : depth_mm.shape[1] : stride]
    z = depth_m[yy, xx]
    valid = (z > 0.2) & (z < max_depth_m)
    x = (xx.astype(np.float32) - intr.cx) * z / intr.fx
    y = (yy.astype(np.float32) - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=-1)[valid]
    pixels = np.stack([xx, yy], axis=-1)[valid]
    return points.astype(np.float32), pixels.astype(np.int32), valid


def color_points_from_masks(
    pixels: np.ndarray,
    robot_mask: np.ndarray,
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    colors = np.repeat(GRAY[None, :], len(pixels), axis=0)
    labels = np.zeros((len(pixels),), dtype=np.int8)
    ys = pixels[:, 1]
    xs = pixels[:, 0]
    object_sel = object_mask[ys, xs]
    robot_sel = robot_mask[ys, xs] & ~object_sel
    colors[robot_sel] = YELLOW
    colors[object_sel] = CYAN
    labels[robot_sel] = 1
    labels[object_sel] = 2
    return colors, robot_sel, object_sel


def write_point_ply(path: pathlib.Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors, strict=True):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_line_ply(path: pathlib.Path, starts: np.ndarray, ends: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty((len(starts) * 2, 3), dtype=np.float32)
    vertices[0::2] = starts
    vertices[1::2] = ends
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(starts)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in vertices:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for i, c in enumerate(colors):
            f.write(f"{2*i} {2*i+1} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_scene_with_lines_ply(
    path: pathlib.Path,
    points: np.ndarray,
    point_colors: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    line_colors: np.ndarray,
) -> None:
    line_vertices = np.empty((len(starts) * 2, 3), dtype=np.float32)
    line_vertices[0::2] = starts
    line_vertices[1::2] = ends
    line_vertex_colors = np.empty((len(starts) * 2, 3), dtype=np.uint8)
    line_vertex_colors[0::2] = line_colors
    line_vertex_colors[1::2] = line_colors
    vertices = np.concatenate([points, line_vertices], axis=0)
    colors = np.concatenate([point_colors, line_vertex_colors], axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(starts)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(vertices, colors, strict=True):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        offset = len(points)
        for i, c in enumerate(line_colors):
            f.write(f"{offset + 2*i} {offset + 2*i + 1} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def compute_nn_flow(
    src_points: np.ndarray,
    src_robot: np.ndarray,
    src_object: np.ndarray,
    tgt_points: np.ndarray,
    tgt_robot: np.ndarray,
    tgt_object: np.ndarray,
    *,
    max_lines_per_class: int,
    max_match_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_starts = []
    all_ends = []
    all_colors = []
    rng = np.random.default_rng(0)
    for src_sel, tgt_sel, color in (
        (src_robot, tgt_robot, YELLOW),
        (src_object, tgt_object, CYAN),
    ):
        src = src_points[src_sel]
        tgt = tgt_points[tgt_sel]
        if len(src) == 0 or len(tgt) == 0:
            continue
        if len(src) > max_lines_per_class:
            keep = rng.choice(len(src), size=max_lines_per_class, replace=False)
            src = src[keep]
        tree = cKDTree(tgt)
        dist, nn = tree.query(src, k=1, workers=-1)
        valid = dist < max_match_distance_m
        if not np.any(valid):
            continue
        all_starts.append(src[valid])
        all_ends.append(tgt[nn[valid]])
        all_colors.append(np.repeat(color[None, :], int(valid.sum()), axis=0))
    if not all_starts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(all_starts), np.concatenate(all_ends), np.concatenate(all_colors)


def save_overlay(path: pathlib.Path, rgb: np.ndarray, robot_mask: np.ndarray, object_mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = np.zeros((*robot_mask.shape, 4), dtype=np.uint8)
    overlay[robot_mask] = [245, 191, 35, 120]
    overlay[object_mask] = [0, 210, 210, 150]
    blended = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(blended)
    for x, y in points:
        r = 5
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 30, 30, 255), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    blended.convert("RGB").save(path)


def parse_points(text: str) -> list[tuple[int, int]]:
    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        x, y = item.split(",")
        points.append((int(x), int(y)))
    if not points:
        raise ValueError("At least one object prompt point is required.")
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=pathlib.Path,
        default=pathlib.Path("/data/shared_workspace/zhangshiqi/dataset/tactile_xhand_ur7e/grasp_pipette_and_press_button"),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--future-step", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-depth", type=float, default=2.5)
    parser.add_argument("--max-lines-per-class", type=int, default=2500)
    parser.add_argument("--max-match-distance", type=float, default=0.20)
    parser.add_argument("--object-points", default="238,215;242,250;246,275")
    parser.add_argument("--object-negative-points", default="")
    parser.add_argument("--segmenter", choices=("heuristic", "sam3"), default="heuristic")
    parser.add_argument("--sam3-checkpoint", default="/data/shared_workspace/zhangshiqi/hf/SAM/sam3/sam3.pt")
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-robot-prompts", default="robot arm;robot hand")
    parser.add_argument("--sam3-object-box-padding-x", type=int, default=16)
    parser.add_argument("--sam3-object-box-padding-up", type=int, default=25)
    parser.add_argument("--sam3-object-box-padding-down", type=int, default=105)
    parser.add_argument("--sam3-object-negative-radius", type=int, default=14)
    parser.add_argument("--sam3-object-negative-y-margin", type=int, default=8)
    parser.add_argument("--sam3-object-negative-mode", choices=("local", "below"), default="local")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/front_scene_flow_episode0_sam3_plan"))
    args = parser.parse_args()

    object_points = parse_points(args.object_points)
    object_negative_points = parse_points(args.object_negative_points) if args.object_negative_points else []
    ep = f"episode_{args.episode_index:06d}"
    current = args.frame_index
    future = args.frame_index + args.future_step
    video_path = args.repo_id / "videos" / "chunk-000" / "observation.images.cam_front" / f"{ep}.mp4"
    parquet_path = args.repo_id / "data" / "chunk-000" / f"{ep}.parquet"

    rgb0 = read_rgb(video_path, current)
    rgb1 = read_rgb(video_path, future)
    depth0 = read_depth(parquet_path, current, "observation.depths.cam_front")
    depth1 = read_depth(parquet_path, future, "observation.depths.cam_front")

    sam3_info0: dict[str, object] | None = None
    sam3_info1: dict[str, object] | None = None
    if args.segmenter == "sam3":
        robot_prompts = tuple(p.strip() for p in args.sam3_robot_prompts.split(";") if p.strip())
        masker = Sam3FrameMasker(
            args.sam3_checkpoint,
            device=args.sam3_device,
            confidence_threshold=args.sam3_confidence_threshold,
            robot_prompts=robot_prompts,
            object_box_padding_x=args.sam3_object_box_padding_x,
            object_box_padding_up=args.sam3_object_box_padding_up,
            object_box_padding_down=args.sam3_object_box_padding_down,
            object_negative_radius=args.sam3_object_negative_radius,
            object_negative_y_margin=args.sam3_object_negative_y_margin,
            object_negative_mode=args.sam3_object_negative_mode,
        )
        robot0, object0 = masker.masks_for_frame(rgb0, object_points, object_negative_points)
        sam3_info0 = masker.last_info
        robot1, object1 = masker.masks_for_frame(rgb1, object_points, object_negative_points)
        sam3_info1 = masker.last_info
    else:
        robot0, object0 = masks_for_frame(rgb0, depth0, object_points)
        robot1, object1 = masks_for_frame(rgb1, depth1, object_points)

    points0, pixels0, _ = depth_to_points(depth0, FRONT_DEPTH_INTRINSICS, args.stride, args.max_depth)
    points1, pixels1, _ = depth_to_points(depth1, FRONT_DEPTH_INTRINSICS, args.stride, args.max_depth)
    colors0, robot_sel0, object_sel0 = color_points_from_masks(pixels0, robot0, object0)
    colors1, robot_sel1, object_sel1 = color_points_from_masks(pixels1, robot1, object1)

    flow_starts, flow_ends, flow_colors = compute_nn_flow(
        points0,
        robot_sel0,
        object_sel0,
        points1,
        robot_sel1,
        object_sel1,
        max_lines_per_class=args.max_lines_per_class,
        max_match_distance_m=args.max_match_distance,
    )

    out = args.output_dir / f"{ep}_frame_{current:06d}_to_{future:06d}"
    out.mkdir(parents=True, exist_ok=True)
    save_overlay(out / f"front_overlay_frame_{current:06d}.png", rgb0, robot0, object0, object_points)
    save_overlay(out / f"front_overlay_frame_{future:06d}.png", rgb1, robot1, object1, object_points)
    write_point_ply(out / "current_all_points_robot_object_colored.ply", points0, colors0)
    write_point_ply(out / "future_all_points_robot_object_colored.ply", points1, colors1)
    write_line_ply(out / "nn_scene_flow_lines_robot_object.ply", flow_starts, flow_ends, flow_colors)
    write_scene_with_lines_ply(
        out / "current_points_with_nn_scene_flow_lines.ply",
        points0,
        colors0,
        flow_starts,
        flow_ends,
        flow_colors,
    )

    summary = (
        f"episode={args.episode_index}\n"
        f"frame={current} future_frame={future}\n"
        f"points_current={len(points0)} points_future={len(points1)}\n"
        f"robot_points_current={int(robot_sel0.sum())} object_points_current={int(object_sel0.sum())}\n"
        f"robot_points_future={int(robot_sel1.sum())} object_points_future={int(object_sel1.sum())}\n"
        f"flow_lines={len(flow_starts)}\n"
        f"segmenter={args.segmenter}\n"
        "colors: gray=background, yellow=robot(hand+arm), cyan=object\n"
    )
    if sam3_info0 is not None and sam3_info1 is not None:
        summary += f"sam3_current={sam3_info0}\n"
        summary += f"sam3_future={sam3_info1}\n"
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
