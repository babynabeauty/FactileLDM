"""Sweep a full front-camera episode with SAM3 masks and 3D pseudo-flow previews."""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torchcodec.decoders import VideoDecoder

from visualize_front_scene_flow_episode import (
    FRONT_DEPTH_INTRINSICS,
    Sam3FrameMasker,
    color_points_from_masks,
    compute_nn_flow,
    depth_to_points,
    parse_points,
    save_overlay,
    write_line_ply,
    write_point_ply,
    write_scene_with_lines_ply,
)


def read_depth_episode(parquet_path: pathlib.Path, key: str) -> list[np.ndarray]:
    table = pq.read_table(parquet_path, columns=[key]).to_pandas()
    return [np.asarray(value.tolist(), dtype=np.uint16) for value in table[key]]


def read_rgb_episode(video_path: pathlib.Path, num_frames: int) -> list[np.ndarray]:
    decoder = VideoDecoder(video_path, dimension_order="NHWC", device="cpu")
    frames = []
    for idx in range(num_frames):
        frame = decoder.get_frame_at(idx).data.numpy()
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def write_mp4_from_pngs(png_paths: list[pathlib.Path], output_path: pathlib.Path, fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Writing mp4 requires opencv-python/cv2.") from exc

    if not png_paths:
        return
    first = cv2.imread(str(png_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read {png_paths[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        for path in png_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Failed to read {path}")
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=pathlib.Path,
        default=pathlib.Path("/data/shared_workspace/zhangshiqi/dataset/tactile_xhand_ur7e/grasp_pipette_and_press_button"),
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--future-step", type=int, default=32)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-depth", type=float, default=2.5)
    parser.add_argument("--max-lines-per-class", type=int, default=2500)
    parser.add_argument("--max-match-distance", type=float, default=0.20)
    parser.add_argument("--object-points", default="238,215;242,250;246,275")
    parser.add_argument("--object-negative-points", default="")
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
    parser.add_argument(
        "--overlay-step",
        type=int,
        default=1,
        help="Run SAM3 and save overlay every N frames. Use 1 for the full episode.",
    )
    parser.add_argument(
        "--pair-step",
        type=int,
        default=16,
        help="Export one 3D flow PLY every N source frames. Use 1 to export every t->t+future_step pair.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/front_scene_flow_episode0_sam3_sweep"))
    args = parser.parse_args()

    object_points = parse_points(args.object_points)
    object_negative_points = parse_points(args.object_negative_points) if args.object_negative_points else []
    ep = f"episode_{args.episode_index:06d}"
    video_path = args.repo_id / "videos" / "chunk-000" / "observation.images.cam_front" / f"{ep}.mp4"
    parquet_path = args.repo_id / "data" / "chunk-000" / f"{ep}.parquet"

    depths = read_depth_episode(parquet_path, "observation.depths.cam_front")
    num_frames = len(depths) if args.max_frames is None else min(len(depths), args.max_frames)
    rgbs = read_rgb_episode(video_path, num_frames)

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

    out_root = args.output_dir / ep
    overlay_dir = out_root / "overlays"
    pairs_dir = out_root / "pairs"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    infos: dict[int, dict[str, object]] = {}
    overlay_paths: list[pathlib.Path] = []
    frame_indices = list(range(0, num_frames, args.overlay_step))
    needed_for_pairs = set()
    for source in range(0, max(0, num_frames - args.future_step), args.pair_step):
        needed_for_pairs.add(source)
        needed_for_pairs.add(source + args.future_step)
    frame_indices = sorted(set(frame_indices) | needed_for_pairs)

    for local_i, frame_idx in enumerate(frame_indices):
        robot_mask, object_mask = masker.masks_for_frame(rgbs[frame_idx], object_points, object_negative_points)
        masks[frame_idx] = (robot_mask, object_mask)
        infos[frame_idx] = masker.last_info
        if frame_idx % args.overlay_step == 0:
            overlay_path = overlay_dir / f"front_overlay_frame_{frame_idx:06d}.png"
            save_overlay(overlay_path, rgbs[frame_idx], robot_mask, object_mask, object_points)
            overlay_paths.append(overlay_path)
        print(f"[mask {local_i + 1}/{len(frame_indices)}] frame={frame_idx}")

    if args.overlay_step == 1:
        write_mp4_from_pngs(overlay_paths, out_root / f"{ep}_front_sam3_overlay.mp4", args.fps)

    rows = []
    for source in range(0, max(0, num_frames - args.future_step), args.pair_step):
        target = source + args.future_step
        robot0, object0 = masks[source]
        robot1, object1 = masks[target]
        points0, pixels0, _ = depth_to_points(depths[source], FRONT_DEPTH_INTRINSICS, args.stride, args.max_depth)
        points1, pixels1, _ = depth_to_points(depths[target], FRONT_DEPTH_INTRINSICS, args.stride, args.max_depth)
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

        pair_out = pairs_dir / f"frame_{source:06d}_to_{target:06d}"
        pair_out.mkdir(parents=True, exist_ok=True)
        write_point_ply(pair_out / "current_all_points_robot_object_colored.ply", points0, colors0)
        write_point_ply(pair_out / "future_all_points_robot_object_colored.ply", points1, colors1)
        write_line_ply(pair_out / "nn_scene_flow_lines_robot_object.ply", flow_starts, flow_ends, flow_colors)
        write_scene_with_lines_ply(
            pair_out / "current_points_with_nn_scene_flow_lines.ply",
            points0,
            colors0,
            flow_starts,
            flow_ends,
            flow_colors,
        )

        rows.append(
            {
                "source_frame": source,
                "target_frame": target,
                "points_current": len(points0),
                "points_future": len(points1),
                "robot_points_current": int(robot_sel0.sum()),
                "object_points_current": int(object_sel0.sum()),
                "robot_points_future": int(robot_sel1.sum()),
                "object_points_future": int(object_sel1.sum()),
                "flow_lines": len(flow_starts),
            }
        )
        print(
            f"[pair] {source}->{target} robot={int(robot_sel0.sum())}/{int(robot_sel1.sum())} "
            f"object={int(object_sel0.sum())}/{int(object_sel1.sum())} flow={len(flow_starts)}"
        )

    csv_path = out_root / "pair_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source_frame"])
        writer.writeheader()
        writer.writerows(rows)

    summary = (
        f"episode={args.episode_index}\n"
        f"num_frames={num_frames}\n"
        f"future_step={args.future_step}\n"
        f"overlay_step={args.overlay_step}\n"
        f"pair_step={args.pair_step}\n"
        f"object_negative_mode={args.sam3_object_negative_mode}\n"
        f"num_overlay_frames={len(overlay_paths)}\n"
        f"num_pairs={len(rows)}\n"
        f"output={out_root.resolve()}\n"
    )
    (out_root / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
