"""Sweep one front-camera episode with SAM3-tracked object masks.

Robot masks are still obtained from per-frame SAM3 text prompts.  The object
mask is initialized once on the first frame with positive/negative clicks and
then propagated through the video with SAM3's video tracker.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
import torch
from PIL import Image

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
from visualize_front_scene_flow_episode_sweep import (
    read_depth_episode,
    read_rgb_episode,
    write_mp4_from_pngs,
)


def _track_object_masks(
    video_frame_dir: pathlib.Path,
    *,
    checkpoint_path: str,
    device: str,
    width: int,
    height: int,
    num_frames: int,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
) -> dict[int, np.ndarray]:
    from sam3.model_builder import build_sam3_video_model

    model = build_sam3_video_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=device,
    ).eval()
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    inference_state = predictor.init_state(
            video_path=str(video_frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)
    rel_points = [[x / width, y / height] for x, y in points]
    points_tensor = torch.tensor(rel_points, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int32)

    with torch.inference_mode():
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_tensor,
            labels=labels_tensor,
            clear_old_points=True,
        )

        masks: dict[int, np.ndarray] = {}
        for frame_idx, obj_ids, _low_res_masks, video_res_masks, _obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=num_frames,
            reverse=False,
            tqdm_disable=False,
            propagate_preflight=True,
        ):
            if not obj_ids:
                continue
            mask = (video_res_masks[0] > 0.0).detach().cpu().numpy().squeeze()
            masks[int(frame_idx)] = mask.astype(bool)
    return masks


def _write_tracker_jpegs(frames: list[np.ndarray], frame_dir: pathlib.Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    expected_last = frame_dir / f"{len(frames) - 1:06d}.jpg"
    if expected_last.exists():
        return
    for idx, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"{idx:06d}.jpg", quality=95)


def _remove_negative_point_disks(
    mask: np.ndarray,
    points: list[tuple[int, int]],
    *,
    radius: int,
) -> np.ndarray:
    if not points:
        return mask
    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width]
    refined = mask.copy()
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            refined[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = False
    return refined


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
    parser.add_argument("--object-negative-points", default="249,326;259,361;248,301")
    parser.add_argument("--sam3-checkpoint", default="/data/shared_workspace/zhangshiqi/hf/SAM/sam3/sam3.pt")
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--sam3-robot-prompts", default="robot arm;robot hand")
    parser.add_argument("--sam3-object-box-padding-x", type=int, default=16)
    parser.add_argument("--sam3-object-box-padding-up", type=int, default=25)
    parser.add_argument("--sam3-object-box-padding-down", type=int, default=105)
    parser.add_argument("--sam3-object-negative-radius", type=int, default=18)
    parser.add_argument("--overlay-step", type=int, default=1)
    parser.add_argument("--pair-step", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--save-flow-npz", action="store_true")
    parser.add_argument("--skip-ply", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/front_scene_flow_episode0_sam3_tracked_object"),
    )
    args = parser.parse_args()

    object_points = parse_points(args.object_points)
    object_negative_points = parse_points(args.object_negative_points) if args.object_negative_points else []
    ep = f"episode_{args.episode_index:06d}"
    video_path = args.repo_id / "videos" / "chunk-000" / "observation.images.cam_front" / f"{ep}.mp4"
    parquet_path = args.repo_id / "data" / "chunk-000" / f"{ep}.parquet"

    depths = read_depth_episode(parquet_path, "observation.depths.cam_front")
    num_frames = len(depths) if args.max_frames is None else min(len(depths), args.max_frames)
    rgbs = read_rgb_episode(video_path, num_frames)
    height, width = rgbs[0].shape[:2]

    out_root = args.output_dir / ep
    tracker_frame_dir = out_root / "tracker_jpegs"
    _write_tracker_jpegs(rgbs, tracker_frame_dir)

    object_masks = _track_object_masks(
        tracker_frame_dir,
        checkpoint_path=args.sam3_checkpoint,
        device=args.sam3_device,
        width=width,
        height=height,
        num_frames=num_frames,
        positive_points=object_points,
        negative_points=object_negative_points,
    )

    robot_prompts = tuple(p.strip() for p in args.sam3_robot_prompts.split(";") if p.strip())
    robot_masker = Sam3FrameMasker(
        args.sam3_checkpoint,
        device=args.sam3_device,
        confidence_threshold=args.sam3_confidence_threshold,
        robot_prompts=robot_prompts,
        object_box_padding_x=args.sam3_object_box_padding_x,
        object_box_padding_up=args.sam3_object_box_padding_up,
        object_box_padding_down=args.sam3_object_box_padding_down,
        object_negative_radius=args.sam3_object_negative_radius,
        object_negative_mode="local",
    )

    overlay_dir = out_root / "overlays"
    pairs_dir = out_root / "pairs"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    overlay_paths: list[pathlib.Path] = []
    frame_indices = set(range(0, num_frames, args.overlay_step))
    for source in range(0, max(0, num_frames - args.future_step), args.pair_step):
        frame_indices.add(source)
        frame_indices.add(source + args.future_step)

    for local_i, frame_idx in enumerate(sorted(frame_indices)):
        robot_mask, _unused_object = robot_masker.masks_for_frame(rgbs[frame_idx], object_points, object_negative_points)
        object_mask = object_masks.get(frame_idx, np.zeros((height, width), dtype=bool))
        object_mask = _remove_negative_point_disks(
            object_mask,
            object_negative_points,
            radius=args.sam3_object_negative_radius,
        )
        masks[frame_idx] = (robot_mask, object_mask)
        if frame_idx % args.overlay_step == 0:
            overlay_path = overlay_dir / f"front_overlay_frame_{frame_idx:06d}.png"
            save_overlay(overlay_path, rgbs[frame_idx], robot_mask, object_mask, object_points if frame_idx == 0 else [])
            overlay_paths.append(overlay_path)
        print(
            f"[mask {local_i + 1}/{len(frame_indices)}] frame={frame_idx} "
            f"robot_px={int(robot_mask.sum())} object_px={int(object_mask.sum())}"
        )

    if args.overlay_step == 1:
        write_mp4_from_pngs(overlay_paths, out_root / f"{ep}_front_sam3_tracked_object_overlay.mp4", args.fps)

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
        if args.save_flow_npz:
            class_id = np.zeros((len(flow_colors),), dtype=np.int8)
            class_id[np.all(flow_colors == np.array([245, 191, 35], dtype=np.uint8), axis=1)] = 1
            class_id[np.all(flow_colors == np.array([0, 210, 210], dtype=np.uint8), axis=1)] = 2
            np.savez_compressed(
                pair_out / "nn_scene_flow_robot_object.npz",
                source_frame=np.array(source, dtype=np.int32),
                target_frame=np.array(target, dtype=np.int32),
                xyz=flow_starts.astype(np.float32),
                target_xyz=flow_ends.astype(np.float32),
                flow_xyz=(flow_ends - flow_starts).astype(np.float32),
                class_id=class_id,
                color=flow_colors.astype(np.uint8),
            )
        if not args.skip_ply:
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
        f"object_mask_source=sam3_video_tracker\n"
        f"save_flow_npz={args.save_flow_npz}\n"
        f"skip_ply={args.skip_ply}\n"
        f"num_overlay_frames={len(overlay_paths)}\n"
        f"num_pairs={len(rows)}\n"
        f"output={out_root.resolve()}\n"
    )
    (out_root / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
