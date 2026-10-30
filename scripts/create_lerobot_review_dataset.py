#!/usr/bin/env python3
"""Create a small LeRobot v2.1 review dataset with one episode per source.

The selected episodes are copied in source order and renumbered from zero.
Parquet indices, tasks, episode metadata, statistics, and video paths are kept
consistent so the output remains a loadable LeRobot dataset.

Example:

    python scripts/create_lerobot_review_dataset.py \
      --sources data/0723_open_cabinet data/grasp_pipette_and_press_button_106ep \
      --episode-indices 1 1 \
      --output data/data_example \
      --overwrite

If ``--episode-indices`` is omitted, episode 0 is selected from every source.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        raise KeyError(f"Required parquet column is missing: {name}")
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(values, type=field.type))


def scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [len(values)],
    }


def update_index_stats(
    source_stats: dict[str, Any] | None,
    *,
    episode_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
) -> dict[str, Any] | None:
    if source_stats is None:
        return None
    output = copy.deepcopy(source_stats)
    output["episode_index"] = scalar_stats(episode_indices)
    output["index"] = scalar_stats(global_indices)
    output["task_index"] = scalar_stats(task_indices)
    return output


def format_dataset_path(
    root: Path,
    pattern: str,
    *,
    chunk_size: int,
    episode_index: int,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return root / pattern.format(**values)


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return format_dataset_path(
        root,
        str(info["data_path"]),
        chunk_size=int(info["chunks_size"]),
        episode_index=episode_index,
    )


def video_keys(info: dict[str, Any]) -> list[str]:
    return [
        name
        for name, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def copy_episode_videos(
    source: Path,
    source_info: dict[str, Any],
    output: Path,
    output_info: dict[str, Any],
    *,
    old_episode_index: int,
    new_episode_index: int,
) -> int:
    keys = video_keys(source_info)
    if not keys:
        return 0
    if "video_path" not in source_info or "video_path" not in output_info:
        raise KeyError("Dataset has video features but info.json has no video_path pattern.")

    copied = 0
    for key in keys:
        source_video = format_dataset_path(
            source,
            str(source_info["video_path"]),
            chunk_size=int(source_info["chunks_size"]),
            episode_index=old_episode_index,
            video_key=key,
        )
        if not source_video.exists():
            raise FileNotFoundError(f"Missing video for {key}: {source_video}")
        destination = format_dataset_path(
            output,
            str(output_info["video_path"]),
            chunk_size=int(output_info["chunks_size"]),
            episode_index=new_episode_index,
            video_key=key,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video, destination)
        copied += 1
    return copied


def validate_sources(
    sources: list[Path], episode_indices: list[int]
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for source, episode_index in zip(sources, episode_indices, strict=True):
        info_path = source / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        info = read_json(info_path)
        if not str(info.get("codebase_version", "")).startswith("v2.1"):
            raise ValueError(f"{source} is not LeRobot v2.1: {info.get('codebase_version')}")
        total_episodes = int(info.get("total_episodes", 0))
        if episode_index < 0 or episode_index >= total_episodes:
            raise IndexError(
                f"Episode {episode_index} is outside {source} range [0, {total_episodes})."
            )
        parquet = episode_path(source, info, episode_index)
        if not parquet.exists():
            raise FileNotFoundError(f"Missing selected episode parquet: {parquet}")
        infos.append(info)

    reference = infos[0]
    for source, info in zip(sources[1:], infos[1:], strict=True):
        for key in ("fps", "robot_type"):
            if info.get(key) != reference.get(key):
                raise ValueError(
                    f"Incompatible {key}: {sources[0]}={reference.get(key)!r}, "
                    f"{source}={info.get(key)!r}"
                )
        if info.get("features", {}) != reference.get("features", {}):
            raise ValueError(f"Incompatible feature schema between {sources[0]} and {source}")
        if video_keys(info) != video_keys(reference):
            raise ValueError(f"Incompatible video streams between {sources[0]} and {source}")
    return infos


def validate_output_path(output: Path, sources: list[Path]) -> None:
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if output in forbidden:
        raise ValueError(f"Refusing to use broad directory as output: {output}")
    for source in sources:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Output and source directories must not overlap: {output}, {source}")


def build_review_dataset(
    sources: list[Path],
    episode_indices: list[int],
    infos: list[dict[str, Any]],
    output: Path,
) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)

    output_info = copy.deepcopy(infos[0])
    output_chunk_size = int(output_info["chunks_size"])
    task_to_index: dict[str, int] = {}
    output_tasks: list[dict[str, Any]] = []
    output_episodes: list[dict[str, Any]] = []
    output_episode_stats: list[dict[str, Any]] = []
    selection_manifest: list[dict[str, Any]] = []
    next_global_index = 0
    total_videos = 0

    for new_episode_index, (source, old_episode_index, info) in enumerate(
        zip(sources, episode_indices, infos, strict=True)
    ):
        source_tasks = {
            int(row["task_index"]): str(row["task"])
            for row in read_jsonl(source / "meta" / "tasks.jsonl")
        }
        source_episodes = {
            int(row["episode_index"]): row
            for row in read_jsonl(source / "meta" / "episodes.jsonl")
        }
        source_stats = {
            int(row["episode_index"]): row.get("stats")
            for row in read_jsonl(source / "meta" / "episodes_stats.jsonl")
        }
        if old_episode_index not in source_episodes:
            raise KeyError(
                f"Episode {old_episode_index} is missing from {source}/meta/episodes.jsonl"
            )

        table = pq.read_table(episode_path(source, info, old_episode_index))
        length = table.num_rows
        if length <= 0:
            raise ValueError(f"Selected episode is empty: {source}, episode {old_episode_index}")

        old_task_indices = table["task_index"].to_numpy(zero_copy_only=False)
        new_task_indices = np.empty(length, dtype=np.int64)
        for task_index_value in np.unique(old_task_indices):
            old_task_index = int(task_index_value)
            if old_task_index not in source_tasks:
                raise KeyError(f"Task index {old_task_index} is missing from {source}/meta/tasks.jsonl")
            task = source_tasks[old_task_index]
            if task not in task_to_index:
                new_index = len(task_to_index)
                task_to_index[task] = new_index
                output_tasks.append({"task_index": new_index, "task": task})
            new_task_indices[old_task_indices == old_task_index] = task_to_index[task]

        episode_column = np.full(length, new_episode_index, dtype=np.int64)
        global_indices = np.arange(next_global_index, next_global_index + length, dtype=np.int64)
        table = replace_column(table, "episode_index", episode_column)
        table = replace_column(table, "index", global_indices)
        table = replace_column(table, "task_index", new_task_indices)

        destination = format_dataset_path(
            output,
            str(output_info["data_path"]),
            chunk_size=output_chunk_size,
            episode_index=new_episode_index,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)

        episode_meta = copy.deepcopy(source_episodes[old_episode_index])
        episode_meta["episode_index"] = new_episode_index
        episode_meta["length"] = length
        episode_meta["tasks"] = [
            source_tasks[int(index)] for index in dict.fromkeys(old_task_indices.tolist())
        ]
        output_episodes.append(episode_meta)

        updated_stats = update_index_stats(
            source_stats.get(old_episode_index),
            episode_indices=episode_column,
            global_indices=global_indices,
            task_indices=new_task_indices,
        )
        if updated_stats is not None:
            output_episode_stats.append({"episode_index": new_episode_index, "stats": updated_stats})

        total_videos += copy_episode_videos(
            source,
            info,
            output,
            output_info,
            old_episode_index=old_episode_index,
            new_episode_index=new_episode_index,
        )
        selection_manifest.append(
            {
                "source_order": new_episode_index,
                "source_name": source.name,
                "source_episode_index": old_episode_index,
                "output_episode_index": new_episode_index,
                "frames": length,
            }
        )
        print(
            f"{source.name}: episode {old_episode_index} -> {new_episode_index} "
            f"({length} frames)"
        )
        next_global_index += length

    total_episodes = len(sources)
    output_info["total_episodes"] = total_episodes
    output_info["total_frames"] = next_global_index
    output_info["total_tasks"] = len(output_tasks)
    output_info["total_videos"] = total_videos
    output_info["total_chunks"] = math.ceil(total_episodes / output_chunk_size)
    output_info["splits"] = {"train": f"0:{total_episodes}"}

    write_json(output / "meta" / "info.json", output_info)
    write_jsonl(output / "meta" / "tasks.jsonl", output_tasks)
    write_jsonl(output / "meta" / "episodes.jsonl", output_episodes)
    if output_episode_stats:
        if len(output_episode_stats) != total_episodes:
            raise ValueError(
                "Selected sources contain a mixture of present and missing episode statistics."
            )
        write_jsonl(output / "meta" / "episodes_stats.jsonl", output_episode_stats)
    write_json(output / "meta" / "review_selection.json", selection_manifest)
    return {
        "episodes": total_episodes,
        "frames": next_global_index,
        "tasks": len(output_tasks),
        "videos": total_videos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="*",
        help="One episode index per source. Defaults to episode 0 for every source.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [path.expanduser().resolve() for path in args.sources]
    if not sources:
        raise ValueError("At least one source dataset is required.")
    episode_indices = args.episode_indices
    if not episode_indices:
        episode_indices = [0] * len(sources)
    if len(episode_indices) != len(sources):
        raise ValueError(
            "--episode-indices must contain exactly one integer per --sources entry; "
            f"got {len(episode_indices)} indices for {len(sources)} sources."
        )

    output = args.output.expanduser().resolve()
    validate_output_path(output, sources)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    infos = validate_sources(sources, episode_indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        summary = build_review_dataset(sources, episode_indices, infos, temporary)
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"Created review dataset at {output}: "
        f"{summary['episodes']} episodes, {summary['frames']} frames, "
        f"{summary['tasks']} tasks, {summary['videos']} videos."
    )


if __name__ == "__main__":
    main()
