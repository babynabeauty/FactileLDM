"""Local LeRobot v2.1 observations for probes, independent of the LeRobot Python API.

Reads per-episode Parquet states and current RGB video frames. Ground-truth
actions are intentionally not read. The downstream adapter separates future
taxels from the causal policy input and uses them only for labels.
"""

# Keep metadata preparation usable without installing video/Parquet libraries.
# ruff: noqa: PLC0415

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np

from openpi.shared.future_patch_probe_utils import read_json


def require_video_decoder():
    """Check the software AV1 decoder before loading a large policy checkpoint."""
    try:
        import av

        av.Codec("libdav1d", "r")
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "The local probe reader requires PyAV with the libdav1d software AV1 decoder. "
            "Install the wheel in the probe environment, then retry: "
            f'uv pip install --python "{sys.executable}" --only-binary=:all: "av==17.0.0"'
        ) from exc
    return av


class LocalProbeDataset:
    def __init__(self, root, *, episodes, state_offsets, image_keys):
        self.root = Path(root)
        self.info = read_json(self.root / "meta/info.json")
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError("This probe reader expects the supplied LeRobot v2.1 per-episode layout.")
        self.fps = float(self.info["fps"])
        self.state_offsets = np.asarray(state_offsets, dtype=np.int64)
        self.image_keys = tuple(dict.fromkeys(k for k in image_keys if k is not None))
        self.episode_info = {int(row["episode"]): row for row in episodes}
        task_rows = [
            json.loads(line) for line in (self.root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()
        ]
        self.task_names = {int(row["task_index"]): str(row["task"]) for row in task_rows}
        self.episode_ids = np.concatenate([np.full(row["length"], row["episode"], dtype=np.int64) for row in episodes])
        self.frame_ids = np.concatenate([np.arange(row["length"], dtype=np.int64) for row in episodes])
        self._loaded_episode = None
        self._states = None
        self._timestamps = None
        self._task_indices = None
        self._videos = {}

    def __len__(self):
        return len(self.episode_ids)

    def _path(self, template, episode, **kwargs):
        return self.root / template.format(
            episode_index=episode,
            episode_chunk=episode // int(self.info.get("chunks_size", 1000)),
            **kwargs,
        )

    def _load_episode(self, episode):
        import pyarrow.parquet as pq

        self.close()
        path = self._path(self.info["data_path"], episode)
        table = pq.read_table(
            path, columns=["observation.state", "episode_index", "frame_index", "timestamp", "task_index"]
        )
        frame_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        order = np.argsort(frame_ids)
        expected = int(self.episode_info[episode]["length"])
        if not np.array_equal(frame_ids[order], np.arange(expected)):
            raise ValueError(f"Missing/duplicate/nonzero-based frames in {path}.")
        if not np.all(np.asarray(table["episode_index"].to_numpy()) == episode):
            raise ValueError(f"Mixed episode IDs in {path}.")
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)[order]
        if not np.allclose(np.diff(timestamps), 1 / self.fps, atol=1e-3, rtol=0):
            raise ValueError(f"Episode timestamps do not match declared FPS in {path}.")
        self._states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
        self._timestamps = timestamps
        self._task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)[order]
        missing_tasks = set(self._task_indices) - self.task_names.keys()
        if missing_tasks:
            raise ValueError(f"Task indices {sorted(missing_tasks)} are missing from meta/tasks.jsonl.")
        actual_tasks = {self.task_names[int(index)] for index in self._task_indices}
        if actual_tasks != {self.episode_info[episode]["task"]}:
            logging.warning(
                "Episode %s has prompt labels %s but episode group label %r; check metric grouping.",
                episode,
                sorted(actual_tasks),
                self.episode_info[episode]["task"],
            )
        self._loaded_episode = episode

    def _open_video(self, key):
        av = require_video_decoder()
        path = self._path(self.info["video_path"], self._loaded_episode, video_key=key)
        if not path.is_file():
            raise FileNotFoundError(f"Camera video not found: {path}")
        container = av.open(str(path))
        try:
            if not container.streams.video:
                raise ValueError(f"No video stream in camera file: {path}")
            stream = container.streams.video[0]
            video_fps = float(stream.average_rate) if stream.average_rate is not None else float("nan")
            # A zero count means unknown; decoding below still rejects premature EOF.
            if not np.isclose(video_fps, self.fps, atol=0.05, rtol=0) or (
                stream.frames and stream.frames < len(self._states)
            ):
                raise ValueError(f"Video FPS/frame count does not match episode metadata: {path}")
            stream.codec_context.thread_count = 2
            logging.debug("Camera video %s: PyAV decoder=%s", path, stream.codec_context.name)
            self._videos[key] = {
                "container": container,
                "frames": iter(container.decode(stream)),
                "path": path,
                "next_frame": 0,
                "frame": -1,
                "rgb": None,
            }
        except Exception:
            container.close()
            raise

    def _read_rgb(self, key, frame):
        if key in self._videos:
            video = self._videos[key]
            if video["frame"] == frame:
                return video["rgb"].copy()
            if frame < video["next_frame"]:
                # Reopen instead of approximate keyframe seeking: index and PTS
                # must still agree exactly with the Parquet state timeline.
                video["container"].close()
                del self._videos[key]
        if key not in self._videos:
            self._open_video(key)
        video = self._videos[key]
        while video["next_frame"] <= frame:
            current = video["next_frame"]
            try:
                decoded = next(video["frames"])
            except StopIteration as exc:
                raise RuntimeError(
                    f"Missing RGB frame {current} while reading {key} frame {frame}: {video['path']}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot decode RGB frame {current} for {key} with PyAV: {video['path']}: {exc}"
                ) from exc
            video_timestamp = decoded.time
            if (
                video_timestamp is None
                or not np.isfinite(video_timestamp)
                or not np.isclose(video_timestamp, self._timestamps[current], atol=1e-3, rtol=0)
            ):
                raise ValueError(
                    f"Video timestamp mismatch for {key} frame {current}: "
                    f"video={video_timestamp}s, state={self._timestamps[current]:.6f}s; {video['path']}. "
                    "Use the original training LeRobot reader or check video alignment."
                )
            video["next_frame"] = current + 1
        video.update(frame=frame, rgb=decoded.to_ndarray(format="rgb24"))
        return video["rgb"].copy()

    def __getitem__(self, index):
        episode, frame = int(self.episode_ids[index]), int(self.frame_ids[index])
        if self._loaded_episode != episode:
            self._load_episode(episode)
        requested = frame + self.state_offsets
        padding = (requested < 0) | (requested >= len(self._states))
        sample = {
            "observation.state": self._states[np.clip(requested, 0, len(self._states) - 1)].copy(),
            "observation.state_is_pad": padding,
            "episode_index": np.int64(episode),
            "frame_index": np.int64(frame),
            "timestamp": np.float64(self._timestamps[frame]),
            "prompt": self.task_names[int(self._task_indices[frame])],
        }
        for key in self.image_keys:
            sample[key] = self._read_rgb(key, frame)
        return sample

    def close(self):
        for video in self._videos.values():
            video["container"].close()
        self._videos.clear()
        self._loaded_episode = None

    def __del__(self):
        if hasattr(self, "_videos"):
            self.close()
