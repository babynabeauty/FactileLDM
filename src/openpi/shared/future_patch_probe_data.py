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

import numpy as np

from openpi.shared.future_patch_probe_utils import read_json


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

    def _read_rgb(self, key, frame):
        import cv2

        if key not in self._videos:
            path = self._path(self.info["video_path"], self._loaded_episode, video_key=key)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise FileNotFoundError(f"Cannot decode camera video: {path}")
            video_fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isclose(video_fps, self.fps, atol=0.05, rtol=0) or frame_count < len(self._states):
                capture.release()
                raise ValueError(f"Video FPS/frame count does not match episode metadata: {path}")
            self._videos[key] = {"capture": capture, "next_frame": 0, "frame": -1, "rgb": None}
        video = self._videos[key]
        if video["frame"] == frame:
            return video["rgb"].copy()
        capture = video["capture"]
        if frame < video["next_frame"]:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
                raise RuntimeError(f"Cannot seek to RGB frame {frame} for {key}.")
            video["next_frame"] = frame
        while video["next_frame"] < frame:
            if not capture.grab():
                raise RuntimeError(f"Missing RGB frame before {frame} for {key}.")
            video["next_frame"] += 1
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode RGB frame {frame} for {key}.")
        video_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not np.isfinite(video_timestamp) or not np.isclose(
            video_timestamp, self._timestamps[frame], atol=1e-3, rtol=0
        ):
            raise ValueError(
                f"Video timestamp mismatch for {key} frame {frame}: "
                f"video={video_timestamp:.6f}s, state={self._timestamps[frame]:.6f}s. "
                "Use the original training LeRobot reader or check video alignment."
            )
        video.update(next_frame=frame + 1, frame=frame, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
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
            video["capture"].release()
        self._videos.clear()
        self._loaded_episode = None

    def __del__(self):
        if hasattr(self, "_videos"):
            self.close()
