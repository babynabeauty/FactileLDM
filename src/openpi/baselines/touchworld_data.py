"""Episode-safe data adapter for a TouchWorld-inspired XHand baseline.

This module requires only NumPy; it does not claim a local dataset exists.
Images are uint8 RGB [T,H,W,3] (0..255) or floating RGB (0..1). Tactile is
recorded [T,5,120,3] in the sensor's native units, NOT automatically Newtons.
States/actions are 18 joint positions: six arm then twelve hand joints, in the
recorded controller units. No normalization, coordinate conversion, force
calibration, or train/test inference is performed here.

At anchor t, action j is recorded command t+j; the predictive goal is observation
t+H. sample(offset=o) exposes history only through t+o while retaining the same
chunk-start RGB and terminal goal. Future fields are supervision ONLY; never pass
the entire sample dictionary to an observation encoder. All padding is zero and
has an explicit validity mask. Frame IDs, not adjacent array rows, define time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

import numpy as np


JOINT_NAMES = tuple([f"arm_joint_{i}.pos" for i in range(6)] + [f"hand_joint_{i}.pos" for i in range(12)])
RAW_TACTILE_NAMES = tuple(
    f"hand_tactile_sensor_{finger}.raw_force_{taxel}.{axis}"
    for finger in range(5) for taxel in range(120) for axis in ("x", "y", "z")
)
DEFAULT_IMAGE_KEYS = ("observation.images.cam_front",)


class LazyImageSequence:
    """Bounded frame reader for large videos; construction performs no decode.

    read_frames(row_indices) must return exactly those RGB frames, in that order,
    with shape [N,H,W,3]. Row indices refer to this episode's sorted rows, so a
    callback may map them to timestamps before calling its chosen video decoder.
    This object is intentionally not implicitly convertible to a NumPy array.
    """

    def __init__(self, length, frame_shape, read_frames: Callable, *, dtype=np.uint8):
        if not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("Image sequence length must be a positive integer.")
        if len(frame_shape) != 3 or frame_shape[-1] != 3 or any(int(v) != v or v < 1 for v in frame_shape):
            raise ValueError("frame_shape must be positive [H,W,3].")
        self.shape = (int(length), *(int(v) for v in frame_shape))
        self.ndim = 4
        self.dtype = np.dtype(dtype)
        if self.dtype != np.uint8 and not np.issubdtype(self.dtype, np.floating):
            raise ValueError("Image dtype must be uint8 or floating point.")
        self.read_frames = read_frames

    def __getitem__(self, index):
        scalar = isinstance(index, (int, np.integer))
        if isinstance(index, slice):
            rows = np.arange(self.shape[0], dtype=np.int64)[index]
        else:
            rows = np.asarray([index] if scalar else index)
            if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
                raise IndexError("Image indices must be integers.")
            rows = rows.astype(np.int64, copy=False)
            rows = np.where(rows < 0, rows + self.shape[0], rows)
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise IndexError("Image frame index is out of bounds.")
        if not len(rows):
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        result = np.asarray(self.read_frames(rows))
        expected = (len(rows), *self.shape[1:])
        if result.shape != expected or result.dtype != self.dtype:
            raise ValueError(f"Video callback must return {expected} with dtype {self.dtype}, got {result.shape}/{result.dtype}.")
        if self.dtype != np.uint8 and (not np.isfinite(result).all() or result.min() < 0 or result.max() > 1):
            raise ValueError("Floating video frames must be finite RGB in [0,1].")
        return result[0] if scalar else result


def _integer_ids(values, name: str) -> tuple[int, ...]:
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain integer IDs, got {value!r}.")
        if value < 0:
            raise ValueError(f"{name} cannot contain negative IDs.")
        result.append(int(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs.")
    return tuple(result)


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    train_dataset_id: str = "default"
    test_dataset_id: str = "default"


def validate_episode_splits(train, validation=(), test=(), *, train_dataset_id="default", test_dataset_id=None) -> EpisodeSplit:
    """Validate explicit manifests; no split may share an episode."""
    groups = [_integer_ids(v, name) for v, name in zip((train, validation, test), ("train", "validation", "test"))]
    train_dataset_id = str(train_dataset_id)
    test_dataset_id = train_dataset_id if test_dataset_id is None else str(test_dataset_id)
    if not train_dataset_id or not test_dataset_id:
        raise ValueError("Dataset identities cannot be empty.")
    for i in range(3):
        for j in range(i + 1, 3):
            if j == 2 and train_dataset_id != test_dataset_id:
                continue
            overlap = set(groups[i]) & set(groups[j])
            if overlap:
                raise ValueError(f"Episode split overlap: {sorted(overlap)}.")
    if not groups[0]:
        raise ValueError("Training episode manifest cannot be empty.")
    return EpisodeSplit(*groups, train_dataset_id, test_dataset_id)


def split_training_episodes(train_ids, test_ids, *, val_fraction=0.1, seed=0,
                            train_dataset_id="default", test_dataset_id=None) -> EpisodeSplit:
    """Reserve validation only from the supplied training manifest.

    For a supplied 800/100 protocol, all 100 test IDs remain untouched. The
    function deliberately does not fabricate those IDs from episode counts.
    """
    original = validate_episode_splits(train_ids, (), test_ids, train_dataset_id=train_dataset_id, test_dataset_id=test_dataset_id)
    if not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between zero and one.")
    if len(original.train) < 2:
        raise ValueError("At least two training episodes are needed for validation.")
    ids = np.asarray(sorted(original.train), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    return validate_episode_splits(sorted(shuffled[n_val:].tolist()), sorted(shuffled[:n_val].tolist()), original.test,
                                  train_dataset_id=original.train_dataset_id, test_dataset_id=original.test_dataset_id)


@dataclass
class EpisodeData:
    episode_id: int
    task_id: int
    images: Mapping[str, np.ndarray | LazyImageSequence]
    tactile: np.ndarray
    state: np.ndarray
    actions: np.ndarray
    frame_ids: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    dataset_id: str = "default"

    def __post_init__(self):
        self.episode_id = _integer_ids([self.episode_id], "episode_id")[0]
        self.task_id = _integer_ids([self.task_id], "task_id")[0]
        self.dataset_id = str(self.dataset_id)
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty.")
        self.tactile = np.asarray(self.tactile, dtype=np.float32)
        self.state = np.asarray(self.state, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        length = len(self.state)
        if length < 1:
            raise ValueError("An episode cannot be empty.")
        for name, array, shape in (("tactile", self.tactile, (length, 5, 120, 3)),
                                   ("state", self.state, (length, 18)), ("actions", self.actions, (length, 18))):
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}.")
        if not self.images:
            raise ValueError("At least one RGB camera is required.")
        self.images = {str(k): v if isinstance(v, LazyImageSequence) else np.asarray(v) for k, v in self.images.items()}
        for name, array in self.images.items():
            if array.ndim != 4 or array.shape[0] != length or array.shape[-1] != 3:
                raise ValueError(f"Image {name} must be [T,H,W,3], got {array.shape}.")
            if array.dtype != np.uint8 and not isinstance(array, LazyImageSequence):
                if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                    raise ValueError(f"Image {name} must be uint8 RGB or floating RGB in [0,1].")
        if self.frame_ids is None:
            self.frame_ids = np.arange(length, dtype=np.int64)
        self.frame_ids = np.asarray(self.frame_ids)
        if self.frame_ids.shape != (length,) or not np.issubdtype(self.frame_ids.dtype, np.integer):
            raise ValueError("frame_ids must be integer [T].")
        _integer_ids(self.frame_ids, "frame_ids")
        if np.any(np.diff(self.frame_ids) <= 0):
            raise ValueError("frame_ids must be strictly increasing; sort columns together before construction.")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
            if self.timestamps.shape != (length,) or not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
                raise ValueError("timestamps must be finite, strictly increasing [T].")


class EpisodeWindowDataset:
    """Finite, indexable NumPy dataset, suitable for a PyTorch DataLoader."""

    def __init__(self, episodes: Sequence[EpisodeData], *, horizon=16, history=10, stride=1):
        for name, value in (("horizon", horizon), ("history", history), ("stride", stride)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        self.episodes = tuple(episodes)
        identities = [(e.dataset_id, e.episode_id) for e in self.episodes]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate (dataset_id, episode_id) identities.")
        self.horizon, self.history = int(horizon), int(history)
        self._rows = [{int(f): i for i, f in enumerate(e.frame_ids)} for e in self.episodes]
        self._anchors = [(i, row) for i, e in enumerate(self.episodes) for row in range(0, len(e.frame_ids), int(stride))]

    def __len__(self):
        return len(self._anchors)

    def __getitem__(self, index):
        return self.sample(index, offset=0)

    def valid_offsets(self, index: int) -> tuple[int, ...]:
        """Offsets with a fully recorded prefix, for residual-stage sampling."""
        episode_index, row = self._anchors[index]
        anchor = int(self.episodes[episode_index].frame_ids[row])
        result = []
        for offset in range(self.horizon):
            if anchor + offset not in self._rows[episode_index]:
                break
            result.append(offset)
        return tuple(result)

    @staticmethod
    def _gather(array, rows):
        out = np.zeros((len(rows), *array.shape[1:]), dtype=array.dtype)
        valid = np.asarray([row >= 0 for row in rows], dtype=bool)
        if valid.any():
            out[valid] = array[np.asarray(rows, dtype=np.int64)[valid]]
        return out, valid

    def sample(self, index: int, *, offset: int = 0) -> dict:
        if isinstance(offset, bool) or not isinstance(offset, (int, np.integer)) or not 0 <= offset < self.horizon:
            raise ValueError("offset must be an integer in [0,horizon).")
        episode_index, anchor_row = self._anchors[index]
        episode, lookup = self.episodes[episode_index], self._rows[episode_index]
        anchor = int(episode.frame_ids[anchor_row])
        current = anchor + int(offset)
        if offset not in self.valid_offsets(index):
            raise ValueError(f"Cannot construct offset {offset}: missing current or earlier observation within this chunk.")
        current_row = lookup.get(current, -1)
        goal_row = lookup.get(anchor + self.horizon, -1)
        history_rows = [lookup.get(f, -1) for f in range(current - self.history + 1, current + 1)]
        action_rows = [lookup.get(anchor + j, -1) for j in range(self.horizon)]
        tactile_history, history_valid = self._gather(episode.tactile, history_rows)
        actions, action_valid = self._gather(episode.actions, action_rows)
        action_valid &= np.arange(self.horizon) >= offset
        state, current_mask = self._gather(episode.state, [current_row])
        tactile, _ = self._gather(episode.tactile, [current_row])
        future_tactile, goal_mask = self._gather(episode.tactile, [goal_row])
        # Missing intermediate observations make that recurrent replay invalid.
        observed_prefix_valid = all(anchor + j in lookup for j in range(offset + 1))
        current_valid = bool(current_mask[0] and observed_prefix_valid)
        action_valid &= current_valid
        return {
            "dataset_id": episode.dataset_id, "episode_id": episode.episode_id, "task_id": episode.task_id,
            "anchor_frame_id": anchor, "frame_id": current, "offset": int(offset),
            "timestamp": float(episode.timestamps[current_row]) if episode.timestamps is not None and current_row >= 0 else float("nan"),
            "image": {k: v[anchor_row].copy() for k, v in episode.images.items()},
            "state": state[0], "chunk_state": episode.state[anchor_row].copy(),
            "tactile": tactile[0], "tactile_history": tactile_history, "history_valid": history_valid,
            "actions": actions, "action_valid": action_valid,
            "future_image": {k: self._gather(v, [goal_row])[0][0] for k, v in episode.images.items()},
            "future_tactile": future_tactile[0], "future_valid": bool(goal_mask[0] and current_valid),
            "current_valid": current_valid,
        }


def _column_array(value):
    array = np.asarray(value)
    if array.dtype == object and array.ndim == 1 and len(array) and isinstance(array[0], (list, tuple, np.ndarray)):
        array = np.stack(array.tolist())
    return array


def _first(columns, keys):
    for key in keys:
        if key in columns:
            return key
    raise KeyError(f"Missing required column; tried {keys}.")


def episode_from_lerobot_columns(
    episode_id: int, columns: Mapping, *, image_keys=DEFAULT_IMAGE_KEYS,
    image_loader: Callable | None = None, state_names=None, action_names=None, task_id=None, dataset_id="default",
    allow_legacy_raw_layout: bool = False,
) -> EpisodeData:
    """Parse one episode without depending on LeRobot, Arrow, or video libraries.

    image_loader(episode_id, camera_key, frame_ids, timestamps) must return RGB
    frames in precisely that requested order, or a LazyImageSequence to defer
    decoding until sample access. It is required for video references
    or encoded image objects; numeric RGB columns need no callback. Full raw
    state is decoded by feature names by default. Positional legacy parsing of
    either joints or taxels requires explicit allow_legacy_raw_layout=True and
    the caller's confirmation of the original SDK layout; it is never inferred
    merely from vector width. Complete taxel names override positional parsing.
    """
    raw = _column_array(columns[_first(columns, ("observation.state", "observation/state", "state"))])
    if raw.ndim != 2:
        raise ValueError("observation.state must be [T,D].")
    if not isinstance(allow_legacy_raw_layout, (bool, np.bool_)):
        raise ValueError("allow_legacy_raw_layout must be boolean.")
    length = len(raw)
    ids = np.asarray(columns.get("frame_index", np.arange(length)))
    if ids.shape != (length,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("frame_index must be integer [T].")
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    times = np.asarray(columns["timestamp"], dtype=np.float64)[order] if "timestamp" in columns else None
    if "episode_index" in columns and not np.all(np.asarray(columns["episode_index"]) == episode_id):
        raise ValueError("Columns contain rows from another episode.")
    task_values = np.asarray(columns.get("task_index", []))
    if task_id is None:
        if task_values.shape != (length,) or len(np.unique(task_values)) != 1:
            raise ValueError("Provide one integer task_id or a constant task_index column per episode.")
        task_id = task_values[0].item()
    elif task_values.size and not np.all(task_values == task_id):
        raise ValueError("task_id disagrees with recorded task_index.")
    if state_names is not None:
        names = list(state_names)
        if len(names) != raw.shape[1] or len(set(names)) != len(names) or any(name not in names for name in JOINT_NAMES):
            raise ValueError("State feature names must identify all 18 joints and match the state width.")
        state = raw[:, [names.index(name) for name in JOINT_NAMES]]
    elif raw.shape[1] == 18:
        state = raw
    elif raw.shape[1] >= 52 and allow_legacy_raw_layout:
        state = raw[:, [*range(6), *(28 + 2 * np.arange(12)).tolist()]]
    else:
        raise ValueError("Unknown state layout: supply joint feature names; legacy positional state parsing requires allow_legacy_raw_layout=True.")
    tactile_key = next((key for key in ("observation.tactile", "observation/tactile", "tactile") if key in columns), None)
    if tactile_key is not None:
        tactile = _column_array(columns[tactile_key])
        if tactile.shape == (length, 1800):
            tactile = tactile.reshape(length, 5, 120, 3)
    elif state_names is not None and all(name in names for name in RAW_TACTILE_NAMES):
        name_to_index = {name: i for i, name in enumerate(names)}
        tactile = raw[:, [name_to_index[name] for name in RAW_TACTILE_NAMES]].reshape(length, 5, 120, 3)
    elif raw.shape[1] >= 52 + 5 * 384 and allow_legacy_raw_layout:
        tactile = np.stack([raw[:, 52 + finger * 384 + 24:52 + finger * 384 + 24 + 360].reshape(length, 120, 3)
                            for finger in range(5)], axis=1)
    else:
        raise ValueError("Missing dense tactile: provide an explicit [T,5,120,3] column or complete raw taxel feature names; legacy positional tactile parsing requires allow_legacy_raw_layout=True.")
    actions = _column_array(columns[_first(columns, ("action", "actions"))])
    if action_names is not None:
        names = list(action_names)
        if actions.ndim != 2 or len(names) != actions.shape[1] or any(name not in names for name in JOINT_NAMES):
            raise ValueError("Action names must identify all 18 joint position commands.")
        actions = actions[:, [names.index(name) for name in JOINT_NAMES]]
    images = {}
    for key in image_keys:
        image = _column_array(columns[key]) if key in columns else None
        if image is not None and image.ndim == 4 and image.shape[-1] == 3 and image.dtype != object:
            images[key] = image[order]
        elif image_loader is not None:
            images[key] = image_loader(episode_id, key, ids.copy(), None if times is None else times.copy())
        else:
            raise ValueError(f"Camera {key} needs image_loader for video/encoded image decoding.")
    return EpisodeData(episode_id, task_id, images, tactile[order], state[order], actions[order], ids, times, dataset_id)


def load_lerobot_episodes(
    repo_dir, episode_ids, *, image_keys=DEFAULT_IMAGE_KEYS, image_loader=None, read_table=None,
    allow_legacy_raw_layout: bool = False,
) -> list[EpisodeData]:
    """Read selected local LeRobot Parquet episodes with explicit video hook.

    read_table(path)->mapping of columns is optional; default uses PyArrow on
    demand. This supports per-episode files and partition files containing more
    than one episode. No downloads or inferred split memberships are performed.
    """
    ids = _integer_ids(episode_ids, "episode_ids")
    root = Path(repo_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Local LeRobot repository does not exist: {root}")
    if not ids:
        return []
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names")
    action_names = features.get("action", {}).get("names")
    if read_table is None:
        def read_table(path):
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Install pyarrow or provide read_table(path)->columns.") from exc
            return pq.read_table(path).to_pydict()
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet data files found under {root / 'data'}.")
    fragments = {episode_id: [] for episode_id in ids}
    wanted = set(ids)
    for path in paths:
        match = re.fullmatch(r"episode_(\d+)\.parquet", path.name)
        if match and int(match[1]) not in wanted:
            continue
        columns = {k: _column_array(v) for k, v in read_table(path).items()}
        if "episode_index" not in columns:
            if not match:
                raise ValueError(f"Partition file {path} needs episode_index.")
            n = len(columns[_first(columns, ("action", "actions"))])
            columns["episode_index"] = np.full(n, int(match[1]), dtype=np.int64)
        membership = columns["episode_index"]
        for episode_id in wanted & set(np.unique(membership).tolist()):
            rows = membership == episode_id
            fragments[episode_id].append({key: value[rows] for key, value in columns.items()})
    result = []
    for episode_id in ids:
        parts = fragments[episode_id]
        if not parts:
            raise ValueError(f"Requested episode {episode_id} was not found.")
        keys = set(parts[0])
        if any(set(part) != keys for part in parts):
            raise ValueError(f"Episode {episode_id} has inconsistent columns across partitions.")
        merged = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
        result.append(episode_from_lerobot_columns(episode_id, merged, image_keys=image_keys,
                      image_loader=image_loader, state_names=state_names, action_names=action_names,
                      dataset_id=str(root.resolve()), allow_legacy_raw_layout=allow_legacy_raw_layout))
    return result
