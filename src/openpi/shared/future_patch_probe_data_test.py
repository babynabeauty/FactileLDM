"""Actual v2.1 Parquet/MP4 fixtures; no LeRobot package or policy checkpoint."""

import json

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.shared.future_patch_probe_data import LocalProbeDataset


def write_fixture(root, *, video_fps=15, video_frames=6, timestamp_step=1 / 15, duplicate_frame=False):
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 15,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(
        "\n".join(json.dumps({"task_index": i, "task": f"task-{episode}"}) for i, episode in enumerate((0, 1001)))
    )
    episodes = []
    for episode in (0, 1001):
        episodes.append({"episode": episode, "length": 6, "task": f"task-{episode}"})
        chunk = episode // 1000
        data_path = root / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame_ids = list(range(6))
        if duplicate_frame:
            frame_ids[-1] = 4
        table = pa.table(
            {
                "episode_index": [episode] * 6,
                "frame_index": frame_ids,
                "timestamp": np.arange(6) * timestamp_step,
                "task_index": [int(episode == 1001)] * 6,
                "observation.state": [[episode, frame, frame + 0.5] for frame in range(6)],
                "action": [[999, 999] for _ in range(6)],
            }
        )
        pq.write_table(table, data_path)
        video_path = root / f"videos/chunk-{chunk:03d}/camera/episode_{episode:06d}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (32, 32))
        assert writer.isOpened()
        for frame in range(video_frames):
            # OpenCV writer expects BGR; the reader must return RGB.
            image = np.full((32, 32, 3), [180, 40, 20 + 20 * frame], dtype=np.uint8)
            writer.write(image)
        writer.release()
    return episodes


def reader(root, episodes):
    return LocalProbeDataset(root, episodes=episodes, state_offsets=(-1, 0, 1, 2), image_keys=("camera",))


def test_v21_reader_aligns_states_rgb_and_padding_without_gt_actions(tmp_path):
    dataset = reader(tmp_path, write_fixture(tmp_path))
    assert len(dataset) == 12
    try:
        sample = dataset[3]
        np.testing.assert_array_equal(sample["observation.state"][:, 1], [2, 3, 4, 5])
        np.testing.assert_allclose(sample["camera"].mean(axis=(0, 1)), [80, 40, 180], atol=8)
        assert sample["prompt"] == "task-0"
        assert "action" not in sample
        assert "actions" not in sample
        # Backward access must not silently reuse the previous frame.
        np.testing.assert_allclose(dataset[1]["camera"].mean(axis=(0, 1)), [40, 40, 180], atol=8)
        tail = dataset[5]
        np.testing.assert_array_equal(tail["observation.state"][:, 1], [4, 5, 5, 5])
        np.testing.assert_array_equal(tail["observation.state_is_pad"], [False, False, True, True])
        start = dataset[6]
        assert start["episode_index"] == 1001
        assert start["frame_index"] == 0
        np.testing.assert_array_equal(start["observation.state"][:, 1], [0, 0, 1, 2])
        np.testing.assert_array_equal(start["observation.state_is_pad"], [True, False, False, False])
    finally:
        dataset.close()
    assert not dataset._videos  # noqa: SLF001


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"video_fps": 30}, "Video FPS"),
        ({"video_frames": 5}, "Video FPS/frame count"),
        ({"timestamp_step": 0.1}, "timestamps"),
        ({"duplicate_frame": True}, "Missing/duplicate"),
    ],
)
def test_reader_rejects_temporal_misalignment(tmp_path, kwargs, message):
    dataset = reader(tmp_path, write_fixture(tmp_path, **kwargs))
    try:
        with pytest.raises(ValueError, match=message):
            dataset[0]
    finally:
        dataset.close()


def test_fallback_uses_parquet_task_index_for_prompt_not_episode_label(tmp_path, caplog):
    episodes = write_fixture(tmp_path)
    episodes[0]["task"] = "different episode label"
    dataset = reader(tmp_path, episodes)
    try:
        assert dataset[0]["prompt"] == "task-0"
        assert "different episode label" in caplog.text
    finally:
        dataset.close()


def test_fallback_rejects_frame_timestamp_mismatch(tmp_path, monkeypatch):
    dataset = reader(tmp_path, write_fixture(tmp_path))
    original = cv2.VideoCapture

    class IncorrectTimestamp:
        def __init__(self, path):
            self.inner = original(path)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def get(self, prop):
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 33.0
            return self.inner.get(prop)

    monkeypatch.setattr(cv2, "VideoCapture", IncorrectTimestamp)
    try:
        with pytest.raises(ValueError, match="Video timestamp mismatch"):
            dataset[0]
    finally:
        dataset.close()
