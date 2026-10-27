"""Synthetic contract tests; no real robot data or video decoder is assumed."""

import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openpi.baselines.touchworld_data import EpisodeData
from openpi.baselines.touchworld_data import EpisodeWindowDataset
from openpi.baselines.touchworld_data import JOINT_NAMES
from openpi.baselines.touchworld_data import LazyImageSequence
from openpi.baselines.touchworld_data import RAW_TACTILE_NAMES
from openpi.baselines.touchworld_data import episode_from_lerobot_columns
from openpi.baselines.touchworld_data import load_lerobot_episodes
from openpi.baselines.touchworld_data import split_training_episodes
from openpi.baselines.touchworld_data import validate_episode_splits


def episode(episode_id=0, frames=range(8), base=0, dataset_id="default"):
    frames = np.asarray(list(frames), dtype=np.int64)
    values = frames.astype(np.float32) + base
    n = len(frames)
    return EpisodeData(
        episode_id=episode_id, task_id=2, dataset_id=dataset_id,
        images={"front": np.broadcast_to(values.astype(np.uint8)[:, None, None, None], (n, 4, 6, 3)).copy()},
        tactile=np.broadcast_to(values[:, None, None, None], (n, 5, 120, 3)).copy(),
        state=np.broadcast_to(values[:, None], (n, 18)).copy(),
        actions=np.broadcast_to((values + 100)[:, None], (n, 18)).copy(),
        frame_ids=frames, timestamps=frames / 15.0,
    )


def columns(ep):
    return {
        "episode_index": np.full(len(ep.state), ep.episode_id, dtype=np.int64),
        "task_index": np.full(len(ep.state), ep.task_id, dtype=np.int64),
        "frame_index": ep.frame_ids,
        "timestamp": ep.timestamps,
        "observation.state": ep.state,
        "observation.tactile": ep.tactile,
        "action": ep.actions,
        "front": ep.images["front"],
    }


class TouchWorldDataTest(unittest.TestCase):
    def test_action_and_terminal_observation_have_different_offsets(self):
        data = EpisodeWindowDataset([episode()], horizon=4, history=3)
        sample = data[1]
        np.testing.assert_array_equal(sample["actions"][:, 0], [101, 102, 103, 104])
        self.assertEqual(sample["future_tactile"][0, 0, 0], 5)
        self.assertEqual(sample["future_image"]["front"].shape, (4, 6, 3))
        np.testing.assert_array_equal(sample["tactile_history"][:, 0, 0, 0], [0, 0, 1])
        np.testing.assert_array_equal(sample["history_valid"], [False, True, True])
        self.assertTrue(sample["future_valid"])
        self.assertEqual(sample["task_id"], 2)

    def test_offset_history_state_and_suffix_keep_original_chunk_alignment(self):
        data = EpisodeWindowDataset([episode()], horizon=4, history=3)
        sample = data.sample(1, offset=2)
        np.testing.assert_array_equal(sample["tactile_history"][:, 0, 0, 0], [1, 2, 3])
        self.assertEqual(sample["state"][0], 3)
        self.assertEqual(sample["chunk_state"][0], 1)
        self.assertEqual(sample["image"]["front"][0, 0, 0], 1)
        self.assertEqual(sample["future_tactile"][0, 0, 0], 5)
        np.testing.assert_array_equal(sample["actions"][:, 0], [101, 102, 103, 104])
        np.testing.assert_array_equal(sample["action_valid"], [False, False, True, True])

    def test_future_changes_do_not_change_legal_observations(self):
        original = episode()
        changed = episode()
        changed.tactile[4:] += 1000
        changed.state[4:] += 1000
        changed.images["front"][4:] += 10
        a = EpisodeWindowDataset([original], horizon=4).sample(1, offset=2)
        b = EpisodeWindowDataset([changed], horizon=4).sample(1, offset=2)
        for key in ("tactile", "tactile_history", "state", "chunk_state"):
            np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a["image"]["front"], b["image"]["front"])
        self.assertFalse(np.array_equal(a["future_tactile"], b["future_tactile"]))

    def test_tail_padding_never_reads_next_episode(self):
        data = EpisodeWindowDataset([episode(frames=range(3)), episode(1, frames=range(3), base=50)], horizon=4)
        sample = data[2]
        np.testing.assert_array_equal(sample["action_valid"], [True, False, False, False])
        np.testing.assert_array_equal(sample["actions"][:, 0], [102, 0, 0, 0])
        self.assertFalse(sample["future_valid"])
        self.assertFalse(sample["future_tactile"].any())
        self.assertFalse(sample["future_image"]["front"].any())
        self.assertEqual(data[3]["tactile"][0, 0, 0], 50)

    def test_frame_gap_is_not_treated_as_consecutive_time(self):
        data = EpisodeWindowDataset([episode(frames=[0, 1, 3, 4])], horizon=4, history=3)
        sample = data[0]
        np.testing.assert_array_equal(sample["action_valid"], [True, True, False, True])
        self.assertEqual(data.valid_offsets(0), (0, 1))
        with self.assertRaisesRegex(ValueError, "missing current"):
            data.sample(0, offset=3)
        np.testing.assert_array_equal(data[2]["history_valid"], [True, False, True])

    def test_episode_split_reserves_only_training_data(self):
        split = split_training_episodes(range(800), range(800, 900), seed=7)
        self.assertEqual((len(split.train), len(split.validation), len(split.test)), (720, 80, 100))
        self.assertEqual(set(split.train) | set(split.validation), set(range(800)))
        self.assertEqual(split.test, tuple(range(800, 900)))
        self.assertEqual(split, split_training_episodes(reversed(range(800)), range(800, 900), seed=7))
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_episode_splits([0, 1], [], [1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_episode_splits([0, 0])

    def test_separate_dataset_namespaces_allow_reused_episode_numbers(self):
        split = split_training_episodes(range(800), range(100), train_dataset_id="train_repo", test_dataset_id="test_repo")
        self.assertEqual(len(split.test), 100)
        data = EpisodeWindowDataset([episode(dataset_id="train_repo"), episode(dataset_id="test_repo")])
        self.assertEqual(len(data), 16)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            EpisodeWindowDataset([episode(), episode()])

    def test_explicit_columns_sort_all_modalities_together(self):
        source = columns(episode())
        order = np.array([3, 1, 0, 2, 5, 4, 7, 6])
        source = {key: value[order] for key, value in source.items()}
        parsed = episode_from_lerobot_columns(0, source, image_keys=("front",))
        np.testing.assert_array_equal(parsed.state[:, 0], np.arange(8))
        np.testing.assert_array_equal(parsed.tactile[:, 0, 0, 0], np.arange(8))
        np.testing.assert_array_equal(parsed.images["front"][:, 0, 0, 0], np.arange(8))

    def test_raw_state_layout_and_named_joint_reordering(self):
        source = columns(episode(frames=range(2)))
        raw = np.zeros((2, 1972), dtype=np.float32)
        indices = [*range(6), *(28 + 2 * np.arange(12)).tolist()]
        raw[:, indices] = np.arange(18)[None]
        for finger in range(5):
            raw[:, 52 + finger * 384 + 24:52 + finger * 384 + 384] = finger + 1
        source["observation.state"] = raw
        del source["observation.tactile"]
        with self.assertRaisesRegex(ValueError, "allow_legacy_raw_layout"):
            episode_from_lerobot_columns(0, source, image_keys=("front",))
        parsed = episode_from_lerobot_columns(0, source, image_keys=("front",), allow_legacy_raw_layout=True)
        np.testing.assert_array_equal(parsed.state[0], np.arange(18))
        np.testing.assert_array_equal(parsed.tactile[0, :, 0, 0], np.arange(1, 6))
        source = columns(episode(frames=range(2)))
        source["observation.state"] = np.broadcast_to(np.arange(18)[::-1], (2, 18))
        parsed = episode_from_lerobot_columns(0, source, image_keys=("front",), state_names=JOINT_NAMES[::-1])
        np.testing.assert_array_equal(parsed.state[0], np.arange(18))

    def test_reordered_raw_metadata_reads_named_taxels_without_legacy_offsets(self):
        source = columns(episode(frames=range(2)))
        raw = np.zeros((2, 1972), dtype=np.float32)
        names = [f"unused_{i}" for i in range(1972)]
        joint_indices = [*range(6), *(28 + 2 * np.arange(12)).tolist()]
        for i, index in enumerate(joint_indices):
            names[index] = JOINT_NAMES[i]
            raw[:, index] = i + 10
        tactile_indices = [52 + finger * 384 + 24 + taxel * 3 + axis
                           for finger in range(5) for taxel in range(120) for axis in range(3)]
        for i, index in enumerate(tactile_indices):
            names[index] = RAW_TACTILE_NAMES[i]
            raw[:, index] = i + 1000
        order = np.random.default_rng(9).permutation(1972)
        source["observation.state"] = raw[:, order]
        del source["observation.tactile"]
        parsed = episode_from_lerobot_columns(0, source, image_keys=("front",),
                                              state_names=[names[i] for i in order])
        np.testing.assert_array_equal(parsed.state[0], np.arange(18) + 10)
        np.testing.assert_array_equal(parsed.tactile[0].reshape(-1), np.arange(1800) + 1000)

    def test_joint_metadata_does_not_authorize_positional_tactile_extraction(self):
        source = columns(episode(frames=range(2)))
        raw = np.zeros((2, 1972), dtype=np.float32)
        names = [*JOINT_NAMES, *[f"unknown_{i}" for i in range(1954)]]
        source["observation.state"] = raw
        del source["observation.tactile"]
        with self.assertRaisesRegex(ValueError, "positional tactile"):
            episode_from_lerobot_columns(0, source, image_keys=("front",), state_names=names)
        source["observation.tactile"] = np.zeros((2, 5, 120, 3), dtype=np.float32)
        # Explicit tactile does not bypass the independent raw joint-layout guard.
        with self.assertRaisesRegex(ValueError, "positional state"):
            episode_from_lerobot_columns(0, source, image_keys=("front",))
        episode_from_lerobot_columns(0, source, image_keys=("front",), state_names=names)

    def test_explicit_video_loader_and_missing_data_fail_loudly(self):
        source = columns(episode())
        del source["front"]
        with self.assertRaisesRegex(ValueError, "image_loader"):
            episode_from_lerobot_columns(0, source, image_keys=("front",))
        calls = []
        def decode(episode_id, camera, frame_ids, timestamps):
            calls.append((episode_id, camera, frame_ids.copy()))
            return episode().images["front"]
        parsed = episode_from_lerobot_columns(0, source, image_keys=("front",), image_loader=decode)
        self.assertEqual(len(calls), 1)
        self.assertEqual(parsed.images["front"].shape, (8, 4, 6, 3))
        source["task_index"][-1] = 3
        with self.assertRaisesRegex(ValueError, "task_id"):
            episode_from_lerobot_columns(0, source, image_keys=("front",), image_loader=decode)

    def test_lazy_video_access_decodes_only_anchor_and_terminal_frames(self):
        source = episode()
        seen = []
        def decode(rows):
            seen.extend(rows.tolist())
            return source.images["front"][rows]
        lazy = LazyImageSequence(8, (4, 6, 3), decode)
        ep = dataclasses.replace(source, images={"front": lazy})
        data = EpisodeWindowDataset([ep], horizon=4)
        self.assertEqual(seen, [])
        sample = data.sample(1, offset=2)
        self.assertEqual(seen, [1, 5])
        self.assertEqual(sample["image"]["front"][0, 0, 0], 1)
        self.assertEqual(sample["future_image"]["front"][0, 0, 0], 5)
        data[7]
        self.assertEqual(seen, [1, 5, 7])  # Invalid goal does not ask the decoder.

    def test_local_loader_reads_only_explicit_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            for i in range(2):
                (root / "data" / f"episode_{i:06d}.parquet").touch()
            seen = []
            def read(path):
                seen.append(path.name)
                return columns(episode(int(path.stem.split("_")[-1])))
            loaded = load_lerobot_episodes(root, [1], image_keys=("front",), read_table=read)
            self.assertEqual([e.episode_id for e in loaded], [1])
            self.assertEqual(seen, ["episode_000001.parquet"])
            self.assertEqual(loaded[0].dataset_id, str(root.resolve()))
            with self.assertRaisesRegex(ValueError, "not found"):
                load_lerobot_episodes(root, [2], image_keys=("front",), read_table=read)

    def test_invalid_layout_and_offsets_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame_ids"):
            dataclasses.replace(episode(), frame_ids=np.zeros(8, dtype=int))
        with self.assertRaisesRegex(ValueError, "tactile"):
            dataclasses.replace(episode(), tactile=np.zeros((8, 5, 3)))
        data = EpisodeWindowDataset([episode()])
        for offset in (-1, 16, 1.5, True):
            with self.assertRaisesRegex(ValueError, "offset"):
                data.sample(0, offset=offset)


if __name__ == "__main__":
    unittest.main()
