"""Small NumPy-only tests. No policy, accelerator, package install or real data.

Run: PYTHONPATH=src:. python -m unittest scripts.eval_fae_contact_controls_test
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openpi.shared.contact_probe_controls import ContactGroup
from openpi.shared.contact_probe_controls import contact_scopes
from openpi.shared.contact_probe_controls import cross_episode_donors
from openpi.shared.contact_probe_controls import fit_contact_prior
from openpi.shared.contact_probe_controls import paired_bootstrap
from openpi.shared.future_patch_probe_utils import fingerprint
from openpi.shared.future_patch_probe_utils import MetricGroup
from openpi.shared.future_patch_probe_utils import read_json
from openpi.shared.future_patch_probe_utils import write_json
from scripts import eval_fae_contact_controls as runner


class ContactControlsTest(unittest.TestCase):
    def test_metrics_match_existing_evaluator_including_bootstrap_weights(self):
        rng = np.random.default_rng(2)
        logits = rng.integers(-3, 4, size=(4, 3, 2, 2)).astype(np.float32)
        labels = rng.integers(0, 2, size=logits.shape).astype(bool)
        valid = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], bool)
        episodes = np.array([0, 0, 1, 1])
        force = np.zeros((*logits.shape, 3))
        old = MetricGroup()
        old.add(logits, labels, force, force, valid, episodes)
        old.finalize()
        mask = np.broadcast_to(valid[..., None, None], labels.shape)
        ids = np.broadcast_to(episodes[:, None, None, None], labels.shape)[mask]
        new = ContactGroup(logits[mask], labels[mask], ids, 0.15)
        for weights in (None, np.array([2, 1]), np.array([0, 2])):
            expected = old.compute(threshold=0.15, multiplicities=weights)
            actual = new.compute(weights)
            for key in ("contact_ap", "contact_f1", "valid_patches", "positive_patches"):
                self.assertAlmostEqual(expected[key], actual[key])

    def test_tied_ap_is_prevalence_and_no_positive_is_undefined(self):
        group = ContactGroup(np.zeros(4), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 1]), 0.5)
        self.assertAlmostEqual(group.compute()["contact_ap"], 0.25)
        self.assertAlmostEqual(group.compute()["contact_f1"], 0.4)
        self.assertTrue(np.isnan(group.compute(np.array([0, 1]))["contact_ap"]))
        # Unequal episode multiplicities must affect prevalence and exact AP.
        self.assertAlmostEqual(group.compute(np.array([2, 1]))["contact_ap"], 1 / 3)

    def test_prior_uses_train_valid_frames_only_once(self):
        data = {"target_contact": np.array([[[[1]], [[0]], [[1]]]], dtype=bool),
                "offsets": np.array([0, 1]), "task_id": np.array([0]),
                "valid_mask": np.array([[[1, 1, 0], [0, 1, 0]]], dtype=bool)}
        prior, counts = fit_contact_prior([data], 1)
        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(prior.item(), 0.5)

    def test_donors_are_other_episode_same_task_valid_and_deterministic(self):
        task = np.repeat([0, 1], 6)
        episode = np.repeat(np.arange(4), 3)
        valid = np.ones((12, 2, 4), dtype=bool)
        valid[:, 1, :2] = False
        valid[0, :, -1] = False
        donors = cross_episode_donors(task, episode, valid, np.random.default_rng(3))
        np.testing.assert_array_equal(donors, cross_episode_donors(task, episode, valid, np.random.default_rng(3)))
        for row in range(12):
            for oi in range(2):
                donor = donors[row, oi]
                self.assertEqual(task[row], task[donor])
                self.assertNotEqual(episode[row], episode[donor])
                self.assertTrue(np.all(valid[donor, oi][valid[row, oi]]))
        with self.assertRaisesRegex(ValueError, "No valid cross-episode donor"):
            cross_episode_donors(np.zeros(2), np.zeros(2), np.ones((2, 1, 1), bool), np.random.default_rng(0))

    def test_reassignment_commutes_with_samplewise_affine_decoder(self):
        rng = np.random.default_rng(8)
        x, weight = rng.normal(size=(6, 2, 3)), rng.normal(size=(3, 4))
        donors = cross_episode_donors(np.zeros(6), np.repeat([0, 1, 2], 2), np.ones((6, 2, 1), bool), rng)
        oi = np.arange(2)[None]
        np.testing.assert_allclose((x @ weight + 1)[donors, oi], x[donors, oi] @ weight + 1)

    def test_lead_and_event_scopes_do_not_leak_observed_frames(self):
        labels = np.array([[[[0]], [[1]], [[0]], [[1]]]], bool)
        current = np.array([[[[0]], [[1]]]], bool)
        valid = np.array([[[1, 1, 1, 1], [0, 0, 1, 1]]], bool)
        scopes = {name: (mask, invert) for name, mask, invert in contact_scopes(labels, current, valid, np.array([0, 2]))}
        np.testing.assert_array_equal(scopes["lead_1"][0][..., 0, 0], [[[1, 0, 0, 0], [0, 0, 1, 0]]])
        self.assertTrue(scopes["release_candidates"][1])
        for mask, _ in scopes.values():
            self.assertFalse(np.any(mask[..., 0, 0] & ~valid))
        self.assertEqual(scopes["onset_candidates"][0].sum(), 4)
        self.assertEqual(scopes["release_candidates"][0].sum(), 2)

    def test_paired_identical_scores_have_zero_delta_interval(self):
        group = ContactGroup(np.array([3, 1, -1, -2.]), np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]), 0.5)
        result = paired_bootstrap({"fixed": {0: group}, "prior": {0: group}}, repetitions=10, seed=4)
        self.assertEqual(result["delta_fixed_minus_prior"]["contact_ap"]["ci95"], [0, 0])

    def test_end_to_end_reuses_checkpoint_and_writes_separate_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            protocol = {"immutable_synthetic_protocol": 1}
            write_json(cache_root / "protocol.json", protocol)
            rng = np.random.default_rng(5)
            for split, ids in (("train", [0, 1]), ("val", [2, 3]), ("test", [0, 1])):
                labels = rng.integers(0, 2, (6, 4, 1, 1)).astype(bool)
                hidden = (labels.reshape(6, 4, 1).astype(float) * 6 - 3).astype(np.float32)
                valid = np.ones((6, 2, 4), bool)
                valid[:, 1, :2] = False
                data = dict(target_contact=labels, current_contact=np.zeros((6, 2, 1, 1), bool),
                            episode_id=np.repeat(ids, 3), task_id=np.zeros(6, np.int32), frame_index=np.tile(np.arange(3), 2),
                            offsets=np.array([0, 2]), valid_mask=valid, fixed_hidden=hidden,
                            refined_hidden=np.stack([hidden, hidden], axis=1))
                index = {"complete": True, "episodes": ids, "samples": 6, "task_names": ["task"],
                         "model": {}, "protocol_sha256": fingerprint(protocol),
                         "shards": [{"file": "shard.npz", "samples": 6}]}
                write_json(cache_root / split / "index.json", index)
                np.savez(cache_root / split / "shard.npz", **data)
            checkpoint = root / "best_probe.npz"
            np.savez(checkpoint, sentinel=np.array([123]))
            meta = {"source": "fae", "training_complete": True, "threshold": 0.5,
                    "protocol_sha256": fingerprint(protocol), "train_episodes": [0, 1], "val_episodes": [2, 3]}
            write_json(checkpoint.with_suffix(".json"), meta)
            before = runner.sha256(checkpoint)
            args = argparse.Namespace(cache_dir=cache_root, probe_checkpoint=checkpoint, output_dir=root / "report",
                                      batch_size=4, num_devices=1, shuffle_repeats=2, bootstrap_samples=4, seed=42)

            class IdentityPlacement:
                def __init__(self, *_args):
                    pass

                def replicate(self, value):
                    return value

            def prediction_fn(*_args):
                return lambda _params, x: {"contact_logits": x.reshape(len(x), 4, 1, 1)}

            with patch.object(runner, "ProbeParallel", IdentityPlacement), \
                 patch.object(runner, "_load_probe", return_value=(None, None, meta)), \
                 patch.object(runner, "_prediction_fn", prediction_fn):
                runner.run(args)
            self.assertEqual(before, runner.sha256(checkpoint))
            self.assertEqual(read_json(args.output_dir / "run.json")["status"], "complete")
            report = read_json(args.output_dir / "contact_controls.json")
            real = next(r for r in report["metrics"] if r["task"] == "__macro__" and r["mode"] == "fixed" and r["scope"] == "all_future")
            self.assertEqual(real["contact_ap"], 1.0)
            self.assertEqual(report["thresholds"]["fixed"], 0.5)
            self.assertEqual(len(list(args.output_dir.glob("donors_*.npy"))), 2)
            self.assertIn("delta_fixed_minus_fixed_shuffled", report["paired_episode_bootstrap"]["all_future"])
            with self.assertRaises(FileExistsError):
                runner.run(args)


if __name__ == "__main__":
    unittest.main()
