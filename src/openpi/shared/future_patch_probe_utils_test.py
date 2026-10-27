import json

import numpy as np
import pytest

from openpi.shared.future_patch_probe_utils import MetricGroup
from openpi.shared.future_patch_probe_utils import check_parameter_tree
from openpi.shared.future_patch_probe_utils import create_manifest
from openpi.shared.future_patch_probe_utils import future_validity
from openpi.shared.future_patch_probe_utils import paired_episode_bootstrap
from openpi.shared.future_patch_probe_utils import probe_features
from openpi.shared.future_patch_probe_utils import summarize_groups


def make_dataset(path, counts):
    (path / "meta").mkdir(parents=True)
    rows = []
    for task, count in counts.items():
        first = len(rows)
        rows.extend({"episode_index": first + i, "tasks": [task], "length": 20} for i in range(count))
    (path / "meta/info.json").write_text(json.dumps({"total_episodes": len(rows), "fps": 15}))
    (path / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_episode_split_is_stratified_reproducible_and_disjoint(tmp_path):
    make_dataset(tmp_path / "train", {"a": 10, "b": 10})
    make_dataset(tmp_path / "test", {"a": 2, "b": 2})
    m = create_manifest(tmp_path / "train", tmp_path / "test")
    assert m == create_manifest(tmp_path / "train", tmp_path / "test")
    assert len(m["splits"]["val"]["episodes"]) == 2
    assert set(m["splits"]["train"]["episodes"]).isdisjoint(m["splits"]["val"]["episodes"])
    assert len(m["splits"]["test"]["episodes"]) == 4
    with pytest.raises(ValueError, match="separate"):
        create_manifest(tmp_path / "train", tmp_path / "train")


def test_strict_parameter_check_rejects_missing_and_mismatched_weights():
    expected = {"branch": {0: {"weight": np.zeros((2, 3))}}}
    check_parameter_tree(expected, expected)
    with pytest.raises(ValueError, match="Missing"):
        check_parameter_tree(expected, {"branch": {0: {}}})
    with pytest.raises(ValueError, match="mismatch"):
        check_parameter_tree(expected, {"branch": {0: {"weight": np.zeros((3, 2))}}})


def test_future_mask_excludes_current_and_repeated_tail():
    np.testing.assert_array_equal(future_validity(7, 10, 4), [True, True, False, False])
    assert not future_validity(9, 10, 4).any()


def group(scores, labels, force=None, target=None, episodes=None, valid=None):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1, 1, 1)
    labels = np.asarray(labels, dtype=np.float32).reshape(scores.shape)
    shape = (*scores.shape, 3)
    force = np.zeros(shape) if force is None else np.asarray(force).reshape(shape)
    target = np.zeros(shape) if target is None else np.asarray(target).reshape(shape)
    ids = np.arange(len(scores)) if episodes is None else np.asarray(episodes)
    valid = np.ones(scores.shape[:2], dtype=bool) if valid is None else np.asarray(valid).reshape(scores.shape[:2])
    g = MetricGroup()
    g.add(scores, labels, force, target, valid, ids)
    return g.finalize()


def test_exact_ap_ties_weighted_bootstrap_and_global_f1():
    g = group([2, 1, -1], [1, 0, 1], episodes=[0, 1, 2])
    m = g.compute()
    assert m["contact_ap"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["contact_f1"] == pytest.approx(0.5)
    tied = group([0, 0], [1, 0], episodes=[0, 1])
    assert tied.compute()["contact_ap"] == pytest.approx(0.5)
    assert tied.compute(multiplicities=np.array([1, 2]))["contact_ap"] == pytest.approx(1 / 3)
    assert np.isnan(group([1, 0], [0, 0]).compute()["contact_ap"])


def test_force_rmse_uses_only_active_coordinates_and_never_gates_prediction():
    g = group(
        [1, -1, 100],
        [1, 0, 1],
        force=[[1, 2, 2], [4, 4, 4], [999, 999, 999]],
        target=np.zeros((3, 3)),
        valid=[True, True, False],
    )
    m = g.compute()
    assert m["active_force_rmse"] == pytest.approx(np.sqrt(3))
    assert m["active_force_mae"] == pytest.approx(5 / 3)
    assert m["inactive_force_rmse"] == pytest.approx(4)
    assert m["valid_patches"] == 2


def test_macro_equal_task_weights_and_paired_episode_ci():
    groups = {}
    for task, n in (("a", 2), ("b", 8)):
        labels = [1, 0] * (n // 2)
        for mode in ("fixed", "refined"):
            groups[task, mode, "all_future"] = group([1, -1] * (n // 2), labels)
    rows = summarize_groups(groups, 0.5)
    assert all(r["contact_ap"] == 1 for r in rows)
    ci = paired_episode_bootstrap(groups, 0.5, repetitions=10)
    assert ci["delta_refined_minus_fixed"]["contact_ap"]["ci95"] == [0.0, 0.0]


def test_history_baseline_gets_both_finger_tokens_and_lead_time():
    cache = {
        "history_tokens": np.arange(16).reshape(1, 2, 4, 2),
        "target_contact": np.zeros((1, 4, 2, 2)),
        "offsets": np.array([0, 2]),
    }
    result = probe_features(cache, [0, 0], [0, 1], [0, 0], source="history", num_segments=2, num_fingers=2)
    assert result.shape == (2, 4, 7)
    np.testing.assert_array_equal(result[0, 0, :4], [0, 1, 4, 5])
    np.testing.assert_array_equal(result[0, 2, :4], [0, 1, 4, 5])
    assert result[0, 0, 4] == 1
    assert result[0, 2, 5] == 1
    assert result[1, 0, -1] == 0.5
