"""NumPy-only controls for contact information in a frozen FAE readout.

No fitting of the policy or probe. Reassignment is within-task, cross-episode,
at the same update offset, with donor validity covering the recipient window.
"""

from __future__ import annotations

import logging

import numpy as np


METRICS = ("contact_ap", "contact_f1", "contact_precision", "contact_recall", "positive_rate")


def fit_contact_prior(shards, num_tasks):
    """Task/finger/patch frequencies from TRAIN labels, each cached future once.

Use the offset-zero validity mask so later offsets do not duplicate targets.
Cached overlapping windows still overlap, as in the original probe protocol.
Laplace smoothing uses one positive and one negative pseudo-observation.
"""
    positives = counts = None
    for cache in shards:
        labels = cache["target_contact"]
        if positives is None:
            positives = np.zeros((num_tasks, *labels.shape[2:]), dtype=np.float64)
            counts = np.zeros(num_tasks, dtype=np.int64)
        zero = np.flatnonzero(cache["offsets"] == 0)
        if len(zero) != 1:
            raise ValueError("Prior fitting requires exactly one offset zero.")
        valid = cache["valid_mask"][:, zero[0]]
        for task in np.unique(cache["task_id"]):
            selected = cache["task_id"] == task
            values = labels[selected][valid[selected]]
            positives[task] += values.sum(axis=0)
            counts[task] += len(values)
    if positives is None:
        raise ValueError("No training shards.")
    return (positives + 1) / (counts[:, None, None] + 2), counts


def contact_scopes(labels, current, valid, offsets):
    """Yield name, frame/patch mask, inversion flag; lead is relative to update.

Change subsets are outcome-conditioned diagnostics, not overall performance.
Onset/release candidate sets depend ONLY on the observed current contact state.
"""
    horizon = labels.shape[1]
    steps = np.arange(1, horizon + 1)
    yield "all_future", valid[..., None, None], False
    for oi, offset in enumerate(offsets):
        selected = np.zeros_like(valid)
        selected[:, oi] = valid[:, oi]
        yield f"offset_{int(offset)}", selected[..., None, None], False
    lead = steps[None, :] - offsets[:, None]
    for distance in range(1, horizon + 1):
        yield f"lead_{distance}", (valid & (lead[None] == distance))[..., None, None], False
    change = labels[:, None] != current[:, :, None]
    yield "contact_change_frames", (valid & change.any(axis=(-2, -1)))[..., None, None], False
    yield "changed_patches", valid[..., None, None] & change, False
    yield "onset_candidates", valid[..., None, None] & ~current[:, :, None].astype(bool), False
    yield "release_candidates", valid[..., None, None] & current[:, :, None].astype(bool), True


def cross_episode_donors(task_ids, episode_ids, valid, rng):
    """Draw donors with replacement; NOT an exact permutation or a null p-test.

For each recipient, uniformly draw an eligible OTHER episode, then a donor
anchor within that episode. No label values or predicted scores enter selection.
The same donor map must be used for fixed/refined and all evaluation scopes.
"""
    donors = np.broadcast_to(np.arange(len(task_ids))[:, None], valid.shape[:2]).copy()
    for task in np.unique(task_ids):
        task_rows = np.flatnonzero(task_ids == task)
        for oi in range(valid.shape[1]):
            masks, membership = np.unique(valid[task_rows, oi], axis=0, return_inverse=True)
            for mi, mask in enumerate(masks):
                if not mask.any():
                    continue
                recipients = task_rows[membership == mi]
                pool = task_rows[np.all(valid[task_rows, oi][:, mask], axis=1)]
                by_episode = {ep: pool[episode_ids[pool] == ep] for ep in np.unique(episode_ids[pool])}
                for recipient_episode in np.unique(episode_ids[recipients]):
                    rows = recipients[episode_ids[recipients] == recipient_episode]
                    choices = np.array([ep for ep in by_episode if ep != recipient_episode])
                    if not len(choices):
                        raise ValueError(
                            f"No valid cross-episode donor: task={task}, episode={recipient_episode}, offset_index={oi}. "
                            "No rows are silently dropped; use a dataset with at least two eligible episodes per task."
                        )
                    chosen = rng.choice(choices, size=len(rows))
                    for ep in np.unique(chosen):
                        selected = rows[chosen == ep]
                        donors[selected, oi] = rng.choice(by_episode[ep], size=len(selected), replace=True)
    return donors


class ContactGroup:
    """Tie-aware exact AP and F1, sorted once for episode bootstrap reuse."""

    def __init__(self, scores, labels, episodes, threshold):
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError("ContactGroup requires nonempty finite scores.")
        if not 0 < threshold < 1:
            raise ValueError("Expected a threshold strictly between zero and one.")
        order = np.argsort(-scores, kind="stable")
        self.scores = scores[order]
        self.labels = labels[order].astype(bool)
        self.ids = episodes[order]
        self.ends = np.r_[np.flatnonzero(np.diff(self.scores)), len(scores) - 1]
        self.threshold = threshold
        self.episodes = np.unique(episodes)

    def compute(self, weights=None):
        weights = np.ones(len(self.ids), dtype=np.float64) if weights is None else weights[self.ids].astype(np.float64)
        positives = weights * self.labels
        positive_count = positives.sum()
        total = weights.sum()
        tp_curve = np.cumsum(positives)[self.ends]
        n_curve = np.cumsum(weights)[self.ends]
        precision = np.divide(tp_curve, n_curve, out=np.zeros_like(tp_curve), where=n_curve > 0)
        ap = float(np.sum(precision * np.diff(np.r_[0, tp_curve])) / positive_count) if positive_count else np.nan
        predicted = self.scores >= np.log(self.threshold / (1 - self.threshold))
        tp = positives[predicted].sum()
        predicted_count = weights[predicted].sum()
        return {
            "contact_ap": ap,
            "contact_f1": float(2 * tp / (predicted_count + positive_count)) if predicted_count + positive_count else np.nan,
            "contact_precision": float(tp / predicted_count) if predicted_count else np.nan,
            "contact_recall": float(tp / positive_count) if positive_count else np.nan,
            "positive_rate": float(positive_count / total) if total else np.nan,
            "valid_patches": int(total),
            "positive_patches": int(positive_count),
        }


def make_groups(scores, labels, episode_ids, task_ids, mask, threshold, *, invert=False):
    shape = scores.shape
    labels = np.broadcast_to(labels[:, None], shape)
    mask = np.broadcast_to(mask, shape)
    ids = np.broadcast_to(episode_ids[:, None, None, None, None], shape)
    groups = {}
    for task in np.unique(task_ids):
        selected = mask & (task_ids == task)[:, None, None, None, None]
        if selected.any():
            s, y = scores[selected], labels[selected]
            groups[int(task)] = ContactGroup(
                -s if invert else s, ~y.astype(bool) if invert else y,
                ids[selected], 1 - threshold if invert else threshold,
            )
    return groups


def macro(values):
    result = {}
    for metric in METRICS:
        finite = [row[metric] for row in values if np.isfinite(row[metric])]
        result[metric] = float(np.mean(finite)) if finite else np.nan
        result[metric + "_tasks"] = len(finite)
    result["valid_patches"] = sum(row["valid_patches"] for row in values)
    result["positive_patches"] = sum(row["positive_patches"] for row in values)
    return result


def paired_bootstrap(groups, *, repetitions, seed):
    """Task-stratified recipient-episode bootstrap on a shared fixed evaluation set.

For reassigned scores, uncertainty is conditional on one saved donor map, not
on training seeds or donor sampling. Random-map variability is reported apart.
"""
    if repetitions <= 0:
        return {}
    modes = sorted(groups)
    tasks = sorted(groups[modes[0]])
    if any(sorted(groups[mode]) != tasks for mode in modes):
        raise ValueError("Paired comparisons must contain the same tasks.")
    if not tasks:
        return {}
    rng = np.random.default_rng(seed)
    draws = {mode: [] for mode in modes}
    for repetition in range(repetitions):
        values = {mode: [] for mode in modes}
        for task in tasks:
            episodes = groups[modes[0]][task].episodes
            if any(not np.array_equal(groups[mode][task].episodes, episodes) for mode in modes):
                raise ValueError("Paired comparisons must contain the same recipient episodes.")
            weights = np.bincount(rng.choice(episodes, len(episodes), replace=True), minlength=int(episodes.max()) + 1)
            for mode in modes:
                values[mode].append(groups[mode][task].compute(weights))
        for mode in modes:
            draws[mode].append(macro(values[mode]))

        if (repetition + 1) % 25 == 0:
            logging.info("BOOTSTRAP draw=%d/%d", repetition + 1, repetitions)

    def interval(values):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        return {"ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
                "valid_resamples": len(finite)}

    output = {mode: {metric: interval([row[metric] for row in draws[mode]]) for metric in METRICS} for mode in modes}
    for mode in ("fixed", "refined"):
        for baseline in ("prior", "persistence", f"{mode}_shuffled"):
            if mode not in draws or baseline not in draws:
                continue
            output[f"delta_{mode}_minus_{baseline}"] = {
                metric: interval([a[metric] - b[metric] for a, b in zip(draws[mode], draws[baseline], strict=True)])
                for metric in ("contact_ap", "contact_f1")
            }
    return output
