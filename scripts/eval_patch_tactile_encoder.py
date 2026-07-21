#!/usr/bin/env python3
"""Evaluate pretrained XHand patch tactile encoder decoder heads.

This is a fast local-parquet evaluator for the Stage-1
`xhand_patch_tactile_encoder_pretrain` checkpoint. It reads raw tactile force
from `observation.state`, runs the patch-informed encoder plus its three
pretraining heads, and reports whether one finger token preserves local
5-patch contact structure.

Example:
  env/.venv/bin/python scripts/eval_patch_tactile_encoder.py \
    --repo-id data/task12345-2 \
    --params checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/<step>/params \
    --filter-path outputs/episode_splits/task12345-2_recursive_revision/val_episodes.json \
    --output-dir outputs/patch_encoder_eval/task12345_2 \
    --max-frames 20000 \
    --batch-size 256
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib

import einops
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.models import model as _model
from openpi.policies import xhand_policy
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    params: str
    output_dir: str = "outputs/patch_encoder_eval"
    config_name: str = "xhand_patch_tactile_encoder_pretrain"
    filter_path: str | None = None
    batch_size: int = 256
    max_frames: int | None = 20000
    frame_stride: int = 1
    seed: int = 42


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _episode_filter(path: str | None) -> set[int] | None:
    if path is None:
        return None
    with pathlib.Path(path).expanduser().open() as f:
        data = json.load(f)
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = (
            data.get("episodes")
            or data.get("episode_indices")
            or data.get("episode_index")
            or data.get("val")
            or data.get("train")
        )
    else:
        raise ValueError(f"Unsupported episode filter format: {path}")
    if episodes is None:
        raise ValueError(f"Could not find episode list in {path}")
    return {int(ep) for ep in episodes}


def _episode_files(repo: pathlib.Path, episode_indices: set[int] | None) -> list[pathlib.Path]:
    info_path = repo / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    glob_pattern = data_pattern.replace("{episode_chunk:03d}", "*").replace("{episode_index:06d}", "*")
    files = sorted(repo.glob(glob_pattern))
    if not files:
        files = sorted((repo / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {repo}")
    if episode_indices is None:
        return files
    selected = []
    for path in files:
        try:
            episode = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if episode in episode_indices:
            selected.append(path)
    return selected


def _extract_raw_tactile_from_state(states: np.ndarray) -> np.ndarray:
    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(states[:, start:end].reshape(states.shape[0], xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(chunks, axis=1).astype(np.float32)  # [N, 5, 120, 3]


def _iter_tactile_batches(args: Args):
    repo = pathlib.Path(args.repo_id).expanduser().resolve()
    episode_indices = _episode_filter(args.filter_path)
    files = _episode_files(repo, episode_indices)
    logging.info("Reading %d parquet episodes from %s", len(files), repo)

    rng = np.random.default_rng(args.seed)
    frame_count = 0
    buffer = []
    for parquet_file in files:
        table = pq.read_table(parquet_file, columns=["observation.state"])
        states = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
        if args.frame_stride > 1:
            states = states[:: args.frame_stride]
        tactile = _extract_raw_tactile_from_state(states)
        if args.max_frames is not None and frame_count + tactile.shape[0] > args.max_frames:
            remaining = args.max_frames - frame_count
            if remaining <= 0:
                break
            # Sample within the final episode rather than taking only its prefix.
            indices = np.sort(rng.choice(tactile.shape[0], size=remaining, replace=False))
            tactile = tactile[indices]
        frame_count += tactile.shape[0]

        for frame in tactile:
            buffer.append(frame)
            if len(buffer) == args.batch_size:
                yield np.stack(buffer, axis=0)
                buffer = []
        if args.max_frames is not None and frame_count >= args.max_frames:
            break
    if buffer:
        yield np.stack(buffer, axis=0)


def _load_model(config_name: str, params_path: str):
    train_config = _config.get_config(config_name)
    params = _model.restore_params(pathlib.Path(params_path).expanduser(), restore_type=np.ndarray)
    return train_config.model.load(params)


def _sigmoid_bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    return jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)


def _pearson(x: jax.Array, y: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32).reshape(-1)
    y = y.astype(jnp.float32).reshape(-1)
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / jnp.maximum(jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)), 1e-6)


def _contact_stats(pred_prob: jax.Array, target: jax.Array) -> dict[str, jax.Array]:
    pred = pred_prob >= 0.5
    true = target >= 0.5
    tp = jnp.sum(jnp.logical_and(pred, true))
    fp = jnp.sum(jnp.logical_and(pred, jnp.logical_not(true)))
    fn = jnp.sum(jnp.logical_and(jnp.logical_not(pred), true))
    tn = jnp.sum(jnp.logical_and(jnp.logical_not(pred), jnp.logical_not(true)))
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-6)
    accuracy = (tp + tn) / jnp.maximum(tp + fp + fn + tn, 1.0)
    return {
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "contact_accuracy": accuracy,
    }


def _eval_batch(model, tactile: jax.Array) -> dict[str, jax.Array]:
    effort = tactile[:, None, ...].astype(jnp.float32)  # [B,1,5,120,3]
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)

    dist_logits = model.patch_distribution_head(tokens)
    pred_dist = jax.nn.softmax(dist_logits.astype(jnp.float32), axis=-1)

    summary_pred = model.patch_summary_head(tokens)
    summary_pred = einops.rearrange(
        summary_pred,
        "b t f (r c) -> b t f r c",
        r=model.num_patches,
        c=model.summary_dim,
    ).astype(jnp.float32)

    contact_logits = model.patch_contact_head(tokens).astype(jnp.float32)
    pred_contact = jax.nn.sigmoid(contact_logits)
    target_dist, target_summary, target_contact = model.patch_encoder.patch_reconstruction_targets(effort)
    target_dist = target_dist.astype(jnp.float32)
    target_summary = target_summary.astype(jnp.float32)
    target_contact = target_contact.astype(jnp.float32)

    eps = 1e-6
    dist_ce = -jnp.mean(jnp.sum(target_dist * jnp.log(jnp.maximum(pred_dist, eps)), axis=-1))
    dist_kl = jnp.mean(
        jnp.sum(
            target_dist * (jnp.log(jnp.maximum(target_dist, eps)) - jnp.log(jnp.maximum(pred_dist, eps))),
            axis=-1,
        )
    )
    contact_bce = jnp.mean(_sigmoid_bce_with_logits(contact_logits, target_contact))
    contact = _contact_stats(pred_contact, target_contact)

    pred_strength = summary_pred[..., -1]
    target_strength = target_summary[..., -1]
    strength_mae = jnp.mean(jnp.abs(pred_strength - target_strength))
    strength_smooth_l1 = jnp.mean(_smooth_l1(pred_strength, target_strength))
    summary_smooth_l1 = jnp.mean(_smooth_l1(summary_pred, target_summary))
    strength_corr = _pearson(pred_strength, target_strength)

    metrics = {
        "distribution_ce": dist_ce,
        "distribution_kl": dist_kl,
        "contact_bce": contact_bce,
        "strength_mae": strength_mae,
        "strength_smooth_l1": strength_smooth_l1,
        "summary_smooth_l1": summary_smooth_l1,
        "strength_pearson": strength_corr,
        "pred_contact_ratio": jnp.mean((pred_contact >= 0.5).astype(jnp.float32)),
        "target_contact_ratio": jnp.mean(target_contact),
        "target_strength_mean": jnp.mean(target_strength),
        **contact,
    }

    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        finger_contact = _contact_stats(pred_contact[:, :, finger_idx], target_contact[:, :, finger_idx])
        metrics[f"finger/{finger_name}/contact_f1"] = finger_contact["contact_f1"]
        metrics[f"finger/{finger_name}/contact_accuracy"] = finger_contact["contact_accuracy"]
        metrics[f"finger/{finger_name}/strength_mae"] = jnp.mean(
            jnp.abs(pred_strength[:, :, finger_idx] - target_strength[:, :, finger_idx])
        )
        metrics[f"finger/{finger_name}/strength_pearson"] = _pearson(
            pred_strength[:, :, finger_idx],
            target_strength[:, :, finger_idx],
        )
    return metrics


def _write_outputs(output_dir: pathlib.Path, args: Args, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, ensure_ascii=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (output_dir / "metrics_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "value": metrics[key]})


def _plot_outputs(output_dir: pathlib.Path, metrics: dict[str, float]) -> None:
    f1 = [metrics.get(f"finger/{name}/contact_f1", np.nan) for name in FINGER_NAMES]
    mae = [metrics.get(f"finger/{name}/strength_mae", np.nan) for name in FINGER_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=180)
    axes[0].bar(FINGER_NAMES, f1, color="#2f6f9f")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("contact F1")
    axes[0].set_title("Patch Contact")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(FINGER_NAMES, mae, color="#d07c2c")
    axes[1].set_ylabel("strength MAE")
    axes[1].set_title("Patch Strength")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_finger_patch_metrics.png")
    plt.close(fig)


def main(args: Args) -> None:
    init_logging()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    logging.info("Loading patch encoder checkpoint: %s", args.params)
    model = _load_model(args.config_name, args.params)
    model.eval()

    peval = jax.jit(lambda batch: _eval_batch(model, batch))
    sums: dict[str, float] = {}
    total_frames = 0
    batches = 0
    for tactile_np in _iter_tactile_batches(args):
        metrics = peval(jnp.asarray(tactile_np, dtype=jnp.float32))
        metrics_np = jax.device_get(metrics)
        batch_size = tactile_np.shape[0]
        for key, value in metrics_np.items():
            sums[key] = sums.get(key, 0.0) + float(np.asarray(value)) * batch_size
        total_frames += batch_size
        batches += 1
        if batches == 1 or batches % 20 == 0:
            logging.info("Processed %d frames.", total_frames)

    if total_frames == 0:
        raise RuntimeError("No frames were evaluated.")
    metrics = {key: value / total_frames for key, value in sorted(sums.items())}
    metrics["eval_frames"] = float(total_frames)
    metrics["eval_batches"] = float(batches)
    _write_outputs(output_dir, args, metrics)
    _plot_outputs(output_dir, metrics)
    logging.info("Saved patch encoder metrics to %s", output_dir)
    logging.info(
        "Summary: contact_f1=%.4f strength_mae=%.4f strength_pearson=%.4f distribution_kl=%.4f",
        metrics["contact_f1"],
        metrics["strength_mae"],
        metrics["strength_pearson"],
        metrics["distribution_kl"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
