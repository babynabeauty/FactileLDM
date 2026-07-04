#!/usr/bin/env python3
"""Visualize what future tactile tokens predict.

This script compares student and teacher future-token probes on the same eval
samples and saves intuitive figures:

  - sample_XXX_heatmap.png:
      Last-history baseline / Student / Teacher / GT future force magnitude.
  - sample_XXX_curves.png:
      Per-finger force-magnitude curves over future steps.
  - sample_XXX_arrays.npz:
      Raw arrays used for plotting.

Example:

export HF_LEROBOT_HOME=$PWD
export HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 env/.venv/bin/python -u \
  scripts/visualize_future_tactile_probe_comparison.py \
  --student-probe-dir checkpoints/future_tactile_token_probe/pi0_xhand_tactile_structured_dual_ae_history_future_pool/probe_student_task123_valsplit/student \
  --teacher-probe-dir checkpoints/future_tactile_token_probe/pi0_xhand_tactile_structured_dual_ae_history_future_pool/probe_teacher_task123_valsplit/teacher \
  --output-dir outputs/future_tactile_probe_eval/comparison_task123_valsplit \
  --max-batches 20 \
  --num-plot-samples 24 \
  --batch-size 4 \
  --fsdp-devices 1 \
  --num-workers 0
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sys
import time
from typing import Literal

_START_TIME = time.time()


def _stage(message: str) -> None:
    print(f"[viz_future_probe +{time.time() - _START_TIME:7.1f}s] {message}", flush=True)


_stage("starting imports")
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

_stage("importing openpi/probe modules")
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
from openpi.models import gemma as _gemma

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_future_tactile_token_probe as probe_lib

_stage("imports finished")


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclasses.dataclass(frozen=True)
class Args:
    student_probe_dir: str
    teacher_probe_dir: str
    output_dir: str = "outputs/future_tactile_probe_eval/comparison"

    pretrained_params: str | None = None
    config_name: str | None = None
    repo_id: str | None = None
    asset_id: str | None = None
    assets_dir: str | None = None
    eval_filter_path: str | None = None
    probe_layer: int | None = None

    batch_size: int | None = None
    fsdp_devices: int | None = None
    num_workers: int | None = None
    seed: int | None = None

    max_batches: int = 20
    num_plot_samples: int = 24
    contact_threshold: float | None = None
    save_npz: bool = True


def _resolve_probe_dir(path: pathlib.Path, expected_name: Literal["student", "teacher"]) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.name == expected_name and (path.parent / "args.json").exists():
        return path
    child = path / expected_name
    if child.exists() and (path / "args.json").exists():
        return child
    raise FileNotFoundError(
        f"Could not resolve {expected_name} probe dir from {path}. "
        f"Pass either the experiment dir or its {expected_name}/ child."
    )


def _load_probe_args(probe_dir: pathlib.Path) -> probe_lib.Args:
    args_path = probe_dir / "args.json"
    if not args_path.exists():
        args_path = probe_dir.parent / "args.json"
    raw = json.loads(args_path.read_text())
    valid = {field.name for field in dataclasses.fields(probe_lib.Args)}
    return probe_lib.Args(**{key: value for key, value in raw.items() if key in valid})


def _merge_eval_args(cli: Args, saved: probe_lib.Args) -> probe_lib.Args:
    return dataclasses.replace(
        saved,
        pretrained_params=cli.pretrained_params or saved.pretrained_params,
        config_name=cli.config_name or saved.config_name,
        repo_id=cli.repo_id if cli.repo_id is not None else saved.repo_id,
        asset_id=cli.asset_id if cli.asset_id is not None else saved.asset_id,
        assets_dir=cli.assets_dir if cli.assets_dir is not None else saved.assets_dir,
        eval_filter_path=cli.eval_filter_path if cli.eval_filter_path is not None else saved.eval_filter_path,
        train_filter_path=None,
        probe_layer=cli.probe_layer if cli.probe_layer is not None else saved.probe_layer,
        batch_size=cli.batch_size or saved.batch_size,
        fsdp_devices=cli.fsdp_devices or saved.fsdp_devices,
        num_workers=saved.num_workers if cli.num_workers is None else cli.num_workers,
        seed=saved.seed if cli.seed is None else cli.seed,
        contact_threshold=saved.contact_threshold if cli.contact_threshold is None else cli.contact_threshold,
        resume=True,
        overwrite=False,
    )


def _restore_probe_state(
    probe_dir: pathlib.Path,
    probe_state: probe_lib.ProbeTrainState,
) -> tuple[probe_lib.ProbeTrainState, int]:
    manager = ocp.CheckpointManager(
        probe_dir,
        item_handlers={"probe_state": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    latest_step = manager.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No Orbax probe checkpoints found under {probe_dir}")
    restored = manager.restore(
        latest_step,
        args=ocp.args.Composite(probe_state=ocp.args.PyTreeRestore(probe_state)),
    )["probe_state"]
    return restored, int(latest_step)


def _make_probe_state(
    *,
    input_dim: int,
    args: probe_lib.Args,
    model_config,
    rng,
    replicated,
) -> tuple[nnx.GraphDef, probe_lib.ProbeTrainState]:
    probe = probe_lib.FutureTactileForceProbe(
        input_dim=input_dim,
        hidden_dim=args.decoder_dim,
        action_horizon=model_config.action_horizon,
        num_fingers=model_config.tactile_num_fingers,
        force_dim=model_config.tactile_dim_per_finger,
        depth=args.decoder_depth,
        rngs=nnx.Rngs(rng),
    )
    probe_def, probe_params = nnx.split(probe)
    tx = optax.adamw(args.learning_rate)
    state = probe_lib.ProbeTrainState(
        step=0,
        params=probe_params,
        opt_state=tx.init(probe_params),
        tx=tx,
    )
    return probe_def, jax.device_put(state, replicated)


def _mag(force: np.ndarray) -> np.ndarray:
    return np.linalg.norm(force, axis=-1)


def _repeat_last_history(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1:, :, :], repeats=horizon, axis=0)


def _sample_contact_f1(pred_mag: np.ndarray, target_mag: np.ndarray, threshold: float) -> tuple[float, float, float]:
    pred = pred_mag > threshold
    true = target_mag > threshold
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, np.logical_not(true)).sum()
    fn = np.logical_and(np.logical_not(pred), true).sum()
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
    return precision, recall, float(f1)


def _plot_heatmap(
    path: pathlib.Path,
    *,
    baseline: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
    target: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    arrays = [
        ("Last-history baseline", _mag(baseline)),
        ("Student prediction", _mag(student)),
        ("Teacher prediction", _mag(teacher)),
        ("GT future tactile", _mag(target)),
    ]
    vmax = max(float(np.percentile(arr, 99)) for _, arr in arrays)
    vmax = max(vmax, threshold * 1.5, 1e-6)

    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    im = None
    for ax, (name, values) in zip(axes, arrays, strict=True):
        # imshow expects [rows, cols] = [finger, time].
        im = ax.imshow(values.T, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=vmax)
        ax.set_yticks(np.arange(len(FINGER_NAMES)))
        ax.set_yticklabels(FINGER_NAMES)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=92)
        ax.grid(False)

        contact = values.T > threshold
        ys, xs = np.where(contact)
        if len(xs):
            ax.scatter(xs, ys, s=8, c="white", marker=".", alpha=0.75, linewidths=0)

    axes[-1].set_xlabel("future step")
    axes[-1].set_xticks(np.arange(0, target.shape[0], max(1, target.shape[0] // 8)))
    axes[-1].set_xticklabels([str(x + 1) for x in axes[-1].get_xticks()])
    fig.suptitle(title)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
        cbar.set_label("force magnitude")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(
    path: pathlib.Path,
    *,
    baseline: np.ndarray,
    student: np.ndarray,
    teacher: np.ndarray,
    target: np.ndarray,
    threshold: float,
    title: str,
) -> None:
    bmag = _mag(baseline)
    smag = _mag(student)
    tmag = _mag(teacher)
    gmag = _mag(target)
    steps = np.arange(1, target.shape[0] + 1)

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
    for finger, ax in enumerate(axes):
        ax.plot(steps, gmag[:, finger], color="black", linewidth=2.2, label="GT")
        ax.plot(steps, smag[:, finger], color="tab:blue", linestyle="--", linewidth=1.8, label="Student")
        ax.plot(steps, tmag[:, finger], color="tab:orange", linestyle="-.", linewidth=1.8, label="Teacher")
        ax.plot(steps, bmag[:, finger], color="tab:gray", linestyle=":", linewidth=1.8, label="Last hist")
        ax.axhline(threshold, color="tab:red", linewidth=0.8, alpha=0.5, label="contact threshold" if finger == 0 else None)
        ax.set_ylabel(FINGER_NAMES[finger])
        ax.grid(True, alpha=0.25)
        if finger == 0:
            ax.legend(ncol=5, fontsize=9, loc="upper right")
    axes[-1].set_xlabel("future step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sample(
    output_dir: pathlib.Path,
    sample_index: int,
    *,
    student: np.ndarray,
    teacher: np.ndarray,
    target: np.ndarray,
    history: np.ndarray,
    threshold: float,
    save_npz: bool,
) -> None:
    baseline = _repeat_last_history(history, target.shape[0])

    _, _, student_f1 = _sample_contact_f1(_mag(student), _mag(target), threshold)
    _, _, teacher_f1 = _sample_contact_f1(_mag(teacher), _mag(target), threshold)
    _, _, baseline_f1 = _sample_contact_f1(_mag(baseline), _mag(target), threshold)
    title = (
        f"sample={sample_index:03d} | contact F1: "
        f"baseline={baseline_f1:.3f}, student={student_f1:.3f}, teacher={teacher_f1:.3f}"
    )
    _plot_heatmap(
        output_dir / f"sample_{sample_index:03d}_heatmap.png",
        baseline=baseline,
        student=student,
        teacher=teacher,
        target=target,
        threshold=threshold,
        title=title,
    )
    _plot_curves(
        output_dir / f"sample_{sample_index:03d}_curves.png",
        baseline=baseline,
        student=student,
        teacher=teacher,
        target=target,
        threshold=threshold,
        title=title,
    )
    if save_npz:
        np.savez_compressed(
            output_dir / f"sample_{sample_index:03d}_arrays.npz",
            history=history,
            baseline=baseline,
            student=student,
            teacher=teacher,
            target=target,
            threshold=np.asarray(threshold, dtype=np.float32),
        )


def _scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    student_dir = _resolve_probe_dir(pathlib.Path(args.student_probe_dir), "student")
    teacher_dir = _resolve_probe_dir(pathlib.Path(args.teacher_probe_dir), "teacher")
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    student_saved = _load_probe_args(student_dir)
    teacher_saved = _load_probe_args(teacher_dir)
    eval_args = _merge_eval_args(args, dataclasses.replace(student_saved, token_source="student"))
    teacher_args = _merge_eval_args(args, dataclasses.replace(teacher_saved, token_source="teacher"))
    contact_threshold = float(args.contact_threshold if args.contact_threshold is not None else eval_args.contact_threshold)

    (output_dir / "visualize_args.json").write_text(
        json.dumps(
            {
                "cli": dataclasses.asdict(args),
                "student_probe_args": dataclasses.asdict(student_saved),
                "teacher_probe_args": dataclasses.asdict(teacher_saved),
                "merged_eval_args": dataclasses.asdict(eval_args),
                "contact_threshold": contact_threshold,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    _stage(f"student probe: {student_dir}")
    _stage(f"teacher probe: {teacher_dir}")
    _stage(f"output_dir: {output_dir}")
    _stage("building eval config")
    eval_config = probe_lib._override_config(eval_args, filter_path=eval_args.eval_filter_path)
    probe_lib._validate_model_config(eval_config)
    probe_layer = int(eval_args.probe_layer if eval_args.probe_layer is not None else eval_config.model.future_tactile_align_layer)

    _stage(f"creating mesh fsdp_devices={eval_args.fsdp_devices}; jax_devices={jax.devices()}")
    mesh = sharding.make_mesh(eval_args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    _stage("creating eval dataloader")
    eval_loader = _data_loader.create_data_loader(eval_config, sharding=data_sharding, shuffle=False)

    rng = jax.random.PRNGKey(eval_args.seed)
    model_rng, student_probe_rng, teacher_probe_rng, eval_rng = jax.random.split(rng, 4)
    with sharding.set_mesh(mesh):
        _stage("initializing frozen policy model")
        model_def, model_params = probe_lib._init_frozen_model(eval_config, model_rng)
        model_params = jax.device_put(model_params, replicated)

        student_width = int(_gemma.get_config(eval_config.model.action_expert_variant).width)
        teacher_variant = getattr(eval_config.model, "force_expert_variant", eval_config.model.action_expert_variant)
        teacher_width = int(_gemma.get_config(teacher_variant).width)

        _stage("initializing/restoring student probe")
        student_probe_def, student_state = _make_probe_state(
            input_dim=student_width,
            args=eval_args,
            model_config=eval_config.model,
            rng=student_probe_rng,
            replicated=replicated,
        )
        student_state, student_step = _restore_probe_state(student_dir, student_state)

        _stage("initializing/restoring teacher probe")
        teacher_probe_def, teacher_state = _make_probe_state(
            input_dim=teacher_width,
            args=teacher_args,
            model_config=eval_config.model,
            rng=teacher_probe_rng,
            replicated=replicated,
        )
        teacher_state, teacher_step = _restore_probe_state(teacher_dir, teacher_state)

        def _eval_batch(model_params_arg, student_state_arg, teacher_state_arg, batch_arg, rng_arg):
            model = nnx.merge(model_def, model_params_arg)
            model.eval()
            student_probe = nnx.merge(student_probe_def, student_state_arg.params)
            teacher_probe = nnx.merge(teacher_probe_def, teacher_state_arg.params)

            student_hidden, target_force, history_force = probe_lib._extract_future_token_hiddens(
                model,
                rng_arg,
                batch_arg[0],
                batch_arg[1],
                probe_layer=probe_layer,
                token_source="student",
            )
            teacher_hidden, teacher_target_force, _ = probe_lib._extract_future_token_hiddens(
                model,
                rng_arg,
                batch_arg[0],
                batch_arg[1],
                probe_layer=probe_layer,
                token_source="teacher",
            )
            pred_student = student_probe(student_hidden)
            pred_teacher = teacher_probe(teacher_hidden)

            _, student_stats = probe_lib._loss_and_stats(eval_args, pred_student, target_force, history_force)
            _, teacher_stats = probe_lib._loss_and_stats(teacher_args, pred_teacher, teacher_target_force, history_force)
            return student_stats, teacher_stats, pred_student, pred_teacher, target_force, history_force

        peval = jax.jit(_eval_batch)
        _stage("JIT eval function created; first batch includes compile time")

        totals_student: dict[str, float] = {}
        totals_teacher: dict[str, float] = {}
        count = 0
        plotted = 0
        eval_iter = iter(eval_loader)
        batch_idx = 0
        while args.max_batches <= 0 or batch_idx < args.max_batches:
            _stage(f"fetching batch {batch_idx + 1}/{args.max_batches if args.max_batches > 0 else '?'}")
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            _stage(f"processing batch {batch_idx + 1}")
            student_stats, teacher_stats, pred_student, pred_teacher, target, history = peval(
                model_params,
                student_state,
                teacher_state,
                batch,
                eval_rng,
            )
            jax.block_until_ready(pred_student)
            batch_size = int(pred_student.shape[0])
            for key, value in student_stats.items():
                totals_student[key] = totals_student.get(key, 0.0) + _scalar(value) * batch_size
            for key, value in teacher_stats.items():
                totals_teacher[key] = totals_teacher.get(key, 0.0) + _scalar(value) * batch_size
            count += batch_size

            if plotted < args.num_plot_samples:
                ps = np.asarray(jax.device_get(pred_student))
                pt = np.asarray(jax.device_get(pred_teacher))
                tg = np.asarray(jax.device_get(target))
                hs = np.asarray(jax.device_get(history))
                for i in range(min(batch_size, args.num_plot_samples - plotted)):
                    _plot_sample(
                        output_dir,
                        plotted,
                        student=ps[i],
                        teacher=pt[i],
                        target=tg[i],
                        history=hs[i],
                        threshold=contact_threshold,
                        save_npz=args.save_npz,
                    )
                    plotted += 1
            batch_idx += 1

    if count == 0:
        raise RuntimeError("No eval samples were processed.")

    metrics = {
        "student": {key: value / count for key, value in sorted(totals_student.items())},
        "teacher": {key: value / count for key, value in sorted(totals_teacher.items())},
        "probe_layer": probe_layer,
        "student_probe_step": student_step,
        "teacher_probe_step": teacher_step,
        "eval_samples": count,
        "contact_threshold": contact_threshold,
        "output_dir": str(output_dir),
    }
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    logging.info("Saved comparison visualizations to %s", output_dir)
    logging.info("Student contact_f1 = %.6f", metrics["student"]["metric/contact_f1"])
    logging.info("Teacher contact_f1 = %.6f", metrics["teacher"]["metric/contact_f1"])
    logging.info("Student force_smooth_l1 = %.6f", metrics["student"]["loss/force_smooth_l1"])
    logging.info("Teacher force_smooth_l1 = %.6f", metrics["teacher"]["loss/force_smooth_l1"])


if __name__ == "__main__":
    main(tyro.cli(Args))
