#!/usr/bin/env python3
"""Offline recursive revision analysis for cached-VLM async tactile foresight.

This script evaluates whether asynchronous tactile updates improve future
tactile latents and action chunks. It compares:

  one_shot:      run at offset 0 once and reuse the same result.
  fresh_reinfer: rerun with fresh tactile history, but learned future queries.
  retouch:       rerun with fresh tactile history and previous-prefix warm-start.

The Hindsight/Teacher AE is used only as an offline privileged reference for
latent cosine metrics. It is not used as an input to the Student branch.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
from typing import Literal

import einops
import flax.nnx as nnx
from flax import traverse_util
import jax
from jax import ShapeDtypeStruct
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro

from openpi.models import model_tavla as _model
from openpi.models.pi0_tavla import make_attn_mask
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders


MODES = ("one_shot", "fresh_reinfer", "retouch")


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    pretrained_params: str

    repo_id: str | None = None
    asset_id: str | None = None
    assets_dir: str | None = None
    filter_path: str | None = None

    output_dir: str = "outputs/recursive_revision_analysis"
    batch_size: int = 4
    fsdp_devices: int = 1
    num_workers: int = 0
    max_batches: int = 100
    seed: int = 42
    num_steps: int = 10
    offsets: tuple[int, ...] = (0, 4, 8, 12)

    latent_action_condition: Literal["zero", "gt"] = "zero"


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _path_key(key: tuple[object, ...]) -> str:
    return "/".join(map(str, key))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    flat_shape = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    filtered = {}
    failed = []
    for key, initialized_value in flat_shape.items():
        loaded_value = flat_loaded.get(key)
        reason = None
        if loaded_value is None:
            reason = "missing in checkpoint"
        elif isinstance(loaded_value, ShapeDtypeStruct):
            reason = "is ShapeDtypeStruct"
        elif not hasattr(loaded_value, "shape"):
            reason = "has no shape"
        elif loaded_value.shape != initialized_value.shape:
            reason = f"shape mismatch ckpt={loaded_value.shape}, model={initialized_value.shape}"

        if reason is None:
            filtered[key] = loaded_value
        else:
            filtered[key] = initialized_value
            failed.append((key, reason))

    logging.info("Loaded %d/%d policy params.", len(flat_shape) - len(failed), len(flat_shape))
    if failed:
        logging.warning("Policy checkpoint had %d missing/mismatched params; using initialization.", len(failed))
        for key, reason in failed[:30]:
            logging.warning("  %s -> %s", _path_key(key), reason)
        if len(failed) > 30:
            logging.warning("  ... %d more", len(failed) - 30)
    return traverse_util.unflatten_dict(filtered)


def _override_config(args: Args) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    data = base.data
    if args.repo_id is not None:
        data = dataclasses.replace(data, repo_id=args.repo_id)
    if args.filter_path is not None:
        base_data_config = data.base_config or _config.DataConfig()
        data = dataclasses.replace(
            data,
            base_config=dataclasses.replace(base_data_config, filter_dict_path=args.filter_path),
        )
    if args.asset_id is not None or args.assets_dir is not None:
        assets = dataclasses.replace(
            data.assets,
            asset_id=args.asset_id if args.asset_id is not None else data.assets.asset_id,
            assets_dir=args.assets_dir if args.assets_dir is not None else data.assets.assets_dir,
        )
        data = dataclasses.replace(data, assets=assets)
    return dataclasses.replace(
        base,
        data=data,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.pretrained_params),
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        wandb_enabled=False,
    )


def _validate_config(config: _config.TrainConfig, offsets: tuple[int, ...]) -> None:
    model_config = config.model
    if not getattr(model_config, "structured_tactile", False):
        raise ValueError("Recursive revision analysis requires structured tactile effort.")
    if not getattr(model_config, "cached_vlm_async_ae_enabled", False):
        raise ValueError(
            f"{config.name} does not have cached_vlm_async_ae_enabled=True. "
            "Use an async ReTouch config."
        )
    if int(model_config.action_horizon) <= max(offsets):
        raise ValueError(f"Max offset {max(offsets)} must be smaller than action_horizon={model_config.action_horizon}.")
    if int(model_config.future_steps_per_segment) <= 0:
        raise ValueError("future_steps_per_segment must be positive.")


def _init_frozen_model(config: _config.TrainConfig, rng: at.KeyArrayLike) -> tuple[nnx.GraphDef, nnx.State]:
    model = config.model.create(rng)
    graphdef, params = nnx.split(model)
    partial_params = _load_weights_and_validate(config.weight_loader, params.to_pure_dict())
    params.replace_by_pure_dict(partial_params)
    return graphdef, params


def _preprocess(model, observation: _model.Observation, rng: at.KeyArrayLike) -> _model.Observation:
    original_flow_img = observation.flow_img
    original_wrist_flow_img = observation.wrist_flow_img
    original_future_rgb_img = observation.future_rgb_img
    original_future_wrist_rgb_img = observation.future_wrist_rgb_img
    original_scene_flow = observation.scene_flow
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    return model._restore_aux_images(
        processed,
        original_flow_img,
        original_wrist_flow_img,
        original_future_rgb_img,
        original_future_wrist_rgb_img,
        original_scene_flow,
    )


def _build_branch_attn(
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    branch_tokens: at.Array,
    branch_mask: at.Array,
    branch_ar_mask: at.Array,
) -> tuple[at.Array, at.Array]:
    batch_size = prefix_tokens.shape[0]
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    branch_attn = make_attn_mask(branch_mask, branch_ar_mask)
    branch_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=branch_tokens.shape[1])
    branch_to_prefix = jnp.logical_and(branch_to_prefix, branch_mask[:, :, None])
    prefix_row = jnp.concatenate(
        [prefix_attn, jnp.zeros((batch_size, prefix_tokens.shape[1], branch_tokens.shape[1]), dtype=jnp.bool_)],
        axis=-1,
    )
    branch_row = jnp.concatenate([branch_to_prefix, branch_attn], axis=-1)
    full_attn = jnp.concatenate([prefix_row, branch_row], axis=1)

    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
    branch_positions = prefix_len + jnp.cumsum(branch_mask, axis=-1) - 1
    return full_attn, jnp.concatenate([prefix_positions, branch_positions], axis=1)


def _teacher_future_hidden(
    model,
    processed: _model.Observation,
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    history_effort: at.Array,
    future_effort: at.Array,
    token_actions: at.Array,
    token_time: at.Array,
) -> at.Array:
    branch_tokens, branch_mask, branch_ar_mask, branch_adarms = model.embed_teacher_suffix(
        processed,
        history_effort,
        future_effort,
        token_actions,
        token_time,
    )
    full_attn, positions = _build_branch_attn(
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        branch_tokens,
        branch_mask,
        branch_ar_mask,
    )
    (_, selected_layers), _ = model.PaliGemma.llm(
        [prefix_tokens, None, branch_tokens],
        mask=full_attn,
        positions=positions,
        adarms_cond=[None, None, branch_adarms],
        return_layer_indices=(int(model.distill_layer_indices[-1]),),
    )
    future_force_slice = model._suffix_slices(model.action_horizon)["future_force"]
    return selected_layers[0][2][:, future_force_slice, :].astype(jnp.float32)


def _student_future_hidden(
    model,
    processed: _model.Observation,
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    history_effort: at.Array,
    token_actions: at.Array,
    token_time: at.Array,
    *,
    async_offset: at.Array,
    previous_future_hidden: at.Array | None,
) -> at.Array:
    future_override = (
        model._cached_async_future_query_override(previous_future_hidden, async_offset, token_actions.dtype)
        if previous_future_hidden is not None
        else None
    )
    branch_tokens, branch_mask, branch_ar_mask, branch_adarms, *_ = model.embed_student_suffix(
        processed,
        history_effort,
        token_actions,
        token_time,
        train=False,
        noise_rng=None,
        future_force_query_override=future_override,
        async_offset=async_offset,
    )
    student_out, student_layers = model._forward_student_multilayer(
        prefix_tokens=prefix_tokens,
        prefix_mask=prefix_mask,
        prefix_ar_mask=prefix_ar_mask,
        student_suffix_tokens=branch_tokens,
        student_suffix_mask=branch_mask,
        student_suffix_ar_mask=branch_ar_mask,
        student_adarms=branch_adarms,
    )
    del student_out
    future_force_slice = model._suffix_slices(model.action_horizon)["future_force"]
    return student_layers[-1][:, future_force_slice, :].astype(jnp.float32)


def _project_student_for_cosine(model, student_hidden: at.Array) -> at.Array:
    return model._project_prompt_distill(student_hidden).astype(jnp.float32)


def _latent_cosine(student_hidden: at.Array, teacher_hidden: at.Array, token_mask: at.Array) -> at.Array:
    student = student_hidden / jnp.maximum(jnp.linalg.norm(student_hidden, axis=-1, keepdims=True), 1e-6)
    teacher = teacher_hidden / jnp.maximum(jnp.linalg.norm(teacher_hidden, axis=-1, keepdims=True), 1e-6)
    per_token = jnp.sum(student * teacher, axis=-1)
    mask = token_mask.astype(per_token.dtype)
    return jnp.sum(per_token * mask[None, :]) / jnp.maximum(jnp.sum(mask) * per_token.shape[0], 1.0)


def _future_token_mask(model, offset: at.Array, *, suffix_only: bool) -> at.Array:
    if not suffix_only:
        return jnp.ones((model.future_force_token_count,), dtype=jnp.bool_)
    segment_ids = jnp.repeat(
        jnp.arange(model.future_tactile_segments, dtype=jnp.int32),
        model.tactile_tokens_per_step,
    )
    segment_start = segment_ids * int(model.future_steps_per_segment)
    return segment_start >= jnp.asarray(offset, dtype=jnp.int32)


def _physical_actions(model, actions: at.Array) -> at.Array:
    hand_slice = slice(model.hand_action_start, model.hand_action_start + model.hand_action_dim)
    return jnp.concatenate([actions[..., : model.arm_action_dim], actions[..., hand_slice]], axis=-1)


def _action_mse(model, prediction: at.Array, target: at.Array, offset: at.Array, *, suffix_only: bool) -> at.Array:
    pred = _physical_actions(model, prediction)
    tgt = _physical_actions(model, target)
    sq = jnp.square(pred - tgt)
    if not suffix_only:
        return jnp.mean(sq)
    mask = (
        jnp.arange(model.action_horizon, dtype=jnp.int32)[None, :, None]
        >= jnp.asarray(offset, dtype=jnp.int32)
    ).astype(sq.dtype)
    return jnp.sum(sq * mask) / jnp.maximum(jnp.sum(mask) * sq.shape[0] * sq.shape[-1], 1.0)


def _eval_batch(
    args: Args,
    offsets: tuple[int, ...],
    model_def: nnx.GraphDef,
    model_params: nnx.State,
    batch,
    rng: at.KeyArrayLike,
) -> dict[str, at.Array]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    observation, target_actions = batch
    preprocess_rng, noise_rng = jax.random.split(rng)
    processed = _preprocess(model, observation, preprocess_rng)
    history_effort, future_effort = model._split_effort(processed, require_future=True, dtype=target_actions.dtype)
    if future_effort is None:
        raise ValueError("Recursive revision analysis requires future tactile in observation.effort.")

    batch_size = processed.state.shape[0]
    token_actions = (
        jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=target_actions.dtype)
        if args.latent_action_condition == "zero"
        else target_actions
    )
    token_time = jnp.ones((batch_size,), dtype=target_actions.dtype)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    prefix_kv_cache, _ = model._prefix_kv_cache(prefix_tokens, prefix_mask, prefix_ar_mask)
    action_noise = jax.random.normal(noise_rng, (batch_size, model.action_horizon, model.action_dim), dtype=target_actions.dtype)

    offset0 = jnp.asarray(offsets[0], dtype=jnp.int32)
    if int(offsets[0]) != 0:
        raise ValueError("--offsets must start with 0 for one-shot and recursive baselines.")

    one_hidden0 = _student_future_hidden(
        model,
        processed,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        history_effort,
        token_actions,
        token_time,
        async_offset=offset0,
        previous_future_hidden=None,
    )
    one_actions0, sample_hidden0 = model._cached_vlm_async_denoise_with_prefix_cache(
        rng,
        processed,
        history_effort=history_effort,
        prefix_mask=prefix_mask,
        prefix_kv_cache=prefix_kv_cache,
        async_chunk_offset=offset0,
        prefix_future_hidden=None,
        noise=action_noise,
        num_steps=args.num_steps,
    )

    results: dict[str, at.Array] = {}
    previous_latent_hidden = one_hidden0
    previous_sample_hidden = sample_hidden0

    for offset_index, offset_int in enumerate(offsets):
        offset = jnp.asarray(offset_int, dtype=jnp.int32)
        offset_history = model._history_effort_for_offset(history_effort, future_effort, offset)
        teacher_hidden = _teacher_future_hidden(
            model,
            processed,
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            offset_history,
            future_effort,
            token_actions,
            token_time,
        )
        teacher_hidden = jax.lax.stop_gradient(teacher_hidden)

        if offset_index == 0:
            fresh_hidden = one_hidden0
            retouch_hidden = one_hidden0
            fresh_actions = one_actions0
            retouch_actions = one_actions0
        else:
            fresh_hidden = _student_future_hidden(
                model,
                processed,
                prefix_tokens,
                prefix_mask,
                prefix_ar_mask,
                offset_history,
                token_actions,
                token_time,
                async_offset=offset,
                previous_future_hidden=None,
            )
            retouch_hidden = _student_future_hidden(
                model,
                processed,
                prefix_tokens,
                prefix_mask,
                prefix_ar_mask,
                offset_history,
                token_actions,
                token_time,
                async_offset=offset,
                previous_future_hidden=previous_latent_hidden,
            )
            fresh_actions, _ = model._cached_vlm_async_denoise_with_prefix_cache(
                rng,
                processed,
                history_effort=offset_history,
                prefix_mask=prefix_mask,
                prefix_kv_cache=prefix_kv_cache,
                async_chunk_offset=offset,
                prefix_future_hidden=None,
                noise=action_noise,
                num_steps=args.num_steps,
            )
            retouch_actions, previous_sample_hidden = model._cached_vlm_async_denoise_with_prefix_cache(
                rng,
                processed,
                history_effort=offset_history,
                prefix_mask=prefix_mask,
                prefix_kv_cache=prefix_kv_cache,
                async_chunk_offset=offset,
                prefix_future_hidden=previous_sample_hidden,
                noise=action_noise,
                num_steps=args.num_steps,
            )
            previous_latent_hidden = retouch_hidden

        hidden_by_mode = {
            "one_shot": one_hidden0,
            "fresh_reinfer": fresh_hidden,
            "retouch": retouch_hidden,
        }
        action_by_mode = {
            "one_shot": one_actions0,
            "fresh_reinfer": fresh_actions,
            "retouch": retouch_actions,
        }

        full_token_mask = _future_token_mask(model, offset, suffix_only=False)
        suffix_token_mask = _future_token_mask(model, offset, suffix_only=True)
        for mode in MODES:
            projected_student = _project_student_for_cosine(model, hidden_by_mode[mode])
            results[f"latent_cosine/full/{mode}/{offset_int}"] = _latent_cosine(
                projected_student,
                teacher_hidden,
                full_token_mask,
            )
            results[f"latent_cosine/suffix/{mode}/{offset_int}"] = _latent_cosine(
                projected_student,
                teacher_hidden,
                suffix_token_mask,
            )
            results[f"action_mse/full/{mode}/{offset_int}"] = _action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                suffix_only=False,
            )
            results[f"action_mse/suffix/{mode}/{offset_int}"] = _action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                suffix_only=True,
            )

    return results


def _write_metrics(output_dir: pathlib.Path, rows: list[dict[str, object]], metrics: dict[str, float], args: Args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, ensure_ascii=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (output_dir / "metrics_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "scope", "mode", "offset", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    output_dir: pathlib.Path,
    rows: list[dict[str, object]],
    *,
    metric: str,
    scope: str,
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=180)
    for mode in MODES:
        selected = [row for row in rows if row["metric"] == metric and row["scope"] == scope and row["mode"] == mode]
        selected = sorted(selected, key=lambda row: int(row["offset"]))
        ax.plot(
            [int(row["offset"]) for row in selected],
            [float(row["value"]) for row in selected],
            marker="o",
            linewidth=2,
            label=mode,
        )
    ax.set_xlabel("async offset")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def main(args: Args) -> None:
    init_logging()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    config = _override_config(args)
    _validate_config(config, args.offsets)
    if args.batch_size % jax.process_count() != 0:
        raise ValueError(f"--batch-size must be divisible by jax.process_count()={jax.process_count()}.")

    logging.info("Config: %s", args.config_name)
    logging.info("Policy params: %s", args.pretrained_params)
    logging.info("Dataset: repo=%s asset=%s filter=%s", args.repo_id, args.asset_id, args.filter_path)
    logging.info("Offsets: %s", args.offsets)

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)

    rng = jax.random.PRNGKey(args.seed)
    model_rng, eval_rng = jax.random.split(rng)
    with sharding.set_mesh(mesh):
        model_def, model_params = _init_frozen_model(config, model_rng)
        model_params = jax.device_put(model_params, replicated)

        def peval(model_params_arg, batch_arg, rng_arg):
            return _eval_batch(args, args.offsets, model_def, model_params_arg, batch_arg, rng_arg)

        peval = jax.jit(peval)
        sums: dict[str, float] = {}
        count = 0
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= args.max_batches:
                break
            batch_rng = jax.random.fold_in(eval_rng, batch_index)
            batch_metrics = peval(model_params, batch, batch_rng)
            batch_metrics = jax.device_get(batch_metrics)
            for key, value in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + float(np.asarray(value))
            count += 1
            if count == 1 or count % 10 == 0:
                logging.info("Processed %d/%d batches.", count, args.max_batches)

    if count == 0:
        raise RuntimeError("No batches were evaluated.")

    metrics = {key: value / count for key, value in sorted(sums.items())}
    rows = []
    for key, value in metrics.items():
        metric, scope, mode, offset = key.split("/")
        rows.append(
            {
                "metric": metric,
                "scope": scope,
                "mode": mode,
                "offset": int(offset),
                "value": value,
            }
        )
    _write_metrics(output_dir, rows, metrics, args)
    _plot_metric(
        output_dir,
        rows,
        metric="latent_cosine",
        scope="suffix",
        ylabel="latent cosine (suffix, higher is better)",
        filename="latent_cosine_suffix.png",
    )
    _plot_metric(
        output_dir,
        rows,
        metric="action_mse",
        scope="suffix",
        ylabel="action suffix MSE (lower is better)",
        filename="action_mse_suffix.png",
    )
    logging.info("Saved recursive revision metrics to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
