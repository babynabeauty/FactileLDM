from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import model_tavla as _model_tavla
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

DEBUG_INFER_PRINT_LIMIT = 3


def _summarize_debug_value(value: Any) -> str:
    if isinstance(value, str):
        return f"str={value!r}"
    if isinstance(value, bytes):
        return f"bytes len={len(value)}"
    if not hasattr(value, "shape"):
        return f"{type(value).__name__}={value}"

    array = np.asarray(value)
    summary = f"shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return summary + ", empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            summary += (
                f", min={float(finite.min()):.6g}, max={float(finite.max()):.6g}, "
                f"mean={float(finite.mean()):.6g}"
            )
        else:
            summary += ", no finite values"
    return summary


def _debug_print_tree(title: str, tree: dict, *, infer_index: int) -> None:
    print(f"=== {title} #{infer_index} ===", flush=True)
    flat = flax.traverse_util.flatten_dict(tree, sep="/")
    for key in sorted(flat):
        print(f"[server] {key}: {_summarize_debug_value(flat[key])}", flush=True)

    effort = flat.get("effort")
    if effort is not None:
        effort_array = np.asarray(effort)
        print(f"[server] effort detailed shape={effort_array.shape}", flush=True)
        if effort_array.ndim >= 3 and effort_array.shape[-2:] == (5, 3):
            current = effort_array[-1] if effort_array.ndim == 3 else effort_array.reshape(-1, 5, 3)[-1]
            print(f"[server] latest calc_force [5,3]:\n{np.array2string(current, precision=4)}", flush=True)
        elif effort_array.ndim >= 4 and effort_array.shape[-3:] == (5, 120, 3):
            latest = effort_array[-1] if effort_array.ndim == 4 else effort_array.reshape(-1, 5, 120, 3)[-1]
            magnitude = np.linalg.norm(latest, axis=-1)
            print(
                "[server] latest raw tactile per-finger "
                f"mag_max={np.array2string(magnitude.max(axis=-1), precision=4)}, "
                f"mag_mean={np.array2string(magnitude.mean(axis=-1), precision=4)}",
                flush=True,
            )


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._debug_infer_count = 0

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_trex = (
                nnx_utils.module_jit(model.sample_actions_trex)
                if hasattr(model, "sample_actions_trex")
                else None
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        debug_index = self._debug_infer_count
        trex_mode = obs.get("trex_mode", obs.get("mode"))
        if isinstance(trex_mode, np.ndarray):
            trex_mode = str(trex_mode.item())
        elif trex_mode is not None:
            trex_mode = str(trex_mode)
        trex_chunk_offset = obs.get("trex_chunk_offset", 0)
        trex_chunk_id = obs.get("trex_chunk_id", -1)
        trex_requested = trex_mode in {"slow", "fast", "slow_and_fast"}
        inputs = jax.tree.map(lambda x: x, obs)
        if debug_index < DEBUG_INFER_PRINT_LIMIT:
            _debug_print_tree("SERVER RAW OBS FROM CLIENT", inputs, infer_index=debug_index)
        inputs = self._input_transform(inputs)
        if debug_index < DEBUG_INFER_PRINT_LIMIT:
            _debug_print_tree("SERVER MODEL INPUT AFTER TRANSFORM", inputs, infer_index=debug_index)
        self._debug_infer_count += 1
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_fn = self._sample_actions
        if trex_requested:
            if self._is_pytorch_model:
                raise RuntimeError("T-Rex slow/fast websocket mode is only wired for JAX policies in this server.")
            if self._sample_actions_trex is None:
                # Old configs should not receive fast requests. For slow_and_fast,
                # fall back to the normal one-shot sampler so legacy deployment
                # remains usable if a client accidentally includes the mode field.
                if trex_mode == "fast":
                    raise RuntimeError(
                        "Received mode='fast', but this model/server does not expose sample_actions_trex."
                    )
            else:
                sample_fn = self._sample_actions_trex
                sample_kwargs.update(
                    trex_mode=trex_mode,
                    trex_chunk_offset=jnp.asarray(trex_chunk_offset),
                    trex_chunk_id=jnp.asarray(trex_chunk_id),
                )
        # sample_kwargs.update(debug_query_noise_scale=0.3)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        
        if "effort" in inputs.keys():
            observation = _model_tavla.Observation.from_dict(inputs)
        else:
            observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": sample_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if "effort" in inputs.keys():
            outputs["effort"] = inputs['effort']
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if trex_requested:
            outputs["trex_mode"] = trex_mode
            outputs["trex_chunk_offset"] = np.asarray(trex_chunk_offset)
            outputs["trex_chunk_id"] = np.asarray(trex_chunk_id)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
