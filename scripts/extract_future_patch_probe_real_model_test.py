"""Run actual FAE/checkpoint code with small substitute vision/Gemma on CPU.

This does not load the user's checkpoint, real SigLIP backbone, or robot data.
"""

import builtins
import sys

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from openpi.models import model
from openpi.models import model_tavla
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.shared.effort_type import EffortType
from openpi.shared.future_patch_probe_utils import check_parameter_tree
from scripts.extract_future_patch_probe import extract_probe_batch


class TinyVision(nn.Module):
    num_classes: int
    variant: str
    pool_type: str
    scan: bool
    dtype_mm: str

    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        token = nn.Dense(self.num_classes)(jnp.mean(image, axis=(1, 2)))
        return token[:, None, :], None


def test_real_fae_checkpoint_and_four_offset_inference_without_diffusers(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_diffusers(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ModuleNotFoundError("diffusers deliberately unavailable", name="diffusers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_diffusers)
    monkeypatch.delitem(sys.modules, "openpi.models.pi0_latent_flow", raising=False)
    monkeypatch.setattr(siglip, "Module", TinyVision)
    config = pi0_config.Pi0LatentFlowConfig(
        action_horizon=16,
        action_dim=32,
        structured_tactile=True,
        effort_type=EffortType.MOT,
        effort_dim=1800,
        tactile_points_per_finger=120,
        tactile_dim_per_finger=3,
        tactile_num_fingers=5,
        force_input_frames=10,
        future_tactile_segments=4,
        future_steps_per_segment=4,
        pool_tactile_history=True,
        cached_vlm_async_ae_enabled=True,
        cached_vlm_async_history_mode="pooled_current",
        tactile_patch_informed_tokenizer=True,
        use_future_flow=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        future_tactile_align_layer=1,
        dtype="float32",
    )
    shaped = nnx.eval_shape(config.create, jax.random.key(0))
    graph, state = nnx.split(shaped)
    instance = config.create(jax.random.key(1))
    _, original = nnx.split(instance)
    params_path = tmp_path / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_path, {"params": original})
    loaded = model.restore_params(params_path, restore_type=np.ndarray)
    check_parameter_tree(state.to_pure_dict(), loaded)
    state.replace_by_pure_dict(loaded)
    policy = nnx.merge(graph, state)
    policy.eval()
    graph, state = nnx.split(policy)

    images = {key: jnp.zeros((1, 224, 224, 3), jnp.float32) for key in model_tavla.IMAGE_KEYS}
    observation = model_tavla.Observation(
        images=images,
        image_masks={key: jnp.ones((1,), bool) for key in images},
        state=jnp.zeros((1, 32)),
        effort=jnp.ones((1, 26, 5, 120, 3)),
        tokenized_prompt=jnp.ones((1, 4), jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 4), bool),
    )

    def forward(policy_state, obs):
        return extract_probe_batch(
            nnx.merge(graph, policy_state),
            jax.random.key(42),
            obs,
            jnp.ones((1, 16), bool),
            num_steps=10,
            label_effort=obs.effort,
            label_contact_threshold=1.0,
            label_contact_temperature=0.5,
        )

    output = jax.device_get(jax.jit(forward)(state, observation))
    assert output["refined_hidden"].shape == (1, 4, 20, 64)
    assert output["target_contact"].shape == (1, 16, 5, 5)
    np.testing.assert_array_equal(output["valid_mask"].sum(axis=-1), [[16, 12, 8, 4]])
    np.testing.assert_array_equal(output["fixed_hidden"][:, 0], output["refined_hidden"][:, 0])

    # Configurations that actually request the VAE still require diffusers.
    from openpi.models.pi0_latent_flow import preload_flow_vae  # noqa: PLC0415

    with pytest.raises(ModuleNotFoundError, match="diffusers deliberately unavailable"):
        preload_flow_vae("not-cached-test-vae")
