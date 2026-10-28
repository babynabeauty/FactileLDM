"""Exercise the no-HAE training and async inference paths with a small CPU model."""

import dataclasses
import importlib
import sys
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model_tavla
from openpi.models import siglip
from openpi.training import config as training_config
from scripts.extract_future_patch_probe_real_model_test import TinyVision


def test_no_hae_config_preserves_main_policy_settings():
    main = training_config.get_config("pi0_xhand_dual_patch_pretrained_f4_h16_async")
    no_hae = training_config.get_config("pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae")
    freeze = training_config.get_config("pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae_freeze")
    allowed = {
        "disable_teacher_ae", "teacher_action_loss_weight", "future_force_align_loss_weight",
        "future_flow_align_loss_weight", "cached_vlm_async_future_align_loss_weight",
    }
    for field in dataclasses.fields(main.model):
        if field.name not in allowed:
            assert getattr(no_hae.model, field.name) == getattr(main.model, field.name), field.name
    assert no_hae.model == freeze.model
    assert no_hae.model.disable_teacher_ae
    assert no_hae.model.cached_vlm_async_use_predicted_prefix_queries
    assert no_hae.model.cached_vlm_async_offsets == (0, 4, 8, 12)
    with pytest.raises(ValueError, match="loss weights"):
        dataclasses.replace(main.model, disable_teacher_ae=True)


def test_no_hae_checkpoint_policy_keeps_all_and_rejects_overwrite(monkeypatch):
    # Avoid a real checkpoint directory or any filesystem mutation.
    monkeypatch.setitem(sys.modules, "openpi.training.data_loader", types.ModuleType("openpi.training.data_loader"))
    checkpoints = importlib.import_module("openpi.training.checkpoints")
    with pytest.raises(ValueError, match="overwrite"):
        checkpoints.initialize_checkpoint_dir(
            "unused", keep_period=1, overwrite=True, resume=False, preserve_all_checkpoints=True
        )

    class FakePath:
        def __init__(self, *_):
            pass

        def resolve(self):
            return self

        def exists(self):
            return False

        def mkdir(self, **_kwargs):
            pass

    captured = {}

    def manager(_path, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(checkpoints.epath, "Path", FakePath)
    monkeypatch.setattr(checkpoints.ocp, "CheckpointManager", manager)
    checkpoints.initialize_checkpoint_dir(
        "unused", keep_period=10000, overwrite=False, resume=False, preserve_all_checkpoints=True
    )
    assert captured["options"].max_to_keep is None


def test_no_hae_training_and_recursive_inference(monkeypatch):
    monkeypatch.setattr(siglip, "Module", TinyVision)
    model_config = dataclasses.replace(
        training_config.get_config("pi0_xhand_dual_patch_pretrained_f4_h16_async_no_hae").model,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        future_tactile_align_layer=1,
        dtype="float32",
    )
    policy = model_config.create(jax.random.key(0))
    assert not policy.use_teacher_ae
    assert policy.future_force_token_count == 20
    image_keys = model_tavla.IMAGE_KEYS
    observation = model_tavla.Observation(
        images={key: jnp.zeros((1, 224, 224, 3), jnp.float32) for key in image_keys},
        image_masks={key: jnp.ones((1,), bool) for key in image_keys},
        state=jnp.zeros((1, model_config.action_dim)),
        effort=jnp.ones((1, 26, 5, 120, 3)),
        tokenized_prompt=jnp.ones((1, 4), jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 4), bool),
    )
    actions = jnp.zeros((1, 16, model_config.action_dim))
    loss, stats = policy.compute_loss_with_stats(jax.random.key(1), observation, actions, train=False)
    assert np.isfinite(np.asarray(loss)).all()
    for name in ("loss/teacher_action", "loss/distill_future_force", "loss/cached_async_future_align"):
        np.testing.assert_array_equal(np.asarray(stats[name]), 0)
    assert np.isfinite(np.asarray(stats["loss/student_action"])).all()
    assert np.isfinite(np.asarray(stats["loss/cached_async_action"])).all()
    graph, state = nnx.split(policy)

    def objective(current_state):
        model = nnx.merge(graph, current_state)
        value, _ = model.compute_loss_with_stats(jax.random.key(1), observation, actions, train=False)
        return jnp.mean(value)

    gradients = jax.jit(jax.grad(objective))(state).to_pure_dict()
    query_gradient = np.asarray(gradients["student_query_base"])
    assert np.isfinite(query_gradient).all()
    assert np.linalg.norm(query_gradient) > 0
    with pytest.raises(ValueError, match="merged history\\+future effort"):
        policy.compute_loss_with_stats(
            jax.random.key(1), dataclasses.replace(observation, effort=observation.effort[:, :10]), actions
        )

    history_obs = dataclasses.replace(observation, effort=observation.effort[:, :10])
    noise = jnp.zeros_like(actions)
    _, cache, mask, hidden = policy.sample_actions_cached_vlm_async_ae_slow(
        jax.random.key(2), history_obs, noise=noise, num_steps=1
    )
    query = policy._cached_async_future_query_override(hidden, 4, hidden.dtype)  # noqa: SLF001
    np.testing.assert_allclose(np.asarray(query[:, :5]), np.asarray(hidden[:, :5]))
    np.testing.assert_allclose(
        np.asarray(query[:, 5:]), np.asarray(policy._student_query_token(1, hidden.dtype)[:, 5:])  # noqa: SLF001
    )
    refined_actions, refined_hidden = policy.sample_actions_cached_vlm_async_ae_fast(
        jax.random.key(2), history_obs, prefix_kv_cache=cache, prefix_mask=mask,
        prefix_future_hidden=hidden, async_chunk_offset=4,
        async_fresh_effort=jnp.full_like(history_obs.effort, 2.0), noise=noise, num_steps=1,
    )
    assert refined_actions.shape == (1, 16, model_config.action_dim)
    assert refined_hidden.shape == (1, 20, policy.student_width)
    assert np.isfinite(np.asarray(refined_hidden)).all()
