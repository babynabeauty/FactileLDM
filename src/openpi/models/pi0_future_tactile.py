import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
from openpi.models.tactile_tokenizer import DexterousForceTokenizer
from openpi.models.tactile_tokenizer import FingerRoleFutureTactileQFormer


class Pi0FutureTactile(Pi0):
    """Single action expert with structured future tactile queries."""

    def __init__(self, config: pi0_config.Pi0FutureTactileConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        action_width = self.action_in_proj.out_features
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.dim_per_finger = int(config.tactile_dim_per_finger)
        self.future_segments = int(config.future_tactile_segments)
        self.steps_per_segment = int(config.future_steps_per_segment)
        self.history_token_count = self.force_input_frames * self.num_fingers
        self.future_token_count = self.future_segments * self.num_fingers
        self.align_layer = int(config.future_tactile_align_layer)
        self.latent_loss_weight = float(config.future_tactile_latent_loss_weight)
        self.force_loss_weight = float(config.future_force_loss_weight)
        self.delta_loss_weight = float(config.future_force_delta_loss_weight)
        self.future_encoder_type = config.future_tactile_encoder_type
        self.future_hand_action_loss_weight = float(config.future_hand_action_loss_weight)
        self.hand_action_start = int(config.hand_action_start)
        self.hand_action_dim = int(config.hand_action_dim)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.force_tokenizer = DexterousForceTokenizer(
            output_dim=action_width,
            hidden_dim=config.tactile_tokenizer_dim,
            num_fingers=self.num_fingers,
            dim_per_finger=self.dim_per_finger,
            future_segments=self.future_segments,
            future_steps_per_segment=self.steps_per_segment,
            rngs=rngs,
        )
        if self.future_encoder_type == "dexterous":
            self.target_force_tokenizer = DexterousForceTokenizer(
                output_dim=action_width,
                hidden_dim=config.tactile_tokenizer_dim,
                num_fingers=self.num_fingers,
                dim_per_finger=self.dim_per_finger,
                future_segments=self.future_segments,
                future_steps_per_segment=self.steps_per_segment,
                rngs=rngs,
            )
            nnx.update(self.target_force_tokenizer, nnx.state(self.force_tokenizer, nnx.Param))
            self.use_target_ema = True
        elif self.future_encoder_type == "finger_role":
            self.target_force_tokenizer = FingerRoleFutureTactileQFormer(
                output_dim=action_width,
                hidden_dim=config.tactile_tokenizer_dim,
                num_fingers=self.num_fingers,
                dim_per_finger=self.dim_per_finger,
                future_segments=self.future_segments,
                future_steps_per_segment=self.steps_per_segment,
                num_layers=config.future_tactile_encoder_depth,
                num_heads=config.future_tactile_encoder_num_heads,
                rngs=rngs,
            )
            self.use_target_ema = False
        else:
            raise ValueError(f"Unsupported future_tactile_encoder_type={self.future_encoder_type!r}.")
        self.future_query_base = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (action_width,), dtype=jnp.float32)
        )
        self.future_query_segment_embedding = nnx.Param(
            0.02
            * jax.random.normal(rngs.params(), (self.future_segments, action_width), dtype=jnp.float32)
        )
        self.future_query_finger_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_fingers, action_width), dtype=jnp.float32)
        )
        self.future_force_decoder = nnx.Linear(
            action_width, self.steps_per_segment * self.dim_per_finger, rngs=rngs
        )
        if self.future_hand_action_loss_weight > 0.0:
            self.future_hand_condition_norm = nnx.LayerNorm(num_features=action_width, rngs=rngs)
            self.future_hand_head_in = nnx.Linear(action_width, action_width, rngs=rngs)
            self.future_hand_head_out = nnx.Linear(action_width, self.hand_action_dim, rngs=rngs)
        self.uses_train_progress = True

    def update_target_encoder(self, train_progress) -> None:
        if not self.use_target_ema:
            return
        decay = 0.996 + 0.003 * jnp.clip(jnp.asarray(train_progress, dtype=jnp.float32), 0.0, 1.0)
        online_state = nnx.state(self.force_tokenizer, nnx.Param)
        target_state = nnx.state(self.target_force_tokenizer, nnx.Param)
        updated_state = jax.tree.map(
            lambda target, online: decay * target + (1.0 - decay) * online,
            target_state,
            online_state,
        )
        nnx.update(self.target_force_tokenizer, updated_state)

    def _split_effort(self, observation: _model.Observation, *, require_future: bool, dtype):
        if observation.effort is None:
            raise ValueError("Structured future tactile model requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (self.force_input_frames + self.action_horizon, self.num_fingers, self.dim_per_finger)
        if effort.ndim != 4 or effort.shape[1:] != expected:
            if not require_future and effort.ndim == 4 and effort.shape[1:] == (
                self.force_input_frames,
                self.num_fingers,
                self.dim_per_finger,
            ):
                return effort, None
            raise ValueError(f"Expected structured effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _future_queries(self, batch_size: int, dtype) -> jax.Array:
        query = (
            self.future_query_base.value[None, None, :]
            + self.future_query_segment_embedding.value[:, None, :]
            + self.future_query_finger_embedding.value[None, :, :]
        )
        query = einops.rearrange(query, "s f d -> (s f) d").astype(dtype)
        return jnp.broadcast_to(query[None], (batch_size, *query.shape))

    def embed_prefix(self, obs):
        tokens = []
        input_mask = []
        ar_mask = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]
        if obs.tokenized_prompt is not None:
            prompt_tokens = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(prompt_tokens)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * prompt_tokens.shape[1]
        return (
            jnp.concatenate(tokens, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.asarray(ar_mask),
        )

    def embed_structured_suffix(self, obs, history_effort, noisy_actions, timestep):
        state_token = self.state_proj(obs.state)[:, None, :]
        history_tokens = self.force_tokenizer.encode_history(
            history_effort, jnp.asarray(self.history_times, dtype=jnp.float32)
        )
        future_queries = self._future_queries(obs.state.shape[0], history_tokens.dtype)

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_tokens = einops.repeat(time_emb, "b d -> b s d", s=self.action_horizon)
        action_tokens = self.action_time_mlp_in(jnp.concatenate([action_tokens, time_tokens], axis=-1))
        action_tokens = self.action_time_mlp_out(nnx.swish(action_tokens))

        tokens = jnp.concatenate([state_token, history_tokens, future_queries, action_tokens], axis=1)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        obs_count = 1 + self.history_token_count
        ar_mask = jnp.array(
            ([True] + [False] * (obs_count - 1))
            + ([True] + [False] * (self.future_token_count - 1))
            + ([True] + [False] * (self.action_horizon - 1))
        )
        return tokens, input_mask, ar_mask

    def _decode_future_force(self, query_hidden: jax.Array) -> jax.Array:
        force = self.future_force_decoder(query_hidden)
        force = einops.rearrange(
            force,
            "b (s f) (p c) -> b s p f c",
            s=self.future_segments,
            f=self.num_fingers,
            p=self.steps_per_segment,
            c=self.dim_per_finger,
        )
        return einops.rearrange(force, "b s p f c -> b (s p) f c")

    def _predict_future_hand_action(
        self,
        c_future: jax.Array,
        observation: _model.Observation,
        history_effort: jax.Array,
    ) -> jax.Array:
        history_tokens = self.force_tokenizer.encode_history(
            history_effort, jnp.asarray(self.history_times, dtype=jnp.float32)
        )
        condition = self.future_hand_condition_norm(
            self.state_proj(observation.state) + jnp.mean(history_tokens, axis=1)
        )
        segment_tokens = einops.rearrange(
            c_future,
            "b (s f) d -> b s f d",
            s=self.future_segments,
            f=self.num_fingers,
        )
        segment_tokens = jnp.mean(segment_tokens, axis=2) + condition[:, None, :]
        step_tokens = jnp.repeat(segment_tokens, repeats=self.steps_per_segment, axis=1)
        hidden = nnx.swish(self.future_hand_head_in(step_tokens))
        return self.future_hand_head_out(hidden)

    @staticmethod
    def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
        error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
        loss = jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)
        return jnp.mean(loss, axis=tuple(range(1, loss.ndim)))

    @staticmethod
    def _cosine_distance(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs /= jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs /= jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        return jnp.mean(1.0 - jnp.sum(lhs * rhs, axis=-1), axis=-1)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False, train_progress=None):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, effort_type=self.effort_type
        )
        history_effort, future_effort = self._split_effort(observation, require_future=True, dtype=actions.dtype)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_structured_suffix(
            observation, history_effort, x_t, time
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (outputs, selected_layers), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=make_attn_mask(input_mask, ar_mask),
            positions=positions,
            adarms_cond=[None, None],
            return_layer_indices=(self.align_layer,),
        )
        suffix_out = outputs[1]
        selected_suffix = selected_layers[0][1]
        query_start = 1 + self.history_token_count
        query_slice = slice(query_start, query_start + self.future_token_count)
        query_hidden = selected_suffix[:, query_slice]
        final_query_hidden = suffix_out[:, query_slice]

        c_future = self.target_force_tokenizer.encode_future(
            future_effort, jnp.asarray(self.future_times, dtype=jnp.float32)
        )
        target_tokens = jax.lax.stop_gradient(c_future)
        latent_loss = self._cosine_distance(query_hidden, target_tokens)
        predicted_force = self._decode_future_force(final_query_hidden)
        force_loss = self._smooth_l1(predicted_force, future_effort)
        delta_loss = self._smooth_l1(
            predicted_force[:, 1:] - predicted_force[:, :-1],
            future_effort[:, 1:] - future_effort[:, :-1],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(velocity - u_t), axis=(-2, -1))
        arm_action_loss = jnp.mean(jnp.square(velocity[:, :, : self.hand_action_start] - u_t[:, :, : self.hand_action_start]), axis=(-2, -1))
        hand_action_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        hand_action_loss = jnp.mean(jnp.square(velocity[:, :, hand_action_slice] - u_t[:, :, hand_action_slice]), axis=(-2, -1))

        future_hand_loss = jnp.zeros_like(action_loss)
        if self.future_hand_action_loss_weight > 0.0:
            predicted_hand_action = self._predict_future_hand_action(c_future, observation, history_effort)
            target_hand_action = actions[:, :, hand_action_slice]
            future_hand_loss = self._smooth_l1(predicted_hand_action, target_hand_action)

        total = (
            action_loss
            + self.latent_loss_weight * latent_loss
            + self.force_loss_weight * force_loss
            + self.delta_loss_weight * delta_loss
            + self.future_hand_action_loss_weight * future_hand_loss
        )
        return total, {
            "loss/action": action_loss,
            "loss/action_arm": arm_action_loss,
            "loss/action_hand": hand_action_loss,
            "loss/future_tactile_latent": latent_loss,
            "loss/future_hand_action": future_hand_loss,
            "loss/future_force": force_loss,
            "loss/future_force_delta": delta_loss,
            "loss/total": total,
        }

    def compute_loss(self, rng, observation, actions, *, train=False, train_progress=None):
        loss, _ = self.compute_loss_with_stats(
            rng, observation, actions, train=train, train_progress=train_progress
        )
        return loss

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=make_attn_mask(prefix_mask, prefix_ar_mask),
            positions=prefix_positions,
        )
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_structured_suffix(
                observation, history_effort, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_mask_full = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            outputs, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=jnp.concatenate([prefix_to_suffix, suffix_mask_full], axis=-1),
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None],
            )
            velocity = self.action_out_proj(outputs[1][:, -self.action_horizon :])
            return x_t + dt * velocity, time + dt

        return jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2, step, (noise, 1.0))[0]
