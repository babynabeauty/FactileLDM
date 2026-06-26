import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


def _continuous_time_embedding(times_seconds: jax.Array, dim: int) -> jax.Array:
    """Sinusoidal encoding for physical time offsets in seconds."""
    half = dim // 2
    if half == 0:
        return jnp.zeros((*times_seconds.shape, dim), dtype=jnp.float32)
    frequencies = jnp.exp(
        jnp.linspace(jnp.log(1.0), jnp.log(1000.0), half, dtype=jnp.float32)
    )
    angles = times_seconds[..., None].astype(jnp.float32) * (2.0 * math.pi) * frequencies
    embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if embedding.shape[-1] < dim:
        embedding = jnp.pad(embedding, [(0, 0)] * (embedding.ndim - 1) + [(0, dim - embedding.shape[-1])])
    return embedding


class DexterousForceTokenizer(nnx.Module):
    """Tokenizes five-finger forces with explicit content and metadata subspaces."""

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        num_fingers: int,
        dim_per_finger: int,
        future_segments: int,
        future_steps_per_segment: int,
        rngs: nnx.Rngs,
    ):
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_fingers = int(num_fingers)
        self.dim_per_finger = int(dim_per_finger)
        self.future_segments = int(future_segments)
        self.future_steps_per_segment = int(future_steps_per_segment)
        self.finger_embedding_dim = 32
        self.time_embedding_dim = 64
        self.type_embedding_dim = 16
        fusion_input_dim = (
            self.hidden_dim
            + self.finger_embedding_dim
            + self.time_embedding_dim
            + self.type_embedding_dim
        )

        self.force_proj_in = nnx.Linear(self.dim_per_finger, 128, rngs=rngs)
        self.force_proj_hidden = nnx.Linear(128, self.hidden_dim, rngs=rngs)
        self.fusion_proj_in = nnx.Linear(fusion_input_dim, self.hidden_dim, rngs=rngs)
        self.fusion_proj_out = nnx.Linear(self.hidden_dim, self.output_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=self.output_dim, rngs=rngs)
        self.finger_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(), (self.num_fingers, self.finger_embedding_dim), dtype=jnp.float32
            )
        )
        self.type_embedding = nnx.Param(
            0.02
            * jax.random.normal(rngs.params(), (2, self.type_embedding_dim), dtype=jnp.float32)
        )
        self.segment_pool_logits = nnx.Param(
            jnp.zeros((self.future_steps_per_segment,), dtype=jnp.float32)
        )

    def _encode_steps(self, forces: jax.Array, times_seconds: jax.Array, *, future: bool) -> jax.Array:
        if forces.ndim != 4:
            raise ValueError(f"Expected structured force [B,T,F,C], got {forces.shape}.")
        if forces.shape[2:] != (self.num_fingers, self.dim_per_finger):
            raise ValueError(
                f"Expected force finger shape {(self.num_fingers, self.dim_per_finger)}, got {forces.shape[2:]}."
            )
        if times_seconds.shape != (forces.shape[1],):
            raise ValueError(f"Expected {forces.shape[1]} time offsets, got {times_seconds.shape}.")

        force_feature = nnx.swish(self.force_proj_in(forces))
        force_feature = nnx.swish(self.force_proj_hidden(force_feature))
        batch_size, time_steps = forces.shape[:2]

        finger_feature = jnp.broadcast_to(
            self.finger_embedding.value[None, None, :, :],
            (batch_size, time_steps, self.num_fingers, self.finger_embedding_dim),
        )
        time_feature = _continuous_time_embedding(times_seconds, self.time_embedding_dim)
        time_feature = jnp.broadcast_to(
            time_feature[None, :, None, :],
            (batch_size, time_steps, self.num_fingers, self.time_embedding_dim),
        )
        type_feature = jnp.broadcast_to(
            self.type_embedding.value[int(future)][None, None, None, :],
            (batch_size, time_steps, self.num_fingers, self.type_embedding_dim),
        )
        fused = jnp.concatenate(
            [force_feature, finger_feature, time_feature, type_feature],
            axis=-1,
        )
        fused = nnx.swish(self.fusion_proj_in(fused))
        return self.norm(self.fusion_proj_out(fused))

    def encode_history(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        tokens = self._encode_steps(forces, times_seconds, future=False)
        return einops.rearrange(tokens, "b t f d -> b (t f) d")

    def encode_future(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        expected_steps = self.future_segments * self.future_steps_per_segment
        if forces.shape[1] != expected_steps:
            raise ValueError(f"Expected {expected_steps} future force steps, got {forces.shape[1]}.")
        tokens = self._encode_steps(forces, times_seconds, future=True)
        tokens = einops.rearrange(
            tokens,
            "b (s p) f d -> b s p f d",
            s=self.future_segments,
            p=self.future_steps_per_segment,
        )
        weights = jax.nn.softmax(self.segment_pool_logits.value.astype(jnp.float32)).astype(tokens.dtype)
        pooled = jnp.einsum("p,bspfd->bsfd", weights, tokens)
        return einops.rearrange(pooled, "b s f d -> b (s f) d")


class RawTactileSpatialTokenizer(nnx.Module):
    """Tokenizes sparse per-taxel tactile forces into per-finger tokens.

    Input shape is [B, T, F, P, 3], where P is the number of tactile points per
    finger. The tokenizer keeps point identity through a learned point embedding
    and uses contact-aware attention so near-zero taxels contribute less.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        num_fingers: int,
        num_points: int,
        dim_per_point: int,
        future_segments: int,
        future_steps_per_segment: int,
        contact_top_k: int,
        contact_threshold: float,
        contact_temperature: float,
        rngs: nnx.Rngs,
    ):
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_fingers = int(num_fingers)
        self.num_points = int(num_points)
        self.dim_per_point = int(dim_per_point)
        self.future_segments = int(future_segments)
        self.future_steps_per_segment = int(future_steps_per_segment)
        self.contact_top_k = int(contact_top_k)
        self.contact_threshold = float(contact_threshold)
        self.contact_temperature = float(contact_temperature)
        self.time_embedding_dim = 64
        self.tokens_per_tactile_step = self.num_fingers

        self.force_proj_in = nnx.Linear(self.dim_per_point, self.hidden_dim, rngs=rngs)
        self.force_proj_out = nnx.Linear(self.hidden_dim, self.output_dim, rngs=rngs)
        self.time_proj = nnx.Linear(self.time_embedding_dim, self.output_dim, rngs=rngs)
        self.point_score = nnx.Linear(self.output_dim, 1, rngs=rngs)
        self.contact_proj = nnx.Linear(2, self.output_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=self.output_dim, rngs=rngs)

        self.finger_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_fingers, self.output_dim), dtype=jnp.float32)
        )
        self.point_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_points, self.output_dim), dtype=jnp.float32)
        )
        self.type_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (2, self.output_dim), dtype=jnp.float32)
        )
        self.segment_pool_logits = nnx.Param(
            jnp.zeros((self.future_steps_per_segment,), dtype=jnp.float32)
        )

    def _contact_top_k_mask(self, magnitude: jax.Array) -> jax.Array:
        """Select hard top-k contacts without XLA's problematic TopK lowering."""
        selected = jnp.zeros(magnitude.shape, dtype=jnp.bool_)
        remaining = magnitude

        def select_one(_, carry):
            selected_mask, remaining_values = carry
            index = jnp.argmax(remaining_values, axis=-1)
            one_hot = jax.nn.one_hot(index, self.num_points, dtype=jnp.bool_)
            selected_mask = jnp.logical_or(selected_mask, one_hot)
            remaining_values = jnp.where(one_hot, -jnp.inf, remaining_values)
            return selected_mask, remaining_values

        selected, _ = jax.lax.fori_loop(
            0,
            self.contact_top_k,
            select_one,
            (selected, remaining),
        )
        return selected

    def _encode_steps(self, forces: jax.Array, times_seconds: jax.Array, *, future: bool) -> jax.Array:
        if forces.ndim != 5:
            raise ValueError(f"Expected raw tactile force [B,T,F,P,C], got {forces.shape}.")
        if forces.shape[2:] != (self.num_fingers, self.num_points, self.dim_per_point):
            raise ValueError(
                "Expected raw tactile finger/point shape "
                f"{(self.num_fingers, self.num_points, self.dim_per_point)}, got {forces.shape[2:]}."
            )
        if times_seconds.shape != (forces.shape[1],):
            raise ValueError(f"Expected {forces.shape[1]} time offsets, got {times_seconds.shape}.")

        batch_size, time_steps = forces.shape[:2]
        force_feature = nnx.swish(self.force_proj_in(forces))
        force_feature = self.force_proj_out(force_feature)

        finger_feature = jnp.broadcast_to(
            self.finger_embedding.value[None, None, :, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        point_feature = jnp.broadcast_to(
            self.point_embedding.value[None, None, None, :, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        time_feature = self.time_proj(_continuous_time_embedding(times_seconds, self.time_embedding_dim))
        time_feature = jnp.broadcast_to(
            time_feature[None, :, None, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        type_feature = jnp.broadcast_to(
            self.type_embedding.value[int(future)][None, None, None, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )

        point_tokens = force_feature + finger_feature + point_feature + time_feature + type_feature
        magnitude = jnp.linalg.norm(forces.astype(jnp.float32), axis=-1)
        temperature = jnp.asarray(max(self.contact_temperature, 1e-6), dtype=jnp.float32)
        gate = jax.nn.sigmoid((magnitude - self.contact_threshold) / temperature)

        score = jnp.squeeze(self.point_score(nnx.swish(point_tokens)), axis=-1).astype(jnp.float32)
        score = score + jnp.log(gate + 1e-6)
        if 0 < self.contact_top_k < self.num_points:
            contact_mask = self._contact_top_k_mask(magnitude)
            score = jnp.where(contact_mask, score, -1e4)

        weights = jax.nn.softmax(score, axis=-1).astype(point_tokens.dtype)
        pooled = jnp.einsum("btfp,btfpd->btfd", weights, point_tokens)

        contact_stats = jnp.stack(
            [
                jnp.mean(gate, axis=-1),
                jnp.max(magnitude, axis=-1),
            ],
            axis=-1,
        ).astype(point_tokens.dtype)
        pooled = pooled + self.contact_proj(contact_stats)
        return self.norm(pooled)

    def encode_history(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        tokens = self._encode_steps(forces, times_seconds, future=False)
        return einops.rearrange(tokens, "b t f d -> b (t f) d")

    def encode_future(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        expected_steps = self.future_segments * self.future_steps_per_segment
        if forces.shape[1] != expected_steps:
            raise ValueError(f"Expected {expected_steps} future tactile steps, got {forces.shape[1]}.")
        tokens = self._encode_steps(forces, times_seconds, future=True)
        tokens = einops.rearrange(
            tokens,
            "b (s p) f d -> b s p f d",
            s=self.future_segments,
            p=self.future_steps_per_segment,
        )
        weights = jax.nn.softmax(self.segment_pool_logits.value.astype(jnp.float32)).astype(tokens.dtype)
        pooled = jnp.einsum("p,bspfd->bsfd", weights, tokens)
        return einops.rearrange(pooled, "b s f d -> b (s f) d")


class AdaptiveFingertipPatchTokenizer(RawTactileSpatialTokenizer):
    """Raw tactile tokenizer with summary tokens plus local patch tokens.

    Every finger keeps one summary token. Fingers listed in `patch_fingers`
    additionally emit `num_patches` local tokens. The patch map is derived from
    the XHand tactile delivery files: thumb uses T30 coordinates and the other
    fingers use T16 coordinates.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        num_fingers: int,
        num_points: int,
        dim_per_point: int,
        future_segments: int,
        future_steps_per_segment: int,
        contact_top_k: int,
        contact_threshold: float,
        contact_temperature: float,
        patch_fingers: tuple[int, ...],
        num_patches: int,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_fingers=num_fingers,
            num_points=num_points,
            dim_per_point=dim_per_point,
            future_segments=future_segments,
            future_steps_per_segment=future_steps_per_segment,
            contact_top_k=contact_top_k,
            contact_threshold=contact_threshold,
            contact_temperature=contact_temperature,
            rngs=rngs,
        )
        self.patch_fingers = tuple(int(finger) for finger in patch_fingers)
        self.num_patches = int(num_patches)
        self.tokens_per_tactile_step = self.num_fingers + len(self.patch_fingers) * self.num_patches
        self.patch_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_patches, self.output_dim), dtype=jnp.float32)
        )

        if self.num_patches != 5:
            raise ValueError("AdaptiveFingertipPatchTokenizer currently expects num_patches=5.")
        if any(finger < 0 or finger >= self.num_fingers for finger in self.patch_fingers):
            raise ValueError(f"patch_fingers must be in [0, {self.num_fingers}), got {self.patch_fingers}.")

        self.point_patch_ids = self._official_xhand_patch_ids(self.num_fingers, self.num_points)

    @staticmethod
    def _official_xhand_patch_ids(num_fingers: int, num_points: int) -> tuple[tuple[int, ...], ...]:
        if num_points != 120:
            raise ValueError("Official XHand patch map currently supports 120 taxels per finger.")

        # Patch IDs: 0=tip, 1=center, 2=base, 3=left, 4=right.
        # T30 is the thumb sensor. It is split using the delivered transformed
        # right-hand coordinates: distal axis y, lateral axis x.
        t30_thumb = (
            2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0,
            2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0,
            2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0,
            2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0,
            2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0,
            2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 0,
        )
        # T16 is used by index/middle/ring/little. It is split using the
        # delivered transformed coordinates: distal axis z, lateral axis y.
        t16_other = (
            2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0,
            2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0,
            2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0,
            2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0,
            2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
            2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0,
            2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0,
            2, 2, 2, 2, 4, 4, 4, 4, 0, 0, 0, 0,
        )
        if len(t30_thumb) != num_points or len(t16_other) != num_points:
            raise ValueError("Internal XHand patch map has the wrong number of points.")
        if num_fingers < 1:
            raise ValueError("num_fingers must be positive.")
        return (t30_thumb,) + tuple(t16_other for _ in range(num_fingers - 1))

    def _encode_steps(self, forces: jax.Array, times_seconds: jax.Array, *, future: bool) -> jax.Array:
        if forces.ndim != 5:
            raise ValueError(f"Expected raw tactile force [B,T,F,P,C], got {forces.shape}.")
        if forces.shape[2:] != (self.num_fingers, self.num_points, self.dim_per_point):
            raise ValueError(
                "Expected raw tactile finger/point shape "
                f"{(self.num_fingers, self.num_points, self.dim_per_point)}, got {forces.shape[2:]}."
            )
        if times_seconds.shape != (forces.shape[1],):
            raise ValueError(f"Expected {forces.shape[1]} time offsets, got {times_seconds.shape}.")

        batch_size, time_steps = forces.shape[:2]
        force_feature = nnx.swish(self.force_proj_in(forces))
        force_feature = self.force_proj_out(force_feature)

        finger_feature = jnp.broadcast_to(
            self.finger_embedding.value[None, None, :, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        point_feature = jnp.broadcast_to(
            self.point_embedding.value[None, None, None, :, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        time_feature = self.time_proj(_continuous_time_embedding(times_seconds, self.time_embedding_dim))
        time_feature = jnp.broadcast_to(
            time_feature[None, :, None, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )
        type_feature = jnp.broadcast_to(
            self.type_embedding.value[int(future)][None, None, None, None, :],
            (batch_size, time_steps, self.num_fingers, self.num_points, self.output_dim),
        )

        point_tokens = force_feature + finger_feature + point_feature + time_feature + type_feature
        magnitude = jnp.linalg.norm(forces.astype(jnp.float32), axis=-1)
        temperature = jnp.asarray(max(self.contact_temperature, 1e-6), dtype=jnp.float32)
        gate = jax.nn.sigmoid((magnitude - self.contact_threshold) / temperature)

        score = jnp.squeeze(self.point_score(nnx.swish(point_tokens)), axis=-1).astype(jnp.float32)
        score = score + jnp.log(gate + 1e-6)
        if 0 < self.contact_top_k < self.num_points:
            contact_mask = self._contact_top_k_mask(magnitude)
            score = jnp.where(contact_mask, score, -1e4)

        summary_weights = jax.nn.softmax(score, axis=-1).astype(point_tokens.dtype)
        summary_tokens = jnp.einsum("btfp,btfpd->btfd", summary_weights, point_tokens)
        summary_contact_stats = jnp.stack(
            [
                jnp.mean(gate, axis=-1),
                jnp.max(magnitude, axis=-1),
            ],
            axis=-1,
        ).astype(point_tokens.dtype)
        summary_tokens = summary_tokens + self.contact_proj(summary_contact_stats)

        patch_outputs = []
        point_patch_ids = jnp.asarray(self.point_patch_ids, dtype=jnp.int32)
        for finger in self.patch_fingers:
            finger_patch_ids = point_patch_ids[finger]
            for patch_id in range(self.num_patches):
                patch_mask = finger_patch_ids == patch_id
                patch_score = jnp.where(patch_mask[None, None, :], score[:, :, finger, :], -1e9)
                patch_weights = jax.nn.softmax(patch_score, axis=-1).astype(point_tokens.dtype)
                patch_token = jnp.einsum("btp,btpd->btd", patch_weights, point_tokens[:, :, finger, :, :])
                patch_gate = jnp.where(patch_mask[None, None, :], gate[:, :, finger, :], 0.0)
                patch_magnitude = jnp.where(patch_mask[None, None, :], magnitude[:, :, finger, :], 0.0)
                patch_area = jnp.sum(patch_gate, axis=-1) / jnp.maximum(jnp.sum(patch_mask), 1)
                patch_strength = jnp.max(patch_magnitude, axis=-1)
                patch_stats = jnp.stack([patch_area, patch_strength], axis=-1).astype(point_tokens.dtype)
                patch_token = patch_token + self.contact_proj(patch_stats)
                patch_token = patch_token + self.patch_embedding.value[patch_id][None, None, :]
                patch_outputs.append(patch_token)

        if patch_outputs:
            patch_tokens = jnp.stack(patch_outputs, axis=2)
            tokens = jnp.concatenate([summary_tokens, patch_tokens], axis=2)
        else:
            tokens = summary_tokens
        return self.norm(tokens)


class _QFormerBlock(nnx.Module):
    """Small pre-norm query block with self-attention, cross-attention, and FFN."""

    def __init__(self, *, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.head_dim = self.width // self.num_heads

        self.self_q = nnx.Linear(width, width, rngs=rngs)
        self.self_k = nnx.Linear(width, width, rngs=rngs)
        self.self_v = nnx.Linear(width, width, rngs=rngs)
        self.self_out = nnx.Linear(width, width, rngs=rngs)

        self.cross_q = nnx.Linear(width, width, rngs=rngs)
        self.cross_k = nnx.Linear(width, width, rngs=rngs)
        self.cross_v = nnx.Linear(width, width, rngs=rngs)
        self.cross_out = nnx.Linear(width, width, rngs=rngs)

        self.ffn_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, width, rngs=rngs)

        self.self_norm = nnx.LayerNorm(num_features=width, rngs=rngs)
        self.cross_norm = nnx.LayerNorm(num_features=width, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        return einops.rearrange(x, "b n (h d) -> b h n d", h=self.num_heads)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        return einops.rearrange(x, "b h n d -> b n (h d)")

    def _attention(self, query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(float(self.head_dim))
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        return self._merge_heads(attended)

    def __call__(self, queries: jax.Array, memory: jax.Array) -> jax.Array:
        x = self.self_norm(queries)
        queries = queries + self.self_out(
            self._attention(self.self_q(x), self.self_k(x), self.self_v(x))
        )

        x = self.cross_norm(queries)
        queries = queries + self.cross_out(
            self._attention(self.cross_q(x), self.cross_k(memory), self.cross_v(memory))
        )

        x = self.ffn_norm(queries)
        queries = queries + self.ffn_out(nnx.swish(self.ffn_in(x)))
        return queries


class _SelfAttentionFusionBlock(nnx.Module):
    """Pre-norm self-attention block used to fuse tactile tokens before Q-Former."""

    def __init__(self, *, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.head_dim = self.width // self.num_heads
        self.q = nnx.Linear(width, width, rngs=rngs)
        self.k = nnx.Linear(width, width, rngs=rngs)
        self.v = nnx.Linear(width, width, rngs=rngs)
        self.attn_out = nnx.Linear(width, width, rngs=rngs)
        self.ffn_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, width, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

    def _attention(self, x: jax.Array) -> jax.Array:
        query = einops.rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        key = einops.rearrange(self.k(x), "b n (h d) -> b h n d", h=self.num_heads)
        value = einops.rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(float(self.head_dim))
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        return einops.rearrange(attended, "b h n d -> b n (h d)")

    def __call__(self, tokens: jax.Array) -> jax.Array:
        tokens = tokens + self.attn_out(self._attention(self.attn_norm(tokens)))
        tokens = tokens + self.ffn_out(nnx.swish(self.ffn_in(self.ffn_norm(tokens))))
        return tokens


class FingerRoleFutureTactileQFormer(nnx.Module):
    """Extracts action-aware future contact tokens from per-finger future forces.

    The output shape intentionally matches the existing future query layout:
    [B, future_segments * num_fingers, output_dim].
    """

    def __init__(
        self,
        *,
        output_dim: int,
        hidden_dim: int,
        num_fingers: int,
        dim_per_finger: int,
        future_segments: int,
        future_steps_per_segment: int,
        num_layers: int,
        num_heads: int,
        rngs: nnx.Rngs,
        fusion_layers: int = 0,
    ):
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_fingers = int(num_fingers)
        self.dim_per_finger = int(dim_per_finger)
        self.future_segments = int(future_segments)
        self.future_steps_per_segment = int(future_steps_per_segment)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.fusion_layers = int(fusion_layers)
        self.time_embedding_dim = 64
        self.force_proj_in = nnx.Linear(self.dim_per_finger, self.hidden_dim, rngs=rngs)
        self.force_proj_out = nnx.Linear(self.hidden_dim, self.output_dim, rngs=rngs)
        self.time_proj = nnx.Linear(self.time_embedding_dim, self.output_dim, rngs=rngs)
        self.memory_norm = nnx.LayerNorm(num_features=self.output_dim, rngs=rngs)
        self.query_norm = nnx.LayerNorm(num_features=self.output_dim, rngs=rngs)
        self.finger_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_fingers, self.output_dim), dtype=jnp.float32)
        )
        self.query_base = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.output_dim,), dtype=jnp.float32)
        )
        self.query_segment_embedding = nnx.Param(
            0.02
            * jax.random.normal(rngs.params(), (self.future_segments, self.output_dim), dtype=jnp.float32)
        )
        self.query_finger_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_fingers, self.output_dim), dtype=jnp.float32)
        )
        self.fusion_blocks = [
            _SelfAttentionFusionBlock(
                width=self.output_dim,
                num_heads=self.num_heads,
                mlp_dim=max(self.output_dim * 4, self.hidden_dim),
                rngs=rngs,
            )
            for _ in range(self.fusion_layers)
        ]
        self.blocks = [
            _QFormerBlock(
                width=self.output_dim,
                num_heads=self.num_heads,
                mlp_dim=max(self.output_dim * 4, self.hidden_dim),
                rngs=rngs,
            )
            for _ in range(self.num_layers)
        ]

    def _memory_tokens(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        if forces.ndim != 4:
            raise ValueError(f"Expected future force [B,T,F,C], got {forces.shape}.")
        if forces.shape[2:] != (self.num_fingers, self.dim_per_finger):
            raise ValueError(
                f"Expected force finger shape {(self.num_fingers, self.dim_per_finger)}, got {forces.shape[2:]}."
            )
        if times_seconds.shape != (forces.shape[1],):
            raise ValueError(f"Expected {forces.shape[1]} time offsets, got {times_seconds.shape}.")

        force_feature = nnx.swish(self.force_proj_in(forces))
        force_feature = self.force_proj_out(force_feature)
        batch_size, time_steps = forces.shape[:2]
        finger_feature = jnp.broadcast_to(
            self.finger_embedding.value[None, None, :, :],
            (batch_size, time_steps, self.num_fingers, self.output_dim),
        )
        time_feature = self.time_proj(_continuous_time_embedding(times_seconds, self.time_embedding_dim))
        time_feature = jnp.broadcast_to(
            time_feature[None, :, None, :],
            (batch_size, time_steps, self.num_fingers, self.output_dim),
        )
        tokens = self.memory_norm(force_feature + finger_feature + time_feature)
        return einops.rearrange(tokens, "b t f d -> b (t f) d")

    def _queries(self, batch_size: int, dtype) -> jax.Array:
        queries = (
            self.query_base.value[None, None, :]
            + self.query_segment_embedding.value[:, None, :]
            + self.query_finger_embedding.value[None, :, :]
        )
        queries = einops.rearrange(queries, "s f d -> (s f) d").astype(dtype)
        return jnp.broadcast_to(queries[None], (batch_size, *queries.shape))

    def encode_future(self, forces: jax.Array, times_seconds: jax.Array) -> jax.Array:
        expected_steps = self.future_segments * self.future_steps_per_segment
        if forces.shape[1] != expected_steps:
            raise ValueError(f"Expected {expected_steps} future force steps, got {forces.shape[1]}.")
        memory = self._memory_tokens(forces, times_seconds)
        for block in self.fusion_blocks:
            memory = block(memory)
        queries = self._queries(forces.shape[0], memory.dtype)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.query_norm(queries)
