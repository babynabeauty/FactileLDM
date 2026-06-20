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
    ):
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_fingers = int(num_fingers)
        self.dim_per_finger = int(dim_per_finger)
        self.future_segments = int(future_segments)
        self.future_steps_per_segment = int(future_steps_per_segment)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
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
        queries = self._queries(forces.shape[0], memory.dtype)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.query_norm(queries)
