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
