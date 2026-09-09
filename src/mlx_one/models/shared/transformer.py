"""Shared decoder-only Transformer layers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def causal_mask(length: int, offset: int = 0) -> Any | None:
    if length <= 1 and offset > 0:
        return None
    return mx.triu(mx.full((length, offset + length), -float("inf")), k=offset + 1)


def _rotate_half(value: Any) -> Any:
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def apply_rope(
    value: Any,
    *,
    offset: int,
    theta: float,
    position_ids: Any | None = None,
    mrope_section: tuple[int, int, int] | None = None,
    scaling_factor: float = 1.0,
) -> Any:
    batch, _, length, head_dim = value.shape
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    half = head_dim // 2
    inv_freq = mx.power(theta, -mx.arange(half, dtype=mx.float32) / half)
    if position_ids is None:
        positions = mx.arange(offset, offset + length, dtype=mx.float32)
        positions = mx.broadcast_to(positions[None, :], (batch, length))
        frequencies = (positions[..., None] / scaling_factor) * inv_freq
    elif position_ids.ndim == 2:
        frequencies = (
            position_ids.astype(mx.float32)[..., None] / scaling_factor
        ) * inv_freq
    elif position_ids.ndim == 3:
        if position_ids.shape[0] != 3 or mrope_section is None:
            raise ValueError("3D position ids require three mrope_section values")
        if sum(mrope_section) != half:
            raise ValueError("mrope_section values must sum to half the head dimension")
        parts = []
        start = 0
        for axis, width in enumerate(mrope_section):
            stop = start + width
            positions = position_ids[axis].astype(mx.float32)
            parts.append(positions[..., None] * inv_freq[start:stop])
            start = stop
        frequencies = mx.concatenate(parts, axis=-1)
    else:
        raise ValueError("position_ids must have shape [batch, length] or [3, batch, length]")
    frequencies = mx.concatenate((frequencies, frequencies), axis=-1)[:, None, :, :]
    return value * mx.cos(frequencies) + _rotate_half(value) * mx.sin(frequencies)


def _repeat_kv(value: Any, repeats: int) -> Any:
    return value if repeats == 1 else mx.repeat(value, repeats, axis=1)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        rope_theta: float,
        qkv_bias: bool,
        qk_norm: bool,
        norm_eps: float,
        mrope_section: tuple[int, int, int] | None = None,
        rope_scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.rope_scaling_factor = rope_scaling_factor
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim, eps=norm_eps) if qk_norm else None

    def __call__(
        self,
        hidden_states: Any,
        *,
        mask: Any | None = None,
        cache: Any | None = None,
        position_ids: Any | None = None,
    ) -> Any:
        batch, length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        queries = apply_rope(
            queries,
            offset=offset,
            theta=self.rope_theta,
            position_ids=position_ids,
            mrope_section=self.mrope_section,
            scaling_factor=self.rope_scaling_factor,
        )
        keys = apply_rope(
            keys,
            offset=offset,
            theta=self.rope_theta,
            position_ids=position_ids,
            mrope_section=self.mrope_section,
            scaling_factor=self.rope_scaling_factor,
        )
        if cache is not None:
            keys, values = cache.update(keys, values)
        repeats = self.num_heads // self.num_kv_heads
        keys = _repeat_kv(keys, repeats)
        values = _repeat_kv(values, repeats)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        output = probabilities @ values
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, value: Any) -> Any:
        gate = self.gate_proj(value)
        return self.down_proj((gate * mx.sigmoid(gate)) * self.up_proj(value))


class DecoderLayer(nn.Module):
    def __init__(self, config: Any, *, qkv_bias: bool, qk_norm: bool = False) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_eps=config.rms_norm_eps,
            mrope_section=getattr(config, "mrope_section", None),
            rope_scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        position_ids: Any | None = None,
    ) -> Any:
        attended = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attended
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


def quick_gelu(value: Any) -> Any:
    return value * mx.sigmoid(1.702 * value)
