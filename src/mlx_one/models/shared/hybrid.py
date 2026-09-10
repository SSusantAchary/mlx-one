"""Shared Liquid Foundation Model hybrid decoder primitives."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.transformer import apply_rope


def sliding_causal_mask(length: int, offset: int, window: int) -> Any:
    if window < 1:
        raise ValueError("sliding window must be positive")
    query = mx.arange(offset, offset + length)[:, None]
    key = mx.arange(offset + length)[None, :]
    allowed = (key <= query) & (key > query - window)
    return mx.where(allowed, 0.0, -float("inf"))


class LfmAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=False)
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def __call__(self, hidden: Any, *, mask: Any | None, cache: Any | None) -> Any:
        batch, length, _ = hidden.shape
        queries = self.q_proj(hidden).reshape(batch, length, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden).reshape(batch, length, self.num_kv_heads, self.head_dim)
        values = self.v_proj(hidden).reshape(batch, length, self.num_kv_heads, self.head_dim)
        queries = self.q_layernorm(queries).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        queries = apply_rope(queries, offset=offset, theta=self.rope_theta)
        keys = apply_rope(keys, offset=offset, theta=self.rope_theta)
        if cache is not None:
            keys, values = cache.update(keys, values)
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        output = mx.softmax(scores, axis=-1, precise=True) @ values
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(output)


class ShortConv(nn.Module):
    """Double-gated causal depthwise convolution used by LFM2."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_L_cache
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.conv_bias)
        scale = config.hidden_size**-0.5
        self.conv = mx.random.uniform(
            low=-scale, high=scale, shape=(config.hidden_size, config.conv_L_cache)
        )
        self.conv_bias = mx.zeros((config.hidden_size,)) if config.conv_bias else None
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.conv_bias)

    def __call__(self, hidden: Any, *, padding_mask: Any | None, cache: Any | None) -> Any:
        left_gate, right_gate, value = mx.split(self.in_proj(hidden), 3, axis=-1)
        gated = left_gate * value
        if padding_mask is not None:
            gated = mx.where(padding_mask[..., None], gated, 0)
        if cache is None:
            combined = mx.pad(gated, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        else:
            combined = cache.update(gated)
        windows = mx.stack(
            [combined[:, index : index + hidden.shape[1], :] for index in range(self.kernel_size)],
            axis=2,
        )
        convolved = mx.einsum("blkd,dk->bld", windows, self.conv)
        if self.conv_bias is not None:
            convolved = convolved + self.conv_bias
        return self.out_proj(right_gate * convolved)


class LfmMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, value: Any) -> Any:
        gate = self.w1(value)
        return self.w2((gate * mx.sigmoid(gate)) * self.w3(value))


class HybridDecoderLayer(nn.Module):
    def __init__(self, config: Any, layer_index: int, feed_forward: Any | None = None) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_index]
        if self.layer_type in {"full_attention", "sliding_attention"}:
            self.self_attn = LfmAttention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = feed_forward or LfmMLP(
            config.hidden_size, config.adjusted_intermediate_size
        )
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    @property
    def is_attention_layer(self) -> bool:
        return self.layer_type in {"full_attention", "sliding_attention"}

    def __call__(
        self,
        hidden: Any,
        *,
        attention_mask: Any | None,
        padding_mask: Any | None,
        cache: Any | None,
    ) -> Any:
        normalized = self.operator_norm(hidden)
        if self.is_attention_layer:
            operated = self.self_attn(normalized, mask=attention_mask, cache=cache)
        else:
            operated = self.conv(normalized, padding_mask=padding_mask, cache=cache)
        hidden = hidden + operated
        return hidden + self.feed_forward(self.ffn_norm(hidden))
