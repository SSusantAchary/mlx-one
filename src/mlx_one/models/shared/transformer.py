"""Shared decoder-only Transformer layers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.execution import project
from mlx_one.ops.attention import attention, select_attention
from mlx_one.ops.numerical_trace import emit
from mlx_one.ops.profiling import phase


def causal_mask(length: int, offset: int = 0, *, optimized: bool = False) -> Any | None:
    if length <= 1 and offset > 0:
        return None
    if optimized:
        from mlx_one.engine.optimization_policy import current_math

        if current_math().policy.attention == "mlx_sdpa":
            return "causal"
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
    rope_scaling: Any | None = None,
    native_precision: bool = True,
) -> Any:
    batch, _, length, head_dim = value.shape
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    kind = (rope_scaling or {}).get("rope_type", (rope_scaling or {}).get("type", "default"))
    if native_precision and position_ids is None and mrope_section is None and kind != "llama3":
        return mx.fast.rope(value, head_dim, traditional=False, base=theta,
                            scale=1 / scaling_factor, offset=offset)
    half = head_dim // 2
    inv_freq = mx.power(theta, -mx.arange(half, dtype=mx.float32) / half)
    if rope_scaling and rope_scaling.get("rope_type", rope_scaling.get("type")) == "llama3":
        factor = rope_scaling["factor"]
        low, high = rope_scaling["low_freq_factor"], rope_scaling["high_freq_factor"]
        original = rope_scaling["original_max_position_embeddings"]
        denominators = theta ** (mx.arange(0, head_dim, 2) / head_dim)
        wavelengths = 2 * mx.pi * denominators
        smooth = (original / wavelengths - low) / (high - low)
        mixed = denominators / (smooth + (1 - smooth) / factor)
        denominators = mx.where(wavelengths < original / high, denominators,
                                mx.where(wavelengths > original / low,
                                         denominators * factor, mixed))
        if position_ids is None:
            return mx.fast.rope(value, head_dim, traditional=False, base=None, scale=1.0,
                                offset=offset, freqs=denominators)
        inv_freq = 1 / denominators
    if position_ids is None:
        positions = mx.arange(offset, offset + length, dtype=mx.float32)
        positions = mx.broadcast_to(positions[None, :], (batch, length))
        frequencies = (positions[..., None] / scaling_factor) * inv_freq
    elif position_ids.ndim == 2:
        frequencies = (position_ids.astype(mx.float32)[..., None] / scaling_factor) * inv_freq
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
    return (value * mx.cos(frequencies) + _rotate_half(value) * mx.sin(frequencies)).astype(
        value.dtype)


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
        rope_scaling: Any | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling = rope_scaling
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
        trace_layer = getattr(self, "_mlx_one_trace_layer", None)

        def trace(name, value, **metadata):
            if trace_layer is not None:
                emit(trace_layer, name, value, metadata=metadata)

        trace("attention_input", hidden_states)
        execution_dtype = getattr(self, "_mlx_one_activation_dtype", None)
        queries = project(self.q_proj, hidden_states,
            role="attention_qkv", execution_dtype=execution_dtype).reshape(
            batch, length, self.num_heads, self.head_dim)
        keys = project(self.k_proj, hidden_states,
            role="attention_qkv", execution_dtype=execution_dtype).reshape(
            batch, length, self.num_kv_heads, self.head_dim)
        values = project(self.v_proj, hidden_states,
            role="attention_qkv", execution_dtype=execution_dtype).reshape(
            batch, length, self.num_kv_heads, self.head_dim)
        trace("q_projected", queries)
        trace("k_projected", keys)
        trace("v_projected", values)
        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        trace("q_normalized", queries)
        trace("k_normalized", keys)
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
            rope_scaling=self.rope_scaling,
            native_precision=execution_dtype is None,
        )
        keys = apply_rope(
            keys,
            offset=offset,
            theta=self.rope_theta,
            position_ids=position_ids,
            mrope_section=self.mrope_section,
            scaling_factor=self.rope_scaling_factor,
            rope_scaling=self.rope_scaling,
            native_precision=execution_dtype is None,
        )
        trace("q_rope", queries, offset=offset)
        trace("k_rope", keys, offset=offset)
        if execution_dtype is not None:
            dtype = getattr(mx, execution_dtype)
            queries, keys, values = (t.astype(dtype) for t in (queries, keys, values))
        trace("q_operand", queries)
        trace("k_operand", keys)
        trace("v_operand", values)
        trace("positions", position_ids, offset=offset, length=length,
              effective_positions=(list(range(offset, offset + length))
                                   if trace_layer is not None and position_ids is None else None))
        trace("attention_mask", mask, scale=self.scale, offset=offset)
        if trace_layer is not None and cache is not None:
            for component, value in enumerate(cache.state()):
                trace(f"cache_before_{component}", value, offset=cache.offset)
        if cache is not None and callable(getattr(cache, "attend", None)):
            output = cache.attend(queries, keys, values, scale=self.scale,
                                  position_ids=position_ids)
            output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
            if execution_dtype is not None:
                output = output.astype(getattr(mx, execution_dtype))
            return project(self.o_proj, output,
                role="attention_output", execution_dtype=execution_dtype)
        stable_int8 = execution_dtype == "float16" and getattr(self.q_proj, "bits", None) == 8
        decision = select_attention(queries, keys, values, mask=mask, history=offset,
                                    requires_stable_rounding=stable_int8)
        if cache is not None:
            with phase("kv_append", output=lambda: (keys, values)):
                keys, values = cache.update(keys, values)
        trace("k_committed", keys, offset=0 if cache is None else cache.offset)
        trace("v_committed", values, offset=0 if cache is None else cache.offset)
        output = attention(queries, keys, values, scale=self.scale, mask=mask,
                           stable=stable_int8, promote=execution_dtype is not None,
                           decision=decision)
        trace("attention_output", output, selected=decision.selected, reason=decision.reason)
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if execution_dtype is not None:
            output = output.astype(getattr(mx, execution_dtype))
        trace("attention_output_cast", output)
        projected = project(self.o_proj, output,
            role="attention_output", execution_dtype=execution_dtype)
        trace("attention_projected", projected)
        return projected


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, value: Any, *, execution_dtype: str | None = None) -> Any:
        gate = project(self.gate_proj, value, role="mlp", execution_dtype=execution_dtype)
        up = project(self.up_proj, value, role="mlp", execution_dtype=execution_dtype)
        return project(self.down_proj, (gate * mx.sigmoid(gate)) * up,
                       role="mlp", execution_dtype=execution_dtype)


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
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        position_ids: Any | None = None,
    ) -> Any:
        trace_layer = getattr(self, "_mlx_one_trace_layer", None)
        if trace_layer is not None:
            emit(trace_layer, "hidden_input", hidden_states)
        attended = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attended
        if trace_layer is not None:
            emit(trace_layer, "attention_residual", hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states),
            execution_dtype=getattr(self.self_attn, "_mlx_one_activation_dtype", None))
        output = hidden_states + mlp_output
        if trace_layer is not None:
            emit(trace_layer, "mlp_output", mlp_output)
            emit(trace_layer, "hidden_output", output)
        return output


def quick_gelu(value: Any) -> Any:
    return value * mx.sigmoid(1.702 * value)
