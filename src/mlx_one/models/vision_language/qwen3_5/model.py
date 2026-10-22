"""Model: Qwen3.5 multimodal

Paper: Official Qwen model documentation (no formal technical report located)
Paper link: https://qwenlm.github.io/
Architecture gist: Multimodal Qwen architecture combines visual processing, dense language decoding,
linear-attention state, and optional multi-token prediction.
mlx-one implementation notes: Native MLX text-only and multimodal forward paths, caches, and
MTP hooks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import KVCache
from mlx_one.core.outputs import VisionLanguageModelOutput
from mlx_one.engine.compound import CompoundLinearComponent, compound_layout
from mlx_one.models.shared.execution import project
from mlx_one.models.shared.reference_math import attention_mask as causal_mask
from mlx_one.models.shared.reference_math import swiglu
from mlx_one.models.shared.vision import VisionAttention, vision_position_ids
from mlx_one.models.vision_language.qwen2_vl.model import (
    _replace_features,
    multimodal_position_ids,
)
from mlx_one.models.vision_language.qwen3_5.config import Qwen3_5Config, Qwen3_5TextConfig
from mlx_one.models.vision_language.qwen3_5.reference_math import (
    Qwen3NextRMSNormGated,
    gated_delta_forward,
)
from mlx_one.ops.logits import project_logits


def _silu(value: Any) -> Any:
    return value * mx.sigmoid(value)


class Qwen3_5RMSNorm(nn.Module):
    """One-centered RMSNorm used by the Qwen3.5 text tower."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, value: Any) -> Any:
        normalized = value * mx.rsqrt(
            mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.astype(mx.float32))).astype(value.dtype)


class Qwen3_5GatedRMSNorm(nn.Module):
    """RMS normalization followed by the recurrent output's SiLU gate."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, value: Any, gate: Any) -> Any:
        normalized = value * mx.rsqrt(
            mx.mean(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps
        )
        return (normalized * self.weight * _silu(gate.astype(mx.float32))).astype(value.dtype)


class Qwen3_5LinearCache(CompoundLinearComponent):
    """Compatibility name for independent compound convolution/recurrent state."""


def make_qwen3_5_caches(config: Qwen3_5TextConfig, dtype="float32") -> tuple[Any, ...]:
    return tuple(
        Qwen3_5LinearCache(config.linear_conv_kernel_dim, *compound_layout(config, dtype))
        if kind == "linear_attention"
        else KVCache()
        for kind in config.layer_types
    )


class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        scale = math.sqrt(1.0 / kernel_size)
        self.weight = mx.random.uniform(-scale, scale, (channels, kernel_size))

    def __call__(self, values: Any, cache: Qwen3_5LinearCache | None) -> Any:
        width = self.kernel_size - 1
        if cache is None:
            combined = mx.concatenate(
                (mx.zeros((values.shape[0], width, values.shape[2]), dtype=values.dtype), values),
                axis=1,
            )
        else:
            combined = cache.prefix(values)
        outputs = []
        # Convolution accumulates products before rounding the output. Multiplying
        # BF16/FP16 operands first loses cancellation terms before the reduction.
        weights = self.weight.T[None, ...].astype(mx.float32)
        for index in range(values.shape[1]):
            window = combined[:, index : index + self.kernel_size].astype(mx.float32)
            outputs.append(mx.sum(window * weights, axis=1))
        return mx.stack(outputs, axis=1).astype(values.dtype)


def _l2_normalize(value: Any) -> Any:
    return value * mx.rsqrt(mx.sum(value.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6)


def _gated_delta_decay(a: Any, a_log: Any, dt_bias: Any) -> Any:
    shifted = a.astype(mx.float32) + dt_bias
    # Decay is an FP32 recurrent-state operand, including the inner exponential.
    return mx.exp(-mx.exp(a_log.astype(mx.float32)) * mx.logaddexp(shifted, mx.zeros_like(shifted)))


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.config = config
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = DepthwiseCausalConv1d(self.conv_dim, config.linear_conv_kernel_dim)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def __call__(
        self,
        hidden: Any,
        *,
        cache: Qwen3_5LinearCache | None,
        attention_mask: Any | None,
    ) -> Any:
        mask = None if attention_mask is None else attention_mask[:, -hidden.shape[1] :]
        return gated_delta_forward(self, hidden, cache=cache, mask=mask)


def _rotate_half(value: Any) -> Any:
    half = value.shape[-1] // 2
    return mx.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _partial_mrope(value: Any, position_ids: Any, config: Qwen3_5TextConfig) -> Any:
    rotary_dim = config.rotary_dim
    half = rotary_dim // 2
    rope = config.rope_parameters
    theta = rope["rope_theta"]
    inv_freq = mx.power(theta, -mx.arange(half, dtype=mx.float32) / half)
    if position_ids.ndim == 2:
        positions = mx.stack((position_ids, position_ids, position_ids), axis=0)
    else:
        positions = position_ids
    raw = positions.astype(mx.float32)[..., None] * inv_freq
    sections = rope["mrope_section"]
    if sections:
        frequencies = []
        for index in range(half):
            axis = 0
            if index % 3 == 1 and index < sections[1] * 3:
                axis = 1
            elif index % 3 == 2 and index < sections[2] * 3:
                axis = 2
            frequencies.append(raw[axis, ..., index])
        frequency = mx.stack(frequencies, axis=-1)
    else:
        frequency = raw[0]
    embedding = mx.concatenate((frequency, frequency), axis=-1)[:, None]
    rotated, passthrough = value[..., :rotary_dim], value[..., rotary_dim:]
    rotated = rotated * mx.cos(embedding) + _rotate_half(rotated) * mx.sin(embedding)
    return mx.concatenate((rotated, passthrough), axis=-1).astype(value.dtype)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3_5RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(config.head_dim, config.rms_norm_eps)

    def __call__(
        self, hidden: Any, *, mask: Any | None, position_ids: Any, cache: KVCache | None
    ) -> Any:
        batch, length, _ = hidden.shape
        c = self.config
        q_proj_output = self.q_proj(hidden)
        query, gate = mx.split(
            q_proj_output.reshape(batch, length, c.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, length, -1)
        key, value = self.k_proj(hidden), self.v_proj(hidden)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(key.reshape(batch, length, c.num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(batch, length, c.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        if position_ids is None:
            settings = dict(
                dims=c.rotary_dim,
                traditional=False,
                base=c.rope_parameters["rope_theta"],
                scale=1.0,
                offset=offset,
            )
            query, key = mx.fast.rope(query, **settings), mx.fast.rope(key, **settings)
        else:
            query = _partial_mrope(query, position_ids, c)
            key = _partial_mrope(key, position_ids, c)
        if cache is not None and callable(getattr(cache, "attend", None)):
            output = cache.attend(
                query, key, value, scale=c.head_dim**-0.5, position_ids=position_ids
            )
        else:
            if cache is not None:
                key, value = cache.update(key, value)
            output = mx.fast.scaled_dot_product_attention(
                query,
                key.astype(query.dtype),
                value.astype(query.dtype),
                scale=c.head_dim**-0.5,
                mask=mask,
            )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden: Any) -> Any:
        return self.down_proj(swiglu(self.gate_proj(hidden), self.up_proj(hidden)))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, index: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[index]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        attention_mask: Any | None,
        position_ids: Any,
        cache: Any | None,
    ) -> Any:
        normalized = self.input_layernorm(hidden)
        if self.block_type == "linear_attention":
            mixed = self.linear_attn(normalized, cache=cache, attention_mask=attention_mask)
        else:
            mixed = self.self_attn(normalized, mask=mask, position_ids=position_ids, cache=cache)
        hidden = hidden + mixed
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3_5DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Any | None,
        *,
        input_embeddings: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Any, tuple[Any, ...] | None, tuple[Any, ...] | None]:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of input_ids or input_embeddings")
        hidden = self.embed_tokens(input_ids) if input_embeddings is None else input_embeddings
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache count must match Qwen3.5 layer count")
        offset = 0 if cache is None else cache[0].offset
        length = hidden.shape[1]
        if offset + length > self.config.max_position_embeddings:
            raise ValueError("Qwen3.5 sequence exceeds max_position_embeddings")
        mask = causal_mask(length, offset, attention_mask)
        if attention_mask is not None:
            expected = (hidden.shape[0], offset + length)
            if attention_mask.shape != expected:
                raise ValueError("attention_mask must cover the complete cached sequence")
        states = [hidden] if output_hidden_states else None
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                mask=mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=None if cache is None else cache[index],
            )
            if states is not None:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, cache, tuple(states) if states is not None else None


class Qwen3_5MTP(nn.Module):
    """Checkpoint-native multi-token predictor for speculative decoding."""

    def __init__(self, config: Qwen3_5TextConfig) -> None:
        super().__init__()
        self.config = replace(
            config,
            num_hidden_layers=config.mtp_num_hidden_layers,
            layer_types=("full_attention",) * config.mtp_num_hidden_layers,
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = [
            Qwen3_5DecoderLayer(self.config, index) for index in range(config.mtp_num_hidden_layers)
        ]
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def make_cache(self) -> tuple[KVCache, ...]:
        return make_qwen3_5_caches(self.config)  # type: ignore[return-value]

    def __call__(
        self,
        embeddings: Any,
        hidden_states: Any,
        *,
        cache: tuple[KVCache, ...] | None = None,
        position_offset: int = 0,
    ) -> Any:
        hidden = self.fc(
            mx.concatenate(
                (
                    self.pre_fc_norm_embedding(embeddings),
                    self.pre_fc_norm_hidden(hidden_states),
                ),
                axis=-1,
            )
        )
        offset = 0 if cache is None else cache[0].offset
        length = hidden.shape[1]
        positions = mx.arange(position_offset + offset, position_offset + offset + length)[None, :]
        mask = causal_mask(length, offset)
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                mask=mask,
                attention_mask=None,
                position_ids=positions,
                cache=None if cache is None else cache[index],
            )
        return self.norm(hidden)


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, hidden: Any) -> Any:
        value = self.linear_fc1(hidden)
        value = (
            0.5 * value * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)))
        )
        return self.linear_fc2(value)


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = Qwen3_5VisionMLP(config)

    def __call__(self, hidden: Any, positions: Any, lengths: Sequence[int]) -> Any:
        chunks = []
        start = 0
        for length in lengths:
            current = hidden[start : start + length]
            position = positions[start : start + length]
            current = current + self.attn(self.norm1(current), position)
            chunks.append(current + self.mlp(self.norm2(current)))
            start += length
        return mx.concatenate(chunks)


class Qwen3_5PatchEmbed(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.proj = nn.Linear(self.patch_dim, config.hidden_size, bias=True)

    def __call__(self, patches: Any) -> Any:
        if patches.ndim != 2 or patches.shape[-1] != self.patch_dim:
            raise ValueError(f"pixel patches must have shape [patches, {self.patch_dim}]")
        return self.proj(patches)


class Qwen3_5PatchMerger(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.merged_dim = config.hidden_size * config.spatial_merge_size**2
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_dim, self.merged_dim, bias=True)
        self.linear_fc2 = nn.Linear(self.merged_dim, config.out_hidden_size, bias=True)

    def __call__(self, hidden: Any) -> Any:
        hidden = self.norm(hidden).reshape(-1, self.merged_dim)
        hidden = self.linear_fc1(hidden)
        hidden = (
            0.5
            * hidden
            * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden**3)))
        )
        return self.linear_fc2(hidden)


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = Qwen3_5PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [Qwen3_5VisionBlock(config) for _ in range(config.depth)]
        self.merger = Qwen3_5PatchMerger(config)

    def __call__(self, patches: Any, grid_thw: Sequence[Sequence[int]]) -> tuple[Any, Any]:
        hidden = self.patch_embed(patches)
        positions, lengths = vision_position_ids(grid_thw, self.config.spatial_merge_size)
        if sum(lengths) != hidden.shape[0]:
            raise ValueError("vision grid patch count does not match pixel patches")
        hidden = hidden + _interpolate_positions(self.pos_embed, grid_thw, self.config)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths)
        return self.merger(hidden), hidden


def _interpolate_positions(embedding: Any, grids: Sequence[Sequence[int]], config: Any) -> Any:
    side = int(config.num_position_embeddings**0.5)
    outputs = []
    for temporal, height, width in grids:
        merge = config.spatial_merge_size
        for _ in range(int(temporal)):
            for merged_row in range(0, int(height), merge):
                for merged_column in range(0, int(width), merge):
                    for local_row in range(merge):
                        for local_column in range(merge):
                            row = merged_row + local_row
                            column = merged_column + local_column
                            source_row = 0.0 if height == 1 else row * (side - 1) / (height - 1)
                            low_row = int(source_row)
                            high_row = min(low_row + 1, side - 1)
                            row_weight = source_row - low_row
                            source_column = 0.0 if width == 1 else column * (side - 1) / (width - 1)
                            low_col = int(source_column)
                            high_col = min(low_col + 1, side - 1)
                            col_weight = source_column - low_col
                            indexes = mx.array(
                                [
                                    low_row * side + low_col,
                                    low_row * side + high_col,
                                    high_row * side + low_col,
                                    high_row * side + high_col,
                                ]
                            )
                            weights = mx.array(
                                [
                                    (1 - row_weight) * (1 - col_weight),
                                    (1 - row_weight) * col_weight,
                                    row_weight * (1 - col_weight),
                                    row_weight * col_weight,
                                ]
                            )
                            outputs.append(mx.sum(embedding(indexes) * weights[:, None], axis=0))
    return mx.stack(outputs)


class Qwen3_5ForConditionalGeneration(nn.Module):
    _mlx_one_reference_first_math = True

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size, config.text_config.vocab_size, bias=False
            )

    def enable_mtp(self) -> None:
        if not hasattr(self, "mtp"):
            self.mtp = Qwen3_5MTP(self.config.text_config)

    def make_mtp_cache(self) -> tuple[KVCache, ...]:
        if not hasattr(self, "mtp"):
            raise RuntimeError("Qwen3.5 checkpoint has no MTP predictor")
        return self.mtp.make_cache()

    def mtp_logits(
        self,
        input_ids: Any,
        hidden_states: Any,
        *,
        cache: tuple[KVCache, ...] | None = None,
        position_offset: int = 0,
    ) -> Any:
        if not hasattr(self, "mtp"):
            raise RuntimeError("Qwen3.5 checkpoint has no MTP predictor")
        embeddings = self.language_model.embed_tokens(input_ids)
        hidden = self.mtp(embeddings, hidden_states, cache=cache, position_offset=position_offset)
        return (
            project(
                self.language_model.embed_tokens,
                hidden,
                execution_dtype=getattr(self, "_mlx_one_activation_dtype", None),
                as_linear=True,
            )
            if self.config.tie_word_embeddings
            else project(
                self.lm_head,
                hidden,
                execution_dtype=getattr(self, "_mlx_one_activation_dtype", None),
            )
        )

    def make_cache(self) -> tuple[Any, ...]:
        embedding = self.language_model.embed_tokens
        weight = embedding.scales if hasattr(embedding, "bits") else embedding.weight
        dtype = str(weight.dtype).rsplit(".", 1)[-1]
        return make_qwen3_5_caches(self.config.text_config, dtype)

    def make_state_components(self, key_type, pool=None):
        from mlx_one.engine.components import FullKVComponent
        from mlx_one.engine.paged import PagedKVComponent

        dtype = "bfloat16" if key_type == "bf16" else "float16"
        return tuple(
            cache
            if isinstance(cache, CompoundLinearComponent)
            else PagedKVComponent(pool, layer_id=i, dtype=dtype)
            if pool is not None
            else FullKVComponent(dtype)
            for i, cache in enumerate(self.make_cache())
        )

    def __call__(
        self,
        input_ids: Any,
        *,
        pixel_values: Any | None = None,
        pixel_values_videos: Any | None = None,
        image_grid_thw: Sequence[Sequence[int]] | None = None,
        video_grid_thw: Sequence[Sequence[int]] | None = None,
        mm_token_type_ids: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_hidden_states: bool = False,
        logits_mode=None,
    ) -> VisionLanguageModelOutput:
        embeddings = self.language_model.embed_tokens(input_ids)
        vision_states = []
        for patches, grid, token_id, kind in (
            (pixel_values, image_grid_thw, self.config.image_token_id, "image"),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id, "video"),
        ):
            if patches is None:
                continue
            if grid is None:
                raise ValueError(f"{kind}_grid_thw is required with pixel values")
            features, states = self.visual(patches, grid)
            embeddings = _replace_features(embeddings, input_ids == token_id, features, kind)
            vision_states.append(states)
        if mm_token_type_ids is None and vision_states:
            mm_token_type_ids = mx.zeros_like(input_ids)
            mm_token_type_ids = mx.where(
                input_ids == self.config.image_token_id, 1, mm_token_type_ids
            )
            mm_token_type_ids = mx.where(
                input_ids == self.config.video_token_id, 2, mm_token_type_ids
            )
        if position_ids is not None:
            if mm_token_type_ids is not None or vision_states:
                raise ValueError("explicit text positions cannot replace multimodal positions")
            positions = position_ids
            deltas = mx.zeros((input_ids.shape[0], 1), dtype=mx.int32)
        elif mm_token_type_ids is None:
            offset = 0 if cache is None else cache[0].offset
            base = mx.arange(offset, offset + input_ids.shape[1])[None, :]
            positions = mx.stack((base, base, base), axis=0)
            deltas = mx.zeros((input_ids.shape[0], 1), dtype=base.dtype)
        else:
            positions, deltas = multimodal_position_ids(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
                attention_mask=attention_mask,
            )
        hidden, cache, states = self.language_model(
            None,
            input_embeddings=embeddings,
            attention_mask=attention_mask,
            position_ids=positions
            if position_ids is not None or mm_token_type_ids is not None
            else None,
            cache=cache,
            output_hidden_states=output_hidden_states,
        )
        logits = project_logits(
            hidden,
            self.language_model.embed_tokens.as_linear
            if self.config.tie_word_embeddings
            else self.lm_head,
            logits_mode,
            attention_mask=attention_mask,
        )
        return VisionLanguageModelOutput(
            logits,
            hidden,
            cache=cache,
            hidden_states=states,
            vision_hidden_states=mx.concatenate(vision_states) if vision_states else None,
            position_ids=positions,
            rope_deltas=deltas,
        )
