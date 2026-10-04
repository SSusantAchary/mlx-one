"""Native MLX port of the pinned Qwen3-ASR SDPA inference architecture.

Adapted from QwenLM/Qwen3-ASR, commit 7c6daf77a2421100f5fb066495372c00129d39ff.
Copyright 2026 The Alibaba Qwen team. SPDX-License-Identifier: Apache-2.0.
No reference package is imported at inference time.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_kv_caches
from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig, QwenAudioConfig, audio_token_length
from mlx_one.models.shared.transformer import causal_mask


def gelu(x: Any) -> Any:
    # PyTorch's BF16 GELU evaluates in float32, then rounds once at its output.
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


def attention(q: Any, k: Any, v: Any, *, scale: float, mask: Any = None) -> Any:
    # Preserve the established native baseline. CPU-reference BF16 rounding
    # experiments live in qualification-only tooling until checkpoint gates pass.
    dtype = q.dtype
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        k.astype(mx.float32),
        v.astype(mx.float32),
        scale=scale,
        mask=None if mask is None else mask.astype(mx.float32),
    ).astype(dtype)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((width,))
        self._eps = eps

    def __call__(self, x: Any) -> Any:
        value = x.astype(mx.float32)
        normalized = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self._eps)
        # The pinned reference casts normalized states before multiplying gain.
        return self.weight * normalized.astype(x.dtype)


class TextMLP(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def __call__(self, x: Any) -> Any:
        gate = self.gate_proj(x)
        activated = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
        return self.down_proj(activated * self.up_proj(x))


class AudioAttention(nn.Module):
    def __init__(self, config: QwenAudioConfig) -> None:
        super().__init__()
        self._heads = config.encoder_attention_heads
        self._head_dim = config.d_model // self._heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def __call__(self, x: Any) -> Any:
        b, length, _ = x.shape
        q, k, v = [
            proj(x).reshape(b, length, self._heads, self._head_dim).transpose(0, 2, 1, 3)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]
        # The pinned non-FlashAttention reference passes no audio block mask.
        # Do not silently substitute its different FA2 windowed attention path.
        out = attention(q, k, v, scale=self._head_dim**-0.5)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class AudioLayer(nn.Module):
    def __init__(self, config: QwenAudioConfig) -> None:
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def __call__(self, x: Any) -> Any:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(gelu(self.fc1(self.final_layer_norm(x))))


class AudioEncoder(nn.Module):
    def __init__(self, config: QwenAudioConfig) -> None:
        super().__init__()
        self._config = config
        width = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, width, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.conv_out = nn.Linear(width * 16, config.d_model, bias=False)
        self.layers = [AudioLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(config.d_model, eps=1e-5)
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.proj2 = nn.Linear(config.d_model, config.output_dim)

    def __call__(self, features: Any) -> Any:
        if (
            features.ndim != 2
            or features.shape[0] != self._config.num_mel_bins
            or features.shape[1] < 1
        ):
            raise ValueError("Qwen encoder requires one nonempty [128, frames] feature matrix")
        width = self._config.n_window * 2
        chunks = [
            features[:, start : start + width] for start in range(0, features.shape[1], width)
        ]
        maximum = max(chunk.shape[1] for chunk in chunks)
        lengths = [audio_token_length(chunk.shape[1]) for chunk in chunks]
        padded = mx.stack(
            [mx.pad(chunk, ((0, 0), (0, maximum - chunk.shape[1]))) for chunk in chunks]
        )
        padded = padded[..., None].astype(self.conv2d1.weight.dtype)
        outputs = []
        for start in range(0, len(chunks), self._config.conv_chunksize):
            x = padded[start : start + self._config.conv_chunksize]
            for conv in (self.conv2d1, self.conv2d2, self.conv2d3):
                x = gelu(conv(x))
            b, f, t, c = x.shape
            outputs.append(self.conv_out(x.transpose(0, 2, 3, 1).reshape(b, t, c * f)))
        x = mx.concatenate(outputs, axis=0)
        half = self._config.d_model // 2
        increment = math.log(10000.0) / (half - 1)
        inv = mx.exp(-increment * mx.arange(half, dtype=mx.float32))
        phases = mx.arange(x.shape[1], dtype=mx.float32)[:, None] * inv[None, :]
        positions = mx.concatenate((mx.sin(phases), mx.cos(phases)), axis=-1).astype(x.dtype)
        x = x + positions[None]
        hidden = mx.concatenate([x[i, :length] for i, length in enumerate(lengths)])[None]
        for layer in self.layers:
            hidden = layer(hidden)
        return self.proj2(gelu(self.proj1(self.ln_post(hidden[0]))))


def rotary(x: Any, offset: int, theta: float) -> Any:
    half = x.shape[-1] // 2
    # Preserve the reference's positive power followed by reciprocal; folding
    # it into a negative power changes float32 rounding before BF16 sine/cosine.
    inv = 1.0 / mx.power(theta, mx.arange(half, dtype=mx.float32) / half)
    phases = mx.arange(offset, offset + x.shape[2], dtype=mx.float32)[:, None] * inv[None]
    phases = mx.concatenate((phases, phases), axis=-1)[None, None]
    cos, sin = mx.cos(phases).astype(x.dtype), mx.sin(phases).astype(x.dtype)
    rotated = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + rotated * sin


class TextAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self._config = config
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
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
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, hidden: Any, cache: Any | None) -> Any:
        c = self._config
        b, length, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).reshape(b, length, c.num_attention_heads, c.head_dim)
        ).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(hidden).reshape(b, length, c.num_key_value_heads, c.head_dim)
        ).transpose(0, 2, 1, 3)
        v = (
            self.v_proj(hidden)
            .reshape(b, length, c.num_key_value_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        offset = 0 if cache is None else cache.offset
        q, k = rotary(q, offset, c.rope_theta), rotary(k, offset, c.rope_theta)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = causal_mask(length, offset)
        if mask is not None:
            mask = mask.astype(q.dtype)
        out = attention(q, k, v, scale=c.head_dim**-0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class TextLayer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = TextAttention(config)
        self.mlp = TextMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: Any, cache: Any | None) -> Any:
        x = x + self.self_attn(self.input_layernorm(x), cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextDecoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TextLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, embeds: Any, caches: tuple[Any, ...] | None) -> Any:
        if caches is not None and len(caches) != len(self.layers):
            raise ValueError("cache count must match Qwen decoder layers")
        for layer, cache in zip(self.layers, caches or (None,) * len(self.layers), strict=True):
            embeds = layer(embeds, cache)
        return self.norm(embeds)


class QwenASRModel(nn.Module):
    def __init__(self, config: QwenASRConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextDecoder(config.text_config)
        if config.model_type == "qwen3_forced_aligner":
            self.lm_head = nn.Linear(
                config.text_config.hidden_size, config.classify_num, bias=False
            )

    def make_cache(self) -> tuple[Any, ...]:
        return make_kv_caches(self.config.text_config.num_hidden_layers)

    def __call__(
        self,
        ids: Any,
        *,
        features: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_positions: Any | None = None,
    ) -> Any:
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Qwen audio inference currently requires batch size one")
        if cache and len({item.offset for item in cache}) != 1:
            raise ValueError("Qwen decoder caches have inconsistent offsets")
        embeds = self.model.embed_tokens(ids)
        if features is not None:
            if cache and cache[0].offset:
                raise ValueError("audio features can be supplied only at prefill")
            encoded = self.audio_tower(features).astype(embeds.dtype)
            indices = [
                i for i, token in enumerate(ids[0].tolist()) if token == self.config.audio_token_id
            ]
            if len(indices) != encoded.shape[0]:
                raise ValueError("audio placeholders must exactly match encoded feature count")
            embeds[0, mx.array(indices)] = encoded
        hidden = self.model(embeds, cache)
        if output_positions is not None:
            hidden = hidden[:, output_positions, :]
        return (
            self.lm_head(hidden)
            if self.config.model_type == "qwen3_forced_aligner"
            else self.model.embed_tokens.as_linear(hidden)
        )
