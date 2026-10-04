"""Native MLX original Moonshine waveform encoder and cached text decoder.

Architecture adapted from Hugging Face Transformers Moonshine, Copyright 2025
The HuggingFace Inc. team, Apache-2.0. Reference source is pinned in the guide.
No upstream inference package is imported by this implementation.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import EncoderDecoderKVCache, make_encoder_decoder_caches
from mlx_one.core.outputs import ASRModelOutput
from mlx_one.models.audio.moonshine.config import MoonshineConfig, encoded_length


class MoonshineAttention(nn.Module):
    def __init__(self, config: MoonshineConfig, heads: int) -> None:
        super().__init__()
        dim = config.hidden_size
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(
            int(self.head_dim * config.partial_rotary_factor),
            traditional=True,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden: Any,
        *,
        source: Any | None = None,
        cache: EncoderDecoderKVCache | None = None,
        causal: bool = False,
        mask: Any | None = None,
    ) -> Any:
        batch, length, dim = hidden.shape
        query = (
            self.q_proj(hidden)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        cross = source is not None
        offset = cache.offset if cache is not None and not cross else 0
        if cross and cache is not None and cache.cross_keys is not None:
            key, value = cache.cross_keys, cache.cross_values
        else:
            value_source = hidden if source is None else source
            shape = (batch, value_source.shape[1], self.heads, self.head_dim)
            key = self.k_proj(value_source).reshape(shape).transpose(0, 2, 1, 3)
            value = self.v_proj(value_source).reshape(shape).transpose(0, 2, 1, 3)
            if not cross:
                key = self.rope(key, offset=offset)
            if cache is not None:
                key, value = (
                    cache.update_cross(key, value) if cross else cache.update_self(key, value)
                )
        if not cross:
            query = self.rope(query, offset=offset)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        if causal:
            allowed = (
                mx.arange(key.shape[2])[None, :] <= mx.arange(offset, offset + length)[:, None]
            )
            scores = mx.where(allowed, scores, -float("inf"))
        if mask is not None:
            scores = mx.where(mask[:, None, None, :].astype(mx.bool_), scores, -float("inf"))
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        hidden = (probabilities @ value).transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return self.o_proj(hidden)


class MoonshineMLP(nn.Module):
    def __init__(self, config: MoonshineConfig, *, gated: bool) -> None:
        super().__init__()
        self.gated = gated
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * (2 if gated else 1))
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden: Any) -> Any:
        hidden = self.fc1(hidden)
        if self.gated:
            value, gate = mx.split(hidden, 2, axis=-1)
            hidden = value * nn.silu(gate)
        else:
            hidden = nn.gelu(hidden)
        return self.fc2(hidden)


class MoonshineEncoderLayer(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.self_attn = MoonshineAttention(config, config.encoder_num_attention_heads)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.mlp = MoonshineMLP(config, gated=False)

    def __call__(self, hidden: Any, mask: Any | None) -> Any:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), mask=mask)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class MoonshineEncoder(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.conv1 = nn.Conv1d(1, dim, 127, stride=64, bias=False)
        self.conv2 = nn.Conv1d(dim, 2 * dim, 7, stride=3)
        self.conv3 = nn.Conv1d(2 * dim, dim, 3, stride=2)
        self.groupnorm = nn.GroupNorm(1, dim, eps=1e-5, pytorch_compatible=True)
        self.layers = [
            MoonshineEncoderLayer(config) for _ in range(config.encoder_num_hidden_layers)
        ]
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def __call__(self, input_values: Any, attention_mask: Any | None = None) -> Any:
        if input_values.ndim != 2 or encoded_length(input_values.shape[-1]) < 1:
            raise ValueError("Moonshine requires [batch, samples] with at least 895 samples")
        hidden = self.groupnorm(mx.tanh(self.conv1(input_values[..., None])))
        hidden = nn.gelu(self.conv2(hidden))
        hidden = nn.gelu(self.conv3(hidden))
        if attention_mask is not None:
            if attention_mask.shape != input_values.shape:
                raise ValueError("Moonshine attention mask must match waveform shape")
            attention_mask = attention_mask[:, ::384][:, : hidden.shape[1]]
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.layer_norm(hidden)


class MoonshineDecoderLayer(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.self_attn = MoonshineAttention(config, config.decoder_num_attention_heads)
        self.encoder_attn = MoonshineAttention(config, config.decoder_num_attention_heads)
        self.input_layernorm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.final_layernorm = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.mlp = MoonshineMLP(config, gated=True)

    def __call__(
        self,
        hidden: Any,
        encoded: Any,
        cache: EncoderDecoderKVCache | None,
        encoder_mask: Any | None,
    ) -> Any:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cache=cache, causal=True)
        hidden = hidden + self.encoder_attn(
            self.post_attention_layernorm(hidden), source=encoded, cache=cache, mask=encoder_mask
        )
        return hidden + self.mlp(self.final_layernorm(hidden))


class MoonshineDecoder(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            MoonshineDecoderLayer(config) for _ in range(config.decoder_num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)

    def __call__(
        self,
        input_ids: Any,
        encoded: Any,
        cache: tuple[EncoderDecoderKVCache, ...] | None,
        encoder_mask: Any | None = None,
    ) -> Any:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("Moonshine decoder tokens must be nonempty [batch, sequence]")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Moonshine cache count must match decoder layer count")
        hidden = self.embed_tokens(input_ids)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, encoded, cache[index] if cache else None, encoder_mask)
        return self.norm(hidden)


class MoonshineModel(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.encoder = MoonshineEncoder(config)
        self.decoder = MoonshineDecoder(config)


class MoonshineForConditionalGeneration(nn.Module):
    def __init__(self, config: MoonshineConfig) -> None:
        super().__init__()
        self.config = config
        self.model = MoonshineModel(config)

    def make_cache(self) -> tuple[EncoderDecoderKVCache, ...]:
        return make_encoder_decoder_caches(self.config.decoder_num_hidden_layers)

    def encode(self, input_values: Any, attention_mask: Any | None = None) -> Any:
        return self.model.encoder(input_values, attention_mask)

    def decode(
        self,
        input_ids: Any,
        encoded: Any,
        *,
        cache: tuple[EncoderDecoderKVCache, ...] | None = None,
        encoder_mask: Any | None = None,
    ) -> ASRModelOutput:
        hidden = self.model.decoder(input_ids, encoded, cache, encoder_mask)
        logits = self.model.decoder.embed_tokens.as_linear(hidden)
        return ASRModelOutput(logits, encoded, hidden, cache)

    def __call__(
        self,
        input_values: Any,
        decoder_input_ids: Any,
        *,
        attention_mask: Any | None = None,
        cache: tuple[EncoderDecoderKVCache, ...] | None = None,
    ) -> ASRModelOutput:
        encoded = self.encode(input_values, attention_mask)
        mask = None if attention_mask is None else attention_mask[:, ::384][:, : encoded.shape[1]]
        return self.decode(decoder_input_ids, encoded, cache=cache, encoder_mask=mask)
