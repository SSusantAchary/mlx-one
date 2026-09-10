"""Native MLX Whisper encoder-decoder architecture."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import EncoderDecoderKVCache, make_encoder_decoder_caches
from mlx_one.core.outputs import ASRModelOutput
from mlx_one.models.audio.whisper.config import WhisperConfig


class WhisperAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        hidden: Any,
        *,
        key_value: Any | None = None,
        cache: EncoderDecoderKVCache | None = None,
        cross_attention: bool = False,
        causal: bool = False,
        output_attentions: bool = False,
    ) -> tuple[Any, Any | None]:
        batch, query_length, dim = hidden.shape
        query = self.q_proj(hidden).reshape(
            batch, query_length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        source = hidden if key_value is None else key_value
        source_length = source.shape[1]
        shape = (batch, source_length, self.heads, self.head_dim)
        key = self.k_proj(source).reshape(shape).transpose(0, 2, 1, 3)
        value = self.v_proj(source).reshape(shape).transpose(0, 2, 1, 3)
        offset = 0
        if cache is not None:
            if cross_attention:
                key, value = cache.update_cross(key, value)
            else:
                offset = cache.offset
                key, value = cache.update_self(key, value)
        scores = (query * self.scale) @ key.transpose(0, 1, 3, 2)
        if causal:
            query_positions = mx.arange(offset, offset + query_length)[:, None]
            key_positions = mx.arange(key.shape[2])[None, :]
            scores = scores + mx.where(key_positions <= query_positions, 0.0, -1e9)
        probabilities = self.dropout(mx.softmax(scores, axis=-1, precise=True))
        attended = probabilities @ value
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, query_length, dim)
        return self.out_proj(attended), probabilities if output_attentions else None


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        dim = config.d_model
        self.self_attn = WhisperAttention(
            dim, config.encoder_attention_heads, config.attention_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_dropout = nn.Dropout(config.activation_dropout)

    def __call__(self, hidden: Any) -> Any:
        residual = hidden
        attended, _ = self.self_attn(self.self_attn_layer_norm(hidden))
        hidden = residual + self.dropout(attended)
        residual = hidden
        hidden = self.activation_dropout(nn.gelu(self.fc1(self.final_layer_norm(hidden))))
        return residual + self.dropout(self.fc2(hidden))


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        dim = config.d_model
        self.self_attn = WhisperAttention(
            dim, config.decoder_attention_heads, config.attention_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.encoder_attn = WhisperAttention(
            dim, config.decoder_attention_heads, config.attention_dropout
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, config.decoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_dropout = nn.Dropout(config.activation_dropout)

    def __call__(
        self,
        hidden: Any,
        encoder_hidden: Any,
        cache: EncoderDecoderKVCache | None,
        output_attentions: bool,
    ) -> tuple[Any, Any | None, Any | None]:
        residual = hidden
        attended, self_weights = self.self_attn(
            self.self_attn_layer_norm(hidden),
            cache=cache,
            causal=True,
            output_attentions=output_attentions,
        )
        hidden = residual + self.dropout(attended)
        residual = hidden
        attended, cross_weights = self.encoder_attn(
            self.encoder_attn_layer_norm(hidden),
            key_value=encoder_hidden,
            cache=cache,
            cross_attention=True,
            output_attentions=output_attentions,
        )
        hidden = residual + self.dropout(attended)
        residual = hidden
        hidden = self.activation_dropout(nn.gelu(self.fc1(self.final_layer_norm(hidden))))
        hidden = residual + self.dropout(self.fc2(hidden))
        return hidden, self_weights, cross_weights


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self, input_features: Any, output_hidden_states: bool = False
    ) -> tuple[Any, tuple[Any, ...] | None]:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [batch, mel_bins, frames]")
        if input_features.shape[1:] != (self.config.num_mel_bins, self.config.input_frames):
            raise ValueError(
                "input_features must match configured mel bins and padded frame length"
            )
        hidden = input_features.transpose(0, 2, 1)
        hidden = nn.gelu(self.conv1(hidden))
        hidden = nn.gelu(self.conv2(hidden))
        positions = self.embed_positions(mx.arange(self.config.max_source_positions))
        hidden = self.dropout(hidden + positions[None, ...])
        states = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden)
            if states is not None:
                states.append(hidden)
        hidden = self.layer_norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, tuple(states) if states is not None else None


class WhisperDecoder(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)
        self.layers = [WhisperDecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        input_ids: Any,
        encoder_hidden: Any,
        *,
        cache: tuple[EncoderDecoderKVCache, ...] | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> tuple[Any, tuple[Any, ...] | None, tuple[Any, ...] | None, tuple[Any, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [batch, sequence]")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Whisper cache count must match decoder layer count")
        offset = cache[0].offset if cache else 0
        if offset + input_ids.shape[1] > self.config.max_target_positions:
            raise ValueError("decoder sequence exceeds max_target_positions")
        positions = mx.arange(offset, offset + input_ids.shape[1])
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)[None, ...]
        hidden = self.dropout(hidden)
        states = [hidden] if output_hidden_states else None
        self_attentions = [] if output_attentions else None
        cross_attentions = [] if output_attentions else None
        for index, layer in enumerate(self.layers):
            hidden, self_weights, cross_weights = layer(
                hidden,
                encoder_hidden,
                cache[index] if cache is not None else None,
                output_attentions,
            )
            if states is not None:
                states.append(hidden)
            if self_attentions is not None:
                self_attentions.append(self_weights)
                cross_attentions.append(cross_weights)  # type: ignore[union-attr]
        hidden = self.layer_norm(hidden)
        if states is not None:
            states[-1] = hidden
        return (
            hidden,
            tuple(states) if states is not None else None,
            tuple(self_attentions) if self_attentions is not None else None,
            tuple(cross_attentions) if cross_attentions is not None else None,
        )


class WhisperModel(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)


class WhisperForConditionalGeneration(nn.Module):
    def __init__(self, config: WhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.model = WhisperModel(config)

    def make_cache(self) -> tuple[EncoderDecoderKVCache, ...]:
        return make_encoder_decoder_caches(self.config.decoder_layers)

    def encode(
        self, input_features: Any, *, output_hidden_states: bool = False
    ) -> tuple[Any, tuple[Any, ...] | None]:
        return self.model.encoder(input_features, output_hidden_states)

    def __call__(
        self,
        input_features: Any | None,
        decoder_input_ids: Any,
        *,
        encoder_outputs: Any | None = None,
        cache: tuple[EncoderDecoderKVCache, ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ASRModelOutput:
        if encoder_outputs is None:
            if input_features is None:
                raise ValueError("input_features or encoder_outputs is required")
            encoder_hidden, encoder_states = self.encode(
                input_features, output_hidden_states=output_hidden_states
            )
        else:
            encoder_hidden, encoder_states = encoder_outputs, None
        if use_cache and cache is None:
            cache = self.make_cache()
        hidden, decoder_states, decoder_attentions, cross_attentions = self.model.decoder(
            decoder_input_ids,
            encoder_hidden,
            cache=cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.model.decoder.embed_tokens.as_linear(hidden)
        return ASRModelOutput(
            logits,
            encoder_hidden,
            hidden,
            cache=cache,
            encoder_hidden_states=encoder_states,
            decoder_hidden_states=decoder_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
        )
