"""Native MLX LFM2.5-Audio architecture and waveform detokenizer."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import AudioModelOutput
from mlx_one.models.audio.lfm2_audio.config import Lfm2AudioConfig
from mlx_one.models.language.lfm2.model import Lfm2Model


class AudioAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(
        self, value: Any, *, causal: bool = False, window: int | None = None
    ) -> Any:
        batch, length, dim = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, val = [qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        if causal:
            positions = mx.arange(length)
            allowed = positions[None, :] <= positions[:, None]
            if window is not None:
                allowed = allowed & (positions[None, :] > positions[:, None] - window)
            scores = scores + mx.where(allowed, 0.0, -float("inf"))
        hidden = mx.softmax(scores, axis=-1, precise=True) @ val
        return self.out_proj(hidden.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class ConformerBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        dim = config.hidden_size
        self.ffn1_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.ffn1 = nn.Linear(dim, config.intermediate_size, bias=True)
        self.ffn1_out = nn.Linear(config.intermediate_size, dim, bias=True)
        self.attn_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.self_attn = AudioAttention(dim, config.num_attention_heads)
        self.conv_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.conv_in = nn.Linear(dim, 2 * dim, bias=True)
        self.depthwise_kernel = mx.zeros((dim, config.conv_kernel_size))
        self.conv_out = nn.Linear(dim, dim, bias=True)
        self.ffn2_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.ffn2 = nn.Linear(dim, config.intermediate_size, bias=True)
        self.ffn2_out = nn.Linear(config.intermediate_size, dim, bias=True)
        self.final_norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    @staticmethod
    def _ffn(value: Any, norm: Any, first: Any, second: Any) -> Any:
        return second(nn.silu(first(norm(value))))

    def __call__(self, value: Any) -> Any:
        value = value + 0.5 * self._ffn(value, self.ffn1_norm, self.ffn1, self.ffn1_out)
        value = value + self.self_attn(self.attn_norm(value))
        conv = self.conv_in(self.conv_norm(value))
        gate, conv = mx.split(conv, 2, axis=-1)
        conv = mx.sigmoid(gate) * conv
        width = self.depthwise_kernel.shape[1]
        padded = mx.pad(conv, ((0, 0), (width // 2, width // 2), (0, 0)))
        windows = mx.stack(
            [padded[:, index : index + value.shape[1]] for index in range(width)], axis=2
        )
        conv = mx.einsum("blkd,dk->bld", windows, self.depthwise_kernel)
        value = value + self.conv_out(nn.silu(conv))
        value = value + 0.5 * self._ffn(value, self.ffn2_norm, self.ffn2, self.ffn2_out)
        return self.final_norm(value)


class AudioEncoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.subsampling_factor = config.subsampling_factor
        self.input_projection = nn.Linear(config.input_dim, config.hidden_size, bias=True)
        self.blocks = [ConformerBlock(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, features: Any) -> tuple[Any, Any]:
        hidden = self.input_projection(features[:, :: self.subsampling_factor])
        for block in self.blocks:
            hidden = block(hidden)
        lengths = mx.full((hidden.shape[0],), hidden.shape[1], dtype=mx.int32)
        return hidden, lengths


class DepthBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size)
        self.attn = AudioAttention(config.hidden_size, config.num_attention_heads)
        self.ffn_norm = nn.RMSNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.sliding_window = getattr(config, "sliding_window", None)

    def __call__(self, value: Any) -> Any:
        value = value + self.attn(
            self.attn_norm(value), causal=True, window=self.sliding_window
        )
        gate, up = mx.split(self.fc1(self.ffn_norm(value)), 2, axis=-1)
        return value + self.fc2(nn.silu(gate) * up)


class Depthformer(nn.Module):
    def __init__(self, config: Any, vocab_size: int, codebooks: int) -> None:
        super().__init__()
        self.codebooks = codebooks
        self.embeddings = [nn.Embedding(vocab_size, config.hidden_size) for _ in range(codebooks)]
        self.blocks = [DepthBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)
        if not config.tie_word_embeddings:
            self.heads = [
                nn.Linear(config.hidden_size, vocab_size, bias=False) for _ in range(codebooks)
            ]
        self.tie_word_embeddings = config.tie_word_embeddings

    def __call__(self, context: Any, codes: Any | None = None) -> Any:
        hidden = context
        if codes is not None:
            code_hidden = (
                sum(embedding(codes[..., index]) for index, embedding in enumerate(self.embeddings))
                / self.codebooks
            )
            hidden = hidden + code_hidden
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        logits = [
            embedding.as_linear(hidden) if self.tie_word_embeddings else self.heads[index](hidden)
            for index, embedding in enumerate(self.embeddings)
        ]
        return mx.stack(logits, axis=2)


class AudioDetokenizer(nn.Module):
    """LFM detokenizer with a Mimi-compatible codebook input contract."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.embeddings = [
            nn.Embedding(config.codebook_size, config.hidden_size)
            for _ in range(config.num_codebooks)
        ]
        self.blocks = [
            DepthBlock(
                type(
                    "DetokBlockConfig",
                    (),
                    {
                        "hidden_size": config.hidden_size,
                        "num_attention_heads": config.num_attention_heads,
                        "intermediate_size": config.intermediate_size,
                    },
                )
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.n_fft + 2, bias=True)

    def __call__(self, codes: Any) -> Any:
        if codes.ndim != 3 or codes.shape[-1] != self.config.num_codebooks:
            raise ValueError("audio codes must have shape [batch, frames, codebooks]")
        hidden = (
            sum(embedding(codes[..., index]) for index, embedding in enumerate(self.embeddings))
            / self.config.num_codebooks
        )
        hidden = mx.repeat(hidden, self.config.upsample_factor, axis=1)
        for block in self.blocks:
            hidden = block(hidden)
        spectrum = self.output(self.norm(hidden))
        bins = self.config.n_fft // 2 + 1
        magnitude = mx.exp(spectrum[..., :bins])
        phase = spectrum[..., bins:]
        complex_spectrum = magnitude * (mx.cos(phase) + 1j * mx.sin(phase))
        frames = mx.fft.irfft(complex_spectrum, n=self.config.n_fft, axis=-1)
        return overlap_add(frames, self.config.hop_length)


def overlap_add(frames: Any, hop_length: int) -> Any:
    output_length = (frames.shape[1] - 1) * hop_length + frames.shape[2]
    pieces = []
    for index in range(frames.shape[1]):
        left = index * hop_length
        right = output_length - left - frames.shape[2]
        pieces.append(mx.pad(frames[:, index], ((0, 0), (left, right))))
    return mx.sum(mx.stack(pieces), axis=0)


class Lfm2AudioForConditionalGeneration(nn.Module):
    def __init__(self, config: Lfm2AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = AudioEncoder(config.audio_config)
        self.audio_projection = nn.Linear(
            config.audio_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.language_model = Lfm2Model(config.text_config)
        self.depthformer = Depthformer(
            config.depthformer_config, config.audio_vocab_size, config.num_codebooks
        )
        self.detokenizer = AudioDetokenizer(config.detokenizer_config)

    def __call__(
        self,
        input_ids: Any,
        *,
        input_features: Any | None = None,
        audio_token_mask: Any | None = None,
        audio_codes: Any | None = None,
        decode_audio: bool = False,
        cache: tuple[Any, ...] | None = None,
    ) -> AudioModelOutput:
        embeddings = self.language_model.embed_tokens(input_ids)
        audio_lengths = None
        if input_features is not None:
            if audio_token_mask is None:
                raise ValueError("audio_token_mask is required with input_features")
            encoded, audio_lengths = self.audio_encoder(input_features)
            features = self.audio_projection(encoded).reshape(-1, embeddings.shape[-1])
            embeddings = replace_audio_features(embeddings, audio_token_mask, features)
        hidden, cache, _ = self.language_model(None, cache=cache, input_embeddings=embeddings)
        text_logits = self.language_model.embed_tokens.as_linear(hidden)
        audio_logits = self.depthformer(hidden, audio_codes)
        waveform = (
            self.detokenizer(audio_codes) if decode_audio and audio_codes is not None else None
        )
        return AudioModelOutput(
            text_logits,
            audio_logits,
            hidden,
            cache=cache,
            audio_lengths=audio_lengths,
            waveform=waveform,
        )


def replace_audio_features(embeddings: Any, mask: Any, features: Any) -> Any:
    count = int(mx.sum(mask).item())
    if count != features.shape[0]:
        raise ValueError(
            f"audio feature count {features.shape[0]} does not match token count {count}"
        )
    indices = mx.maximum(mx.cumsum(mask.reshape(-1).astype(mx.int32)) - 1, 0)
    replacements = features[indices].reshape(embeddings.shape)
    return mx.where(mask[..., None], replacements, embeddings)
