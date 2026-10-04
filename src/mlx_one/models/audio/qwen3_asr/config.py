"""Backend-free contracts for the original Qwen3-ASR/ForcedAligner checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlx_one.core.config import ConfigError
from mlx_one.models.language.qwen3.config import Qwen3Config


def positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class QwenAudioConfig:
    d_model: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    output_dim: int
    downsample_hidden_size: int = 480
    num_mel_bins: int = 128
    n_window: int = 50
    n_window_infer: int = 800
    conv_chunksize: int = 500
    max_source_positions: int = 1500

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            positive(name, getattr(self, name))
        if self.d_model % self.encoder_attention_heads or self.d_model % 2:
            raise ConfigError("audio width must be even and divisible by attention heads")
        if self.d_model < 4 or self.num_mel_bins != 128 or self.n_window != 50:
            raise ConfigError(
                "only the original 128-Mel, 100-frame convolution contract is supported"
            )
        if self.n_window_infer < 100 or self.n_window_infer % 100:
            raise ConfigError("n_window_infer must be a multiple of 100")
        if self.max_source_positions < 13:
            raise ConfigError("audio positional table must cover a full convolution chunk")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QwenAudioConfig:
        for name, default in (
            ("activation_function", "gelu"),
            ("scale_embedding", False),
            ("dropout", 0),
            ("attention_dropout", 0),
            ("activation_dropout", 0),
        ):
            if data.get(name, default) != default:
                raise ConfigError(f"unsupported Qwen audio setting: {name}")
        try:
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except TypeError as exc:
            raise ConfigError(f"incomplete Qwen audio configuration: {exc}") from exc


@dataclass(frozen=True)
class QwenASRConfig:
    model_type: str
    audio_config: QwenAudioConfig
    text_config: Qwen3Config
    audio_token_id: int = 151676
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    timestamp_token_id: int = 151705
    timestamp_segment_time: float = 80.0
    classify_num: int = 5000
    support_languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_type not in {"qwen3_asr", "qwen3_forced_aligner"}:
            raise ConfigError("unsupported Qwen audio model type")
        if self.audio_config.output_dim != self.text_config.hidden_size:
            raise ConfigError("Qwen audio projection must match decoder width")
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            positive(name, getattr(self.text_config, name))
        if self.text_config.head_dim % 2:
            raise ConfigError("Qwen audio rotary head width must be even")
        for name in ("rms_norm_eps", "rope_theta"):
            value = getattr(self.text_config, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"{name} must be finite and positive")
        if not isinstance(self.text_config.tie_word_embeddings, bool):
            raise ConfigError("tie_word_embeddings must be a boolean")
        for name in (
            "audio_token_id",
            "audio_start_token_id",
            "audio_end_token_id",
            "timestamp_token_id",
        ):
            positive(name, getattr(self, name))
            if getattr(self, name) >= self.text_config.vocab_size:
                raise ConfigError(f"{name} is outside the vocabulary")
        positive("classify_num", self.classify_num)
        if (
            isinstance(self.timestamp_segment_time, bool)
            or not isinstance(self.timestamp_segment_time, (float, int))
            or not math.isfinite(self.timestamp_segment_time)
            or self.timestamp_segment_time <= 0
        ):
            raise ConfigError("timestamp tick must be finite and positive")
        if any(not isinstance(name, str) or not name for name in self.support_languages):
            raise ConfigError("supported languages must be nonempty strings")
        if self.model_type == "qwen3_forced_aligner" and self.text_config.tie_word_embeddings:
            raise ConfigError("timestamp classification cannot tie the language embedding head")
        if self.model_type == "qwen3_asr" and not self.text_config.tie_word_embeddings:
            raise ConfigError("only original tied-head Qwen ASR checkpoints are supported")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QwenASRConfig:
        if data.get("model_type") != "qwen3_asr" or any(
            data.get(key) is not None for key in ("quantization_config", "quantization")
        ):
            raise ConfigError("expected an original, unquantized qwen3_asr composite checkpoint")
        thinker = data.get("thinker_config")
        if not isinstance(thinker, Mapping):
            raise ConfigError("Qwen audio requires nested thinker_config")
        kind = thinker.get("model_type")
        if kind not in {"qwen3_asr", "qwen3_asr_thinker", "qwen3_forced_aligner"}:
            raise ConfigError("unsupported Qwen thinker type")
        if any(thinker.get(key) is not None for key in ("quantization_config", "quantization")):
            raise ConfigError("quantized Qwen thinker checkpoints are unsupported")
        text = thinker.get("text_config")
        audio = thinker.get("audio_config")
        if not isinstance(text, Mapping) or not isinstance(audio, Mapping):
            raise ConfigError("Qwen thinker requires audio_config and text_config objects")
        text = dict(text)
        if "rope_theta" not in text:
            # The upstream audio decoder and mlx-one's general Qwen3 decoder
            # have different defaults. Never silently choose either default.
            raise ConfigError("Qwen audio text_config must explicitly declare rope_theta")
        if text.get("attention_bias", False) or text.get("hidden_act", "silu") != "silu":
            raise ConfigError("only bias-free Qwen3 SwiGLU text attention is supported")
        if text.get("attention_dropout", 0) or any(
            text.get(key) is not None or audio.get(key) is not None
            for key in ("quantization_config", "quantization")
        ):
            raise ConfigError("unsupported Qwen text attention/quantization")
        rope = text.get("rope_scaling", {})
        if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
            raise ConfigError("only original default Qwen audio RoPE is supported")
        if rope and tuple(rope.get("mrope_section", ())) != (24, 20, 20):
            raise ConfigError("unsupported interleaved MRoPE sections")
        # The pinned ASR get_rope_index repeats the same 1D positions on all three
        # axes. Interleaved MRoPE therefore equals ordinary nontraditional RoPE.
        text["rope_scaling"] = None
        model_type = "qwen3_forced_aligner" if kind == "qwen3_forced_aligner" else "qwen3_asr"
        languages = data.get("support_languages", ())
        if not isinstance(languages, (list, tuple)):
            raise ConfigError("support_languages must be a sequence of language names")
        return cls(
            model_type,
            QwenAudioConfig.from_dict(audio),
            Qwen3Config.from_dict(text),
            **{
                k: thinker.get(k, data.get(k, field.default))
                for k, field in cls.__dataclass_fields__.items()
                if k not in {"model_type", "audio_config", "text_config", "support_languages"}
            },
            support_languages=tuple(languages),
        )


def audio_token_length(frames: int) -> int:
    """Three padded stride-two convolutions, independently per 100-Mel-frame chunk."""
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
        raise ValueError("feature length must be a nonnegative integer")
    return frames // 100 * 13 + ((frames % 100 + 7) // 8)
