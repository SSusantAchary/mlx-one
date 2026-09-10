"""Hugging Face compatible configuration for native Whisper models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.shared.encoder import validate_dropout


@dataclass(frozen=True)
class WhisperConfig:
    model_type: str = "whisper"
    vocab_size: int = 51865
    num_mel_bins: int = 80
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_ffn_dim: int = 1536
    d_model: int = 384
    max_source_positions: int = 1500
    max_target_positions: int = 448
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0
    init_std: float = 0.02
    scale_embedding: bool = False
    tie_word_embeddings: bool = True
    use_cache: bool = True
    is_encoder_decoder: bool = True
    apply_spec_augment: bool = False
    pad_token_id: int = 50257
    bos_token_id: int = 50257
    eos_token_id: int = 50257
    decoder_start_token_id: int = 50258
    suppress_tokens: Sequence[int] | None = None
    begin_suppress_tokens: Sequence[int] = (220, 50257)
    median_filter_width: int = 7
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "whisper":
            raise ConfigError(f"WhisperConfig cannot represent model_type={self.model_type!r}")
        for name in (
            "vocab_size",
            "num_mel_bins",
            "encoder_layers",
            "encoder_attention_heads",
            "decoder_layers",
            "decoder_attention_heads",
            "encoder_ffn_dim",
            "decoder_ffn_dim",
            "d_model",
            "max_source_positions",
            "max_target_positions",
            "median_filter_width",
        ):
            require_positive(name, getattr(self, name))
        if self.num_mel_bins not in {80, 128}:
            raise ConfigError("Whisper num_mel_bins must be 80 or 128")
        require_divisible(
            "d_model", self.d_model, "encoder_attention_heads", self.encoder_attention_heads
        )
        require_divisible(
            "d_model", self.d_model, "decoder_attention_heads", self.decoder_attention_heads
        )
        if self.activation_function != "gelu":
            raise ConfigError("only Whisper GELU activation is supported")
        for name in (
            "dropout",
            "attention_dropout",
            "activation_dropout",
            "encoder_layerdrop",
            "decoder_layerdrop",
        ):
            validate_dropout(name, getattr(self, name))
        if self.encoder_layerdrop or self.decoder_layerdrop:
            raise ConfigError("Whisper LayerDrop is not supported in the inference runtime")
        if self.apply_spec_augment:
            raise ConfigError("Whisper SpecAugment is training-only and not supported")
        if not self.is_encoder_decoder:
            raise ConfigError("Whisper must be configured as an encoder-decoder")
        if not self.tie_word_embeddings:
            raise ConfigError("native Whisper requires tied decoder embeddings")
        if self.median_filter_width % 2 == 0:
            raise ConfigError("median_filter_width must be odd")
        require_positive("init_std", self.init_std)
        for name in (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "decoder_start_token_id",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ConfigError(f"{name} must be within the Whisper vocabulary")
        suppress = None if self.suppress_tokens is None else tuple(self.suppress_tokens)
        begin = tuple(self.begin_suppress_tokens)
        all_suppressed = (*begin, *(suppress or ()))
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in all_suppressed
        ):
            raise ConfigError("suppression token IDs must be integers")
        if any(not 0 <= item < self.vocab_size for item in all_suppressed):
            raise ConfigError("suppression token IDs must be within the vocabulary")
        object.__setattr__(self, "suppress_tokens", suppress)
        object.__setattr__(self, "begin_suppress_tokens", begin)

    @property
    def input_frames(self) -> int:
        return 2 * self.max_source_positions

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WhisperConfig:
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Whisper configuration: {exc}") from exc
