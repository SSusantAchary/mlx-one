"""Backend-free contract for the original float32 Parakeet TDT exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError


@dataclass(frozen=True)
class ParakeetConfig:
    model_type: str = "parakeet_tdt"
    vocab_size: int = 8193
    blank_token_id: int = 8192
    pad_token_id: int = 2
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    durations: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_symbols_per_step: int = 10
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_mel_bins: int = 128
    conv_kernel_size: int = 9
    subsampling_conv_channels: int = 256
    attention_bias: bool = False
    convolution_bias: bool = False
    scale_input: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "parakeet_tdt":
            raise ConfigError("only offline Parakeet TDT is implemented")
        for name in (
            "vocab_size",
            "decoder_hidden_size",
            "num_decoder_layers",
            "max_symbols_per_step",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_mel_bins",
            "conv_kernel_size",
            "subsampling_conv_channels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads or self.hidden_size % 2:
            raise ConfigError("Parakeet hidden size must be even and divisible by heads")
        if self.num_mel_bins % 8 or self.conv_kernel_size % 2 != 1:
            raise ConfigError("Parakeet requires Mel bins divisible by 8 and an odd conv kernel")
        if tuple(self.durations) != (0, 1, 2, 3, 4) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.durations
        ):
            raise ConfigError("only the pinned TDT durations [0, 1, 2, 3, 4] are supported")
        for name in ("pad_token_id", "blank_token_id"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ConfigError(f"{name} must be a vocabulary ID")
        if self.blank_token_id != self.vocab_size - 1:
            raise ConfigError("TDT requires a final blank vocabulary ID")
        for name in ("attention_bias", "convolution_bias", "scale_input"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParakeetConfig:
        if data.get("quantization_config") is not None or data.get("quantization") is not None:
            raise ConfigError("quantized Parakeet variants are not supported")
        if data.get("model_type") != "parakeet_tdt":
            raise ConfigError("Parakeet config requires model_type=parakeet_tdt")
        enc = data.get("encoder_config")
        if not isinstance(enc, Mapping) or enc.get("model_type") != "parakeet_encoder":
            raise ConfigError("Parakeet TDT requires a parakeet_encoder configuration")
        for key, expected in (
            ("subsampling_factor", 8),
            ("subsampling_conv_kernel_size", 3),
            ("subsampling_conv_stride", 2),
            ("hidden_act", "silu"),
        ):
            if enc.get(key, expected) != expected:
                raise ConfigError(f"unsupported Parakeet encoder setting: {key}")
        if data.get("hidden_act", "relu") != "relu":
            raise ConfigError("Parakeet TDT joint activation must be relu")
        if data.get("is_encoder_decoder", True) is not True or data.get(
            "tie_word_embeddings", False
        ):
            raise ConfigError("unsupported tied/non-encoder-decoder TDT configuration")
        if enc.get("num_key_value_heads", enc.get("num_attention_heads", 8)) != enc.get(
            "num_attention_heads", 8
        ):
            raise ConfigError("Parakeet uses ordinary multi-head attention")
        known = set(cls.__dataclass_fields__) - {"extra"}
        fields = {key: data[key] for key in known if key in data}
        fields.update({key: enc[key] for key in known if key in enc and key != "model_type"})
        durations = data.get("durations", (0, 1, 2, 3, 4))
        if not isinstance(durations, (list, tuple)):
            raise ConfigError("Parakeet durations must be a list of integer frame counts")
        fields["durations"] = tuple(durations)
        fields["extra"] = dict(data)
        return cls(**fields)


def encoded_length(frames: int) -> int:
    for _ in range(3):
        frames = (frames + 1) // 2
    return frames
