"""Backend-free contract for the original Moonshine Tiny/Base architecture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible


@dataclass(frozen=True)
class MoonshineConfig:
    model_type: str = "moonshine"
    vocab_size: int = 32768
    hidden_size: int = 288
    intermediate_size: int = 1152
    encoder_num_hidden_layers: int = 6
    decoder_num_hidden_layers: int = 6
    encoder_num_attention_heads: int = 8
    decoder_num_attention_heads: int = 8
    encoder_num_key_value_heads: int | None = None
    decoder_num_key_value_heads: int | None = None
    partial_rotary_factor: float = 0.9
    rope_theta: float = 10000.0
    rope_scaling: Mapping[str, Any] | None = None
    pad_head_dim_to_multiple_of: int | None = 8
    encoder_hidden_act: str = "gelu"
    decoder_hidden_act: str = "silu"
    max_position_embeddings: int = 194
    attention_bias: bool = False
    attention_dropout: float = 0.0
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True
    decoder_start_token_id: int = 1
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 2
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "moonshine":
            raise ConfigError("MoonshineConfig requires model_type=moonshine")
        if (
            isinstance(self.partial_rotary_factor, bool)
            or not math.isfinite(self.partial_rotary_factor)
            or not 0 < self.partial_rotary_factor <= 1
        ):
            raise ConfigError("partial_rotary_factor must be in (0, 1]")
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "encoder_num_hidden_layers",
            "decoder_num_hidden_layers",
            "encoder_num_attention_heads",
            "decoder_num_attention_heads",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigError(f"{name} must be a positive integer")
        for side in ("encoder", "decoder"):
            heads = getattr(self, f"{side}_num_attention_heads")
            require_divisible("hidden_size", self.hidden_size, "attention heads", heads)
            kv = getattr(self, f"{side}_num_key_value_heads")
            if kv is None:
                object.__setattr__(self, f"{side}_num_key_value_heads", heads)
            elif isinstance(kv, bool) or kv != heads:
                raise ConfigError("only original Moonshine multi-head attention is supported")
            rotary = int(self.hidden_size // heads * self.partial_rotary_factor)
            if rotary < 2 or rotary % 2:
                raise ConfigError("Moonshine rotary dimension must be positive and even")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ConfigError("rope_theta must be finite and positive")
        if self.rope_scaling is not None:
            raise ConfigError("scaled Moonshine RoPE is not supported")
        if (self.encoder_hidden_act, self.decoder_hidden_act) != ("gelu", "silu"):
            raise ConfigError("Moonshine requires GELU encoder and SiLU gated decoder")
        if self.attention_bias or self.attention_dropout:
            raise ConfigError("only bias-free, zero-dropout Moonshine attention is supported")
        if self.is_encoder_decoder is not True or self.tie_word_embeddings is not True:
            raise ConfigError("Moonshine requires a tied encoder-decoder checkpoint")
        multiple = self.pad_head_dim_to_multiple_of
        if multiple is not None and (
            isinstance(multiple, bool) or not isinstance(multiple, int) or multiple < 1
        ):
            raise ConfigError("head padding multiple must be a positive integer")
        for name in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ConfigError(f"{name} must be a vocabulary ID")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MoonshineConfig:
        payload = dict(data)
        if payload.get("quantization_config") is not None:
            raise ConfigError("quantized Moonshine checkpoints are not supported")
        rope = payload.pop("rope_parameters", None)
        if rope is not None:
            if not isinstance(rope, Mapping) or rope.get("rope_type", "default") != "default":
                raise ConfigError("only default Moonshine RoPE is supported")
            for name in ("rope_theta", "partial_rotary_factor"):
                if name in rope:
                    payload[name] = rope[name]
        known = set(cls.__dataclass_fields__) - {"extra"}
        try:
            return cls(
                **{k: v for k, v in payload.items() if k in known}, extra=extras(payload, known)
            )
        except (TypeError, OverflowError) as exc:
            raise ConfigError(f"invalid Moonshine configuration: {exc}") from exc


def encoded_length(samples: int) -> int:
    """Valid, unpadded 127/7/3 convolutions with strides 64/3/2."""
    for kernel, stride in ((127, 64), (7, 3), (3, 2)):
        samples = (samples - kernel) // stride + 1
    return max(0, samples)
