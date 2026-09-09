"""Configuration and layer-wise dimension expansion for Apple OpenELM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from mlx_one.core.config import ConfigError, extras


def make_divisible(
    value: int | float, divisor: int = 8, minimum: int | float | None = None
) -> int:
    """Round a channel count using OpenELM's MobileNet-compatible rule."""

    if not _is_number(value) or value <= 0:
        raise ConfigError("value must be a positive number")
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
        raise ConfigError("divisor must be a positive integer")
    minimum = divisor if minimum is None else minimum
    if not _is_number(minimum) or minimum <= 0:
        raise ConfigError("minimum must be a positive number")
    rounded = max(minimum, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return int(rounded)


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _expand_multipliers(
    name: str, value: Real | Sequence[Real], layer_count: int
) -> tuple[float, ...]:
    if _is_number(value):
        values = (float(value),) * layer_count
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            start, stop = value
            if not _is_number(start) or not _is_number(stop):
                raise ConfigError(f"{name} values must be numbers")
            if layer_count == 1:
                values = (float(start),)
            else:
                step = (float(stop) - float(start)) / (layer_count - 1)
                values = tuple(
                    round(float(start) + step * index, 2)
                    for index in range(layer_count)
                )
        elif len(value) == layer_count:
            if any(not _is_number(item) for item in value):
                raise ConfigError(f"{name} values must be numbers")
            values = tuple(float(item) for item in value)
        else:
            raise ConfigError(
                f"{name} must be a number, two endpoints, or one value per layer"
            )
    else:
        raise ConfigError(
            f"{name} must be a number, two endpoints, or one value per layer"
        )
    if any(item <= 0 for item in values):
        raise ConfigError(f"{name} values must be positive")
    return values


def _expand_heads(
    name: str, value: int | Sequence[int], layer_count: int
) -> tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value,) * layer_count
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != layer_count:
            raise ConfigError(f"{name} must contain one value per layer")
        values = tuple(value)
    else:
        raise ConfigError(f"{name} must be an integer or one value per layer")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ConfigError(f"{name} values must be positive integers")
    return tuple(values)


@dataclass(frozen=True)
class OpenELMConfig:
    model_type: str = "openelm"
    vocab_size: int = 32000
    max_context_length: int = 2048
    num_transformer_layers: int = 12
    model_dim: int = 2048
    head_dim: int = 128
    qkv_multipliers: Real | Sequence[Real] = 1.0
    num_query_heads: int | Sequence[int] | None = None
    num_kv_heads: int | Sequence[int] | None = None
    num_gqa_groups: int = 1
    ffn_multipliers: Real | Sequence[Real] = 4.0
    ffn_with_glu: bool = True
    ffn_dim_divisor: int = 256
    activation_fn_name: str = "swish"
    normalization_layer_name: str = "rms_norm"
    normalize_qk_projections: bool = False
    share_input_output_layers: bool = False
    rms_norm_eps: float = 1e-6
    rope_freq_constant: float = 10_000.0
    rope_max_length: int = 4096
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "openelm":
            raise ConfigError(
                f"OpenELMConfig cannot represent model_type={self.model_type!r}"
            )
        for name in (
            "vocab_size",
            "max_context_length",
            "num_transformer_layers",
            "model_dim",
            "head_dim",
            "num_gqa_groups",
            "ffn_dim_divisor",
            "rope_max_length",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        for name in (
            "rms_norm_eps",
            "rope_freq_constant",
            "initializer_range",
        ):
            value = getattr(self, name)
            if not _is_number(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive number")
        if self.head_dim % 2:
            raise ConfigError("head_dim must be even for RoPE")
        if self.max_context_length > self.rope_max_length:
            raise ConfigError("max_context_length cannot exceed rope_max_length")
        if self.activation_fn_name not in {"swish", "silu"}:
            raise ConfigError(
                f"unsupported OpenELM activation: {self.activation_fn_name}"
            )
        if self.normalization_layer_name != "rms_norm":
            raise ConfigError(
                f"unsupported OpenELM normalization: {self.normalization_layer_name}"
            )
        for name in (
            "ffn_with_glu",
            "normalize_qk_projections",
            "share_input_output_layers",
            "use_cache",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        for name in ("bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"{name} must be a non-negative integer")

        qkv = _expand_multipliers(
            "qkv_multipliers", self.qkv_multipliers, self.num_transformer_layers
        )
        ffn = _expand_multipliers(
            "ffn_multipliers", self.ffn_multipliers, self.num_transformer_layers
        )
        if self.num_query_heads is None:
            divisor = self.head_dim * self.num_gqa_groups
            query_heads = tuple(
                make_divisible(self.model_dim * multiplier, divisor) // self.head_dim
                for multiplier in qkv
            )
        else:
            query_heads = _expand_heads(
                "num_query_heads",
                self.num_query_heads,
                self.num_transformer_layers,
            )
        if self.num_kv_heads is None:
            if any(heads % self.num_gqa_groups for heads in query_heads):
                raise ConfigError("query heads must be divisible by num_gqa_groups")
            kv_heads = tuple(heads // self.num_gqa_groups for heads in query_heads)
        else:
            kv_heads = _expand_heads(
                "num_kv_heads", self.num_kv_heads, self.num_transformer_layers
            )
        for query, key_value in zip(query_heads, kv_heads, strict=True):
            if query % key_value:
                raise ConfigError("query heads must be divisible by key/value heads")
            if query // key_value != self.num_gqa_groups:
                raise ConfigError(
                    "per-layer query/key-value heads must match num_gqa_groups"
                )

        object.__setattr__(self, "qkv_multipliers", qkv)
        object.__setattr__(self, "ffn_multipliers", ffn)
        object.__setattr__(self, "num_query_heads", query_heads)
        object.__setattr__(self, "num_kv_heads", kv_heads)

    @property
    def ffn_dims(self) -> tuple[int, ...]:
        return tuple(
            make_divisible(self.model_dim * multiplier, self.ffn_dim_divisor)
            for multiplier in self.ffn_multipliers
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OpenELMConfig:
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid OpenELM configuration: {exc}") from exc
