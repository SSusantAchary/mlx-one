"""Configuration for the dense Llama architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import (
    ConfigError,
    extras,
    require_divisible,
    require_positive,
    validate_rope_scaling,
)


@dataclass(frozen=True)
class LlamaConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 2048
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_traditional: bool = False
    rope_scaling: Mapping[str, Any] | None = None
    bos_token_id: int | None = None
    eos_token_id: int | tuple[int, ...] | None = None
    pad_token_id: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "llama":
            raise ConfigError(f"LlamaConfig cannot represent model_type={self.model_type!r}")
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "max_position_embeddings",
        ):
            require_positive(name, getattr(self, name))
        require_divisible(
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )
        if self.head_dim is None:
            require_divisible(
                "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
            )
            object.__setattr__(self, "head_dim", self.hidden_size // self.num_attention_heads)
        require_positive("head_dim", self.head_dim)
        if self.head_dim % 2:
            raise ConfigError("head_dim must be even for RoPE")
        require_positive("rms_norm_eps", self.rms_norm_eps)
        require_positive("rope_theta", self.rope_theta)
        if self.hidden_act != "silu":
            raise ConfigError("only SwiGLU/silu Llama MLPs are supported")
        if self.attention_bias:
            raise ConfigError("attention bias is not supported by the native Llama architecture")
        if self.mlp_bias:
            raise ConfigError("MLP bias is not supported by the native Llama architecture")
        if self.rope_traditional:
            raise ConfigError("traditional RoPE layout is not supported")
        object.__setattr__(self, "rope_scaling", validate_rope_scaling(self.rope_scaling))
        if (
            self.rope_scaling
            and self.rope_scaling.get("rope_type", self.rope_scaling.get("type")) == "mrope"
        ):
            raise ConfigError("mrope is not supported by the Llama text architecture")
        for name in ("bos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ConfigError(f"{name} must be a non-negative integer or null")
        eos_ids = _token_ids(self.eos_token_id, "eos_token_id")
        normalized_eos = eos_ids[0] if len(eos_ids) == 1 else eos_ids or None
        object.__setattr__(self, "eos_token_id", normalized_eos)

    @property
    def rope_scaling_factor(self) -> float:
        if self.rope_scaling is None:
            return 1.0
        return float(self.rope_scaling.get("factor", 1.0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LlamaConfig:
        data = dict(data)
        if "rope_scaling" not in data and "rope_parameters" in data:
            data["rope_scaling"] = data["rope_parameters"]
        if data.get("num_key_value_heads") is None and "num_attention_heads" in data:
            data["num_key_value_heads"] = data["num_attention_heads"]
        if isinstance(data.get("eos_token_id"), list):
            data["eos_token_id"] = tuple(data["eos_token_id"])
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Llama configuration: {exc}") from exc


def _token_ids(value: object, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, tuple) else (value,)
    if not values or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values
    ):
        raise ConfigError(f"{name} must contain non-negative integers")
    return tuple(values)
