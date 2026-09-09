"""Configuration for the dense Qwen3 architecture."""

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
class Qwen3Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    rope_scaling: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen3":
            raise ConfigError(f"Qwen3Config cannot represent model_type={self.model_type!r}")
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
            require_positive(name, getattr(self, name))
        require_divisible(
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )
        require_positive("rms_norm_eps", self.rms_norm_eps)
        require_positive("rope_theta", self.rope_theta)
        object.__setattr__(self, "rope_scaling", validate_rope_scaling(self.rope_scaling))
        if self.rope_scaling and self.rope_scaling.get(
            "rope_type", self.rope_scaling.get("type")
        ) == "mrope":
            raise ConfigError("mrope is not supported by the Qwen3 text architecture")

    @property
    def rope_scaling_factor(self) -> float:
        if self.rope_scaling is None:
            return 1.0
        return float(self.rope_scaling.get("factor", 1.0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3Config:
        data = dict(data)
        if "rope_scaling" not in data and "rope_parameters" in data:
            data["rope_scaling"] = data["rope_parameters"]
        known = {field.name for field in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3 configuration: {exc}") from exc
