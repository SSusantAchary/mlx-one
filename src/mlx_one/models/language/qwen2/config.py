"""Configuration for the dense Qwen2/Qwen2.5 architecture."""

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
class Qwen2Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    rope_scaling: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"qwen2", "qwen2_vl_text", "qwen2_5_vl_text"}:
            raise ConfigError(f"Qwen2Config cannot represent model_type={self.model_type!r}")
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
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )
        require_divisible(
            "num_attention_heads",
            self.num_attention_heads,
            "num_key_value_heads",
            self.num_key_value_heads,
        )
        require_positive("rms_norm_eps", self.rms_norm_eps)
        require_positive("rope_theta", self.rope_theta)
        object.__setattr__(self, "rope_scaling", validate_rope_scaling(self.rope_scaling))
        sections = self.mrope_section
        if sections is not None and sum(sections) != self.head_dim // 2:
            raise ConfigError("mrope_section must sum to half the attention head dimension")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mrope_section(self) -> tuple[int, int, int] | None:
        if self.rope_scaling is None:
            return None
        sections = self.rope_scaling.get("mrope_section")
        return tuple(sections) if sections is not None else None

    @property
    def rope_scaling_factor(self) -> float:
        if self.rope_scaling is None:
            return 1.0
        return float(self.rope_scaling.get("factor", 1.0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2Config:
        data = dict(data)
        if "rope_scaling" not in data and "rope_parameters" in data:
            data["rope_scaling"] = data["rope_parameters"]
        known = {
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "rms_norm_eps",
            "max_position_embeddings",
            "rope_theta",
            "tie_word_embeddings",
            "rope_scaling",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen2 configuration: {exc}") from exc
