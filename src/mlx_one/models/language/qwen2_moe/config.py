"""Configuration for Qwen2-MoE."""

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
class Qwen2MoeConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    rope_scaling: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_moe":
            raise ConfigError(
                f"Qwen2MoeConfig cannot represent model_type={self.model_type!r}"
            )
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
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
        if self.num_experts_per_tok > self.num_experts:
            raise ConfigError("num_experts_per_tok cannot exceed num_experts")
        if self.router_aux_loss_coef < 0:
            raise ConfigError("router_aux_loss_coef cannot be negative")
        if self.use_sliding_window:
            raise ConfigError("Qwen2-MoE sliding-window attention is not implemented")
        object.__setattr__(self, "rope_scaling", validate_rope_scaling(self.rope_scaling))
        if self.rope_scaling and self.rope_scaling.get(
            "rope_type", self.rope_scaling.get("type")
        ) == "mrope":
            raise ConfigError("mrope is not supported by Qwen2-MoE")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_scaling_factor(self) -> float:
        if self.rope_scaling is None:
            return 1.0
        return float(self.rope_scaling.get("factor", 1.0))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2MoeConfig:
        data = dict(data)
        if "rope_scaling" not in data and "rope_parameters" in data:
            data["rope_scaling"] = data["rope_parameters"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen2-MoE configuration: {exc}") from exc
