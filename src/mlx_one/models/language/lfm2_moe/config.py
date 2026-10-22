"""Configuration for Liquid LFM2 mixture-of-experts models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_positive
from mlx_one.core.quantization import quantization_config
from mlx_one.models.language.lfm2.config import Lfm2Config


@dataclass(frozen=True)
class Lfm2MoeConfig(Lfm2Config):
    model_type: str = "lfm2_moe"
    intermediate_size: int = 7168
    moe_intermediate_size: int = 1792
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_dense_layers: int = 2
    norm_topk_prob: bool = True
    use_expert_bias: bool = True
    router_aux_loss_coef: float = 0.0
    output_router_logits: bool = False
    routed_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        original = self.model_type
        object.__setattr__(self, "model_type", "lfm2")
        super().__post_init__()
        object.__setattr__(self, "model_type", original)
        if original != "lfm2_moe":
            raise ConfigError(f"Lfm2MoeConfig cannot represent model_type={original!r}")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "conv_L_cache",
            "max_position_embeddings",
            "moe_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "num_dense_layers",
        ):
            if type(getattr(self, name)) is not int:
                raise ConfigError(f"{name} must be an integer")
        require_positive("routed_scaling_factor", self.routed_scaling_factor)
        for name in ("moe_intermediate_size", "num_experts", "num_experts_per_tok"):
            require_positive(name, getattr(self, name))
        if not 0 <= self.num_dense_layers <= self.num_hidden_layers:
            raise ConfigError("num_dense_layers must be within the decoder layer range")
        if self.num_experts_per_tok > self.num_experts:
            raise ConfigError("num_experts_per_tok cannot exceed num_experts")
        if self.router_aux_loss_coef < 0:
            raise ConfigError("router_aux_loss_coef cannot be negative")
        for name in ("norm_topk_prob", "use_expert_bias", "output_router_logits"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")

    @property
    def adjusted_intermediate_size(self) -> int:
        return self.intermediate_size

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2MoeConfig:
        data = dict(data)
        required = {
            "model_type",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "moe_intermediate_size",
            "num_experts",
            "num_experts_per_tok",
            "num_dense_layers",
        }
        missing = required - set(data)
        if missing:
            raise ConfigError(f"missing LFM2 MoE architecture fields: {sorted(missing)}")
        if "tie_embedding" not in data and "tie_word_embeddings" in data:
            data["tie_embedding"] = data["tie_word_embeddings"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        metadata = {
            "architectures",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "dtype",
            "torch_dtype",
            "initializer_range",
            "transformers_version",
            "quantization",
            "quantization_config",
            "tie_word_embeddings",
            "extra",
            "_name_or_path",
        }
        unknown = set(data) - known - metadata
        if unknown:
            raise ConfigError(f"unknown LFM2 MoE configuration fields: {sorted(unknown)}")
        quantization_config(data.get("quantization_config", data.get("quantization")))
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known | {"tie_word_embeddings"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid LFM2 MoE configuration: {exc}") from exc
