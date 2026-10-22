"""Strict GPT-OSS architecture configuration, independent of checkpoint weights."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError
from mlx_one.core.quantization import quantization_config


@dataclass(frozen=True)
class GptOssConfig:
    model_type: str = "gpt_oss"
    num_hidden_layers: int = 24
    num_local_experts: int = 32
    num_experts_per_tok: int = 4
    vocab_size: int = 201088
    rms_norm_eps: float = 1e-5
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    rope_theta: float = 150000.0
    rope_scaling: Mapping[str, Any] | None = None
    layer_types: tuple[str, ...] | None = None
    swiglu_limit: float = 7.0
    max_position_embeddings: int = 131072
    initial_context_length: int = 4096
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    use_cache: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.0
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.model_type != "gpt_oss":
            raise ConfigError(f"GptOssConfig cannot represent model_type={self.model_type!r}")
        for name in (
            "num_hidden_layers",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "sliding_window",
            "max_position_embeddings",
            "initial_context_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ConfigError("head_dim must be even for RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ConfigError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_experts_per_tok > self.num_local_experts:
            raise ConfigError("num_experts_per_tok cannot exceed num_local_experts")
        for name in ("rms_norm_eps", "rope_theta", "swiglu_limit"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or (not math.isfinite(value) or value <= 0)
            ):
                raise ConfigError(f"{name} must be finite and positive")
        if self.rope_theta <= 1:
            raise ConfigError("rope_theta must exceed one")
        for name in ("attention_bias", "tie_word_embeddings", "use_cache", "output_router_logits"):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"{name} must be a boolean")
        if not self.attention_bias or self.tie_word_embeddings or self.hidden_act != "silu":
            raise ConfigError("GPT-OSS requires biased attention, untied head and silu experts")
        if self.attention_dropout != 0:
            raise ConfigError("GPT-OSS inference requires attention_dropout=0")
        schedule = self.layer_types
        if schedule is None:
            schedule = tuple(
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(self.num_hidden_layers)
            )
        if len(schedule) != self.num_hidden_layers or any(
            item not in {"sliding_attention", "full_attention"} for item in schedule
        ):
            raise ConfigError("layer_types requires one valid attention type per layer")
        object.__setattr__(self, "layer_types", tuple(schedule))
        if self.rope_scaling is not None:
            scaling = self.rope_scaling
            if not isinstance(scaling, Mapping) or scaling.get("rope_type", "yarn") != "yarn":
                raise ConfigError("GPT-OSS supports only YaRN rope_scaling")
            unknown = set(scaling) - {
                "rope_type",
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "truncate",
                "attention_factor",
                "mscale",
                "mscale_all_dim",
            }
            if unknown:
                raise ConfigError(f"unsupported YaRN settings: {sorted(unknown)}")
            for name in (
                "factor",
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "attention_factor",
            ):
                value = scaling.get(
                    name,
                    {
                        "factor": 1,
                        "original_max_position_embeddings": self.initial_context_length,
                    }.get(name, 1),
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or (not math.isfinite(value) or value <= 0)
                ):
                    raise ConfigError(f"YaRN {name} must be finite and positive")
            if type(scaling.get("truncate", True)) is not bool:
                raise ConfigError("YaRN truncate must be a boolean")
            if scaling.get("beta_fast", 32) < scaling.get("beta_slow", 1):
                raise ConfigError("YaRN beta_fast must be >= beta_slow")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        required = {
            "model_type",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_local_experts",
        }
        missing = required - set(data)
        if missing:
            raise ConfigError(f"missing GPT-OSS architecture fields: {sorted(missing)}")
        if "experts_per_token" in data:
            if (
                data.get("num_experts_per_tok", data["experts_per_token"])
                != data["experts_per_token"]
            ):
                raise ConfigError("conflicting experts_per_token and num_experts_per_tok")
            data["num_experts_per_tok"] = data.pop("experts_per_token")
        fields = set(cls.__dataclass_fields__) - {"extra"}
        metadata = {
            "architectures",
            "eos_token_id",
            "bos_token_id",
            "pad_token_id",
            "transformers_version",
            "initializer_range",
            "torch_dtype",
            "dtype",
            "quantization",
            "quantization_config",
            "extra",
            "_name_or_path",
        }
        unknown = set(data) - fields - metadata
        if unknown:
            raise ConfigError(f"unknown GPT-OSS configuration fields: {sorted(unknown)}")
        quantization_config(data.get("quantization_config", data.get("quantization")))
        return cls(
            **{key: value for key, value in data.items() if key in fields},
            extra={key: value for key, value in data.items() if key in metadata},
        )
