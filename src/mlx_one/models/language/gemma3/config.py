"""Gemma 3 text configuration with explicit local/global attention layout."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import (
    ConfigError,
    require_divisible,
    require_positive,
    validate_rope_scaling,
)


@dataclass(frozen=True)
class Gemma3TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0
    max_position_embeddings: int = 32768
    sliding_window: int = 512
    sliding_window_pattern: int = 6
    layer_types: tuple[str, ...] | None = None
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10000.0
    rope_scaling: Mapping[str, Any] | None = None
    tie_word_embeddings: bool = True
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.model_type != "gemma3_text":
            raise ConfigError("native text Gemma3 requires model_type=gemma3_text")
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
            "sliding_window",
            "sliding_window_pattern",
            "rms_norm_eps",
            "query_pre_attn_scalar",
            "rope_theta",
            "rope_local_base_freq",
        ):
            require_positive(name, getattr(self, name))
        require_divisible(
            "query heads", self.num_attention_heads, "KV heads", self.num_key_value_heads
        )
        if self.head_dim % 2:
            raise ConfigError("Gemma3 head_dim must be even")
        if self.hidden_activation not in {"gelu_pytorch_tanh", "gelu_new"}:
            raise ConfigError("Gemma3 requires tanh GELU")
        if self.extra.get("attention_bias") or self.extra.get("use_bidirectional_attention"):
            raise ConfigError("native Gemma3 text requires causal, bias-free attention")
        kinds = self.layer_types or tuple(
            "full_attention"
            if (i + 1) % (self.sliding_window_pattern) == 0
            else "sliding_attention"
            for i in range(self.num_hidden_layers)
        )
        if len(kinds) != self.num_hidden_layers or set(kinds) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ConfigError("Gemma3 layer_types must specify every local/global layer")
        object.__setattr__(self, "layer_types", tuple(kinds))
        scaling = validate_rope_scaling(self.rope_scaling)
        if scaling and scaling.get("rope_type", scaling.get("type", "default")) not in {
            "default",
            "linear",
        }:
            raise ConfigError("Gemma3 global RoPE supports default/linear scaling")
        object.__setattr__(self, "rope_scaling", scaling)
        for name in ("final_logit_softcapping", "attn_logit_softcapping"):
            if getattr(self, name) is not None:
                require_positive(name, getattr(self, name))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        rope = data.get("rope_parameters")
        if isinstance(rope, Mapping):
            global_rope = rope.get("full_attention", {})
            local_rope = rope.get("sliding_attention", {})
            data.setdefault("rope_theta", global_rope.get("rope_theta", 1_000_000.0))
            data.setdefault("rope_local_base_freq", local_rope.get("rope_theta", 10000.0))
            data.setdefault("rope_scaling", global_rope)
            if local_rope.get("rope_type", "default") != "default":
                raise ConfigError("Gemma3 local RoPE must be unscaled")
        known = set(cls.__dataclass_fields__) - {"extra"}
        try:
            return cls(
                **{k: v for k, v in data.items() if k in known},
                extra={k: v for k, v in data.items() if k not in known},
            )
        except TypeError as exc:
            raise ConfigError(f"invalid Gemma3 text configuration: {exc}") from exc
