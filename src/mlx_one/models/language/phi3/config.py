"""Validated Phi-3 configuration; no implicit context extension."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, require_divisible, require_positive


@dataclass(frozen=True)
class Phi3Config:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    rope_scaling: Mapping[str, Any] | None = None
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    sliding_window: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)

    def __post_init__(self):
        if self.model_type != "phi3":
            raise ConfigError("Phi3Config requires model_type=phi3")
        if self.num_key_value_heads is None:
            object.__setattr__(self, "num_key_value_heads", self.num_attention_heads)
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "original_max_position_embeddings",
        ):
            require_positive(name, getattr(self, name))
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ConfigError(f"Phi3 {name} must be an integer")
        require_divisible("hidden_size", self.hidden_size, "heads", self.num_attention_heads)
        require_divisible("heads", self.num_attention_heads, "KV heads", self.num_key_value_heads)
        for name in ("rms_norm_eps", "rope_theta", "partial_rotary_factor"):
            require_positive(name, getattr(self, name))
        if self.partial_rotary_factor > 1 or self.rotary_dim < 2 or self.rotary_dim % 2:
            raise ConfigError("Phi3 rotary dimensions must be positive, even and within head_dim")
        if self.hidden_act != "silu":
            raise ConfigError("Phi3 requires silu activation")
        if self.sliding_window is not None:
            if not isinstance(self.sliding_window, int) or isinstance(self.sliding_window, bool):
                raise ConfigError("Phi3 sliding_window must be an integer")
            require_positive("sliding_window", self.sliding_window)
        if self.rope_scaling:
            kind = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
            if kind not in {"default", "linear", "longrope", "su"}:
                raise ConfigError(f"unsupported Phi3 RoPE type: {kind}")
            if kind == "linear":
                require_positive("RoPE factor", self.rope_scaling.get("factor"))
            if kind in {"longrope", "su"}:
                for name in ("short_factor", "long_factor"):
                    values = self.rope_scaling.get(name)
                    if not isinstance(values, (list, tuple)) or len(values) != self.rotary_dim // 2:
                        raise ConfigError(f"Phi3 {name} must match rotary_dim / 2")
                    for value in values:
                        require_positive(name, value)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        rope = data.get("rope_parameters")
        if isinstance(rope, Mapping):
            data.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
            data.setdefault("partial_rotary_factor", rope.get("partial_rotary_factor", 1.0))
            data.setdefault("rope_scaling", rope)
        known = set(cls.__dataclass_fields__) - {"extra"}
        try:
            return cls(
                **{k: v for k, v in data.items() if k in known},
                extra={k: v for k, v in data.items() if k not in known},
            )
        except TypeError as exc:
            raise ConfigError(f"invalid Phi3 configuration: {exc}") from exc
