"""Configuration for dense Liquid LFM2 and LFM2.5 models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive


@dataclass(frozen=True)
class Lfm2Config:
    model_type: str = "lfm2"
    vocab_size: int = 65536
    hidden_size: int = 1024
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128000
    norm_eps: float = 1e-5
    conv_bias: bool = False
    conv_L_cache: int = 3
    block_dim: int | None = None
    block_ff_dim: int | None = None
    intermediate_size: int = 6656
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    block_use_swiglu: bool = True
    layer_types: Sequence[str] | None = None
    full_attn_idxs: Sequence[int] | None = None
    rope_theta: float = 1_000_000.0
    rope_parameters: Mapping[str, Any] | None = None
    tie_embedding: bool = True
    use_cache: bool = True
    use_pos_enc: bool = True
    sliding_window: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "lfm2":
            raise ConfigError(f"Lfm2Config cannot represent model_type={self.model_type!r}")
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "conv_L_cache",
            "intermediate_size",
            "block_multiple_of",
        ):
            require_positive(name, getattr(self, name))
        for name in ("norm_eps", "block_ffn_dim_multiplier", "rope_theta"):
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
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ConfigError("attention head dimension must be even for RoPE")
        block_dim = self.hidden_size if self.block_dim is None else self.block_dim
        block_ff_dim = self.intermediate_size if self.block_ff_dim is None else self.block_ff_dim
        if block_dim != self.hidden_size:
            raise ConfigError("block_dim must equal hidden_size")
        require_positive("block_ff_dim", block_ff_dim)
        if not self.block_use_swiglu:
            raise ConfigError("only SwiGLU LFM2 blocks are supported")
        if not self.use_pos_enc:
            raise ConfigError("LFM2 requires rotary positional encoding")
        if self.rope_parameters is not None:
            rope_type = self.rope_parameters.get("rope_type", "default")
            if rope_type != "default":
                raise ConfigError(f"unsupported LFM2 RoPE type: {rope_type}")
            rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
            require_positive("rope_theta", rope_theta)
            object.__setattr__(self, "rope_theta", float(rope_theta))
        layer_types = self.layer_types
        if layer_types is None:
            attention = set(
                self.full_attn_idxs
                if self.full_attn_idxs is not None
                else range(3, self.num_hidden_layers, 4)
            )
            layer_types = tuple(
                "full_attention" if index in attention else "conv"
                for index in range(self.num_hidden_layers)
            )
        else:
            layer_types = tuple(layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ConfigError("layer_types must contain one entry per hidden layer")
        allowed = {"conv", "full_attention", "sliding_attention"}
        unknown = sorted(set(layer_types) - allowed)
        if unknown:
            raise ConfigError(f"unsupported LFM2 layer type: {unknown[0]}")
        if not any(item in {"full_attention", "sliding_attention"} for item in layer_types):
            raise ConfigError("LFM2 requires at least one attention layer")
        if "sliding_attention" in layer_types:
            if self.sliding_window is None:
                raise ConfigError("sliding_attention requires sliding_window")
            require_positive("sliding_window", self.sliding_window)
        derived = tuple(
            index
            for index, layer_type in enumerate(layer_types)
            if layer_type in {"full_attention", "sliding_attention"}
        )
        if self.full_attn_idxs is not None and tuple(self.full_attn_idxs) != derived:
            raise ConfigError("full_attn_idxs must match layer_types")
        for name in (
            "conv_bias",
            "block_auto_adjust_ff_dim",
            "block_use_swiglu",
            "tie_embedding",
            "use_cache",
            "use_pos_enc",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        object.__setattr__(self, "block_dim", block_dim)
        object.__setattr__(self, "block_ff_dim", block_ff_dim)
        object.__setattr__(self, "layer_types", layer_types)
        object.__setattr__(self, "full_attn_idxs", derived)

    @property
    def adjusted_intermediate_size(self) -> int:
        if not self.block_auto_adjust_ff_dim:
            return int(self.block_ff_dim)
        width = int(2 * int(self.block_ff_dim) / 3)
        width = int(self.block_ffn_dim_multiplier * width)
        return self.block_multiple_of * (
            (width + self.block_multiple_of - 1) // self.block_multiple_of
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2Config:
        data = dict(data)
        if "tie_embedding" not in data and "tie_word_embeddings" in data:
            data["tie_embedding"] = data["tie_word_embeddings"]
        if "num_attention_heads" not in data and "num_heads" in data:
            data["num_attention_heads"] = data["num_heads"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known | {"tie_word_embeddings", "num_heads"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid LFM2 configuration: {exc}") from exc
