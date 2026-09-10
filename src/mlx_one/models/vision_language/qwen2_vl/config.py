"""Nested configuration for Qwen2-VL."""

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
class Qwen2VLVisionConfig:
    depth: int = 32
    embed_dim: int = 1280
    hidden_size: int = 3584
    hidden_act: str = "quick_gelu"
    mlp_ratio: float = 4.0
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "depth",
            "embed_dim",
            "hidden_size",
            "mlp_ratio",
            "num_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        ):
            require_positive(name, getattr(self, name))
        require_divisible("embed_dim", self.embed_dim, "num_heads", self.num_heads)
        require_divisible(
            "vision head dimension", self.embed_dim // self.num_heads, "rotary groups", 4
        )
        if self.hidden_act not in {"quick_gelu", "gelu", "silu"}:
            raise ConfigError(f"unsupported vision activation: {self.hidden_act}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2VLVisionConfig:
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        return cls(**payload)


@dataclass(frozen=True)
class Qwen2VLTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rope_scaling: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2VLTextConfig:
        data = dict(data)
        if "rope_scaling" not in data and "rope_parameters" in data:
            data["rope_scaling"] = data["rope_parameters"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known)
        return cls(**payload)


@dataclass(frozen=True)
class Qwen2VLConfig:
    model_type: str
    text_config: Qwen2VLTextConfig
    vision_config: Qwen2VLVisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_vl":
            raise ConfigError(f"Qwen2VLConfig cannot represent model_type={self.model_type!r}")
        if self.vision_config.hidden_size != self.text_config.hidden_size:
            raise ConfigError("vision hidden_size must equal text hidden_size")
        token_ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in token_ids
        ):
            raise ConfigError("multimodal token ids must be non-negative integers")
        if len(set(token_ids)) != len(token_ids):
            raise ConfigError("multimodal token ids must be distinct")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2VLConfig:
        known = {
            "model_type",
            "text_config",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "tie_word_embeddings",
        }
        text_data = data.get("text_config")
        if text_data is None:
            text_data = data
        vision_data = data.get("vision_config")
        if not isinstance(text_data, Mapping) or not isinstance(vision_data, Mapping):
            raise ConfigError("Qwen2-VL requires text_config and vision_config objects")
        payload = {key: value for key, value in data.items() if key in known}
        payload["text_config"] = Qwen2VLTextConfig.from_dict(text_data)
        payload["vision_config"] = Qwen2VLVisionConfig.from_dict(vision_data)
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen2-VL configuration: {exc}") from exc
