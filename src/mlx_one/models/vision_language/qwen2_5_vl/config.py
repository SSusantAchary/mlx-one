"""Strict nested configuration for Qwen2.5-VL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.vision_language.qwen2_vl.config import Qwen2VLTextConfig


@dataclass(frozen=True)
class Qwen2_5_VLVisionConfig:
    model_type: str = "qwen2_5_vl_vision"
    depth: int = 32
    hidden_size: int = 1280
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: int = 4
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: tuple[int, ...] = (7, 15, 23, 31)
    initializer_range: float = 0.02
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl_vision":
            raise ConfigError("invalid Qwen2.5-VL vision model_type")
        for name in (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "tokens_per_second",
            "window_size",
            "out_hidden_size",
        ):
            require_positive(name, getattr(self, name))
        require_divisible("hidden_size", self.hidden_size, "num_heads", self.num_heads)
        require_divisible("vision head dimension", self.head_dim, "rotary groups", 4)
        require_divisible("window_size", self.window_size, "patch_size", self.patch_size)
        if self.hidden_act != "silu":
            raise ConfigError("Qwen2.5-VL vision hidden_act must be silu")
        indexes = tuple(self.fullatt_block_indexes)
        if len(set(indexes)) != len(indexes) or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.depth
            for index in indexes
        ):
            raise ConfigError("fullatt_block_indexes must contain unique valid layer indexes")
        object.__setattr__(self, "fullatt_block_indexes", indexes)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2_5_VLVisionConfig:
        known = set(cls.__dataclass_fields__) - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        if "in_chans" in data and "in_channels" not in payload:
            payload["in_channels"] = data["in_chans"]
        payload["extra"] = extras(data, known | {"in_chans"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen2.5-VL vision configuration: {exc}") from exc


@dataclass(frozen=True)
class Qwen2_5_VLConfig:
    model_type: str
    text_config: Qwen2VLTextConfig
    vision_config: Qwen2_5_VLVisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen2_5_vl":
            raise ConfigError(f"Qwen2_5_VLConfig cannot represent model_type={self.model_type!r}")
        if self.vision_config.out_hidden_size != self.text_config.hidden_size:
            raise ConfigError("vision out_hidden_size must equal text hidden_size")
        token_ids = (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids
        ):
            raise ConfigError("multimodal token ids must be non-negative integers")
        if len(set(token_ids)) != len(token_ids):
            raise ConfigError("multimodal token ids must be distinct")
        if not isinstance(self.tie_word_embeddings, bool):
            raise ConfigError("tie_word_embeddings must be a boolean")
        text_extra = self.text_config.extra
        if text_extra.get("hidden_act", "silu") != "silu":
            raise ConfigError("Qwen2.5-VL text hidden_act must be silu")
        if text_extra.get("attention_dropout", 0.0) != 0.0:
            raise ConfigError("Qwen2.5-VL supports zero-dropout text attention")
        if text_extra.get("use_sliding_window", False):
            raise ConfigError("Qwen2.5-VL sliding-window text attention is not supported")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen2_5_VLConfig:
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
        text_data = data.get("text_config", data)
        vision_data = data.get("vision_config")
        if not isinstance(text_data, Mapping) or not isinstance(vision_data, Mapping):
            raise ConfigError("Qwen2.5-VL requires text_config and vision_config objects")
        text_data = dict(text_data)
        text_data.setdefault("rope_scaling", text_data.get("rope_parameters"))
        payload = {key: value for key, value in data.items() if key in known}
        payload["text_config"] = Qwen2VLTextConfig.from_dict(text_data)
        payload["vision_config"] = Qwen2_5_VLVisionConfig.from_dict(vision_data)
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen2.5-VL configuration: {exc}") from exc
