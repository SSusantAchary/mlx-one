"""Nested configuration for Liquid LFM2-VL and LFM2.5-VL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.language.lfm2.config import Lfm2Config


@dataclass(frozen=True)
class Lfm2VisionConfig:
    model_type: str = "mlx_one_legacy"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    patch_size: int = 16
    image_size: int = 384
    downsample_factor: int = 2
    projection_dim: int = 1024
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu"
    use_projection_layernorm: bool = True
    num_patches: int | None = None
    projector_hidden_size: int | None = None
    projector_bias: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_channels",
            "patch_size",
            "image_size",
            "downsample_factor",
            "projection_dim",
            "layer_norm_eps",
        ):
            require_positive(name, getattr(self, name))
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )
        require_divisible("image_size", self.image_size, "patch_size", self.patch_size)
        require_divisible(
            "patch grid",
            self.image_size // self.patch_size,
            "downsample_factor",
            self.downsample_factor,
        )
        if self.hidden_act not in {"gelu", "quick_gelu", "gelu_pytorch_tanh"}:
            raise ConfigError(f"unsupported LFM vision activation: {self.hidden_act}")
        if self.model_type not in {"mlx_one_legacy", "lfm2_vl", "siglip2_vision_model"}:
            raise ConfigError("unsupported LFM vision tower")
        if self.num_patches is not None and (
            self.num_patches < 1 or int(self.num_patches**0.5) ** 2 != self.num_patches
        ):
            raise ConfigError("SigLIP2 position table must be a positive square")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2VisionConfig:
        data = dict(data)
        aliases = {"embed_dim": "hidden_size", "num_heads": "num_attention_heads"}
        for source, target in aliases.items():
            if target not in data and source in data:
                data[target] = data[source]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known | set(aliases))
        return cls(**payload)


@dataclass(frozen=True)
class Lfm2VLConfig:
    model_type: str
    text_config: Lfm2Config
    vision_config: Lfm2VisionConfig
    image_token_id: int
    tie_word_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"lfm2_vl", "lfm2-vl"}:
            raise ConfigError(f"Lfm2VLConfig cannot represent model_type={self.model_type!r}")
        if self.vision_config.projection_dim != self.text_config.hidden_size:
            raise ConfigError("vision projection_dim must equal text hidden_size")
        if isinstance(self.image_token_id, bool) or not isinstance(self.image_token_id, int):
            raise ConfigError("image_token_id must be a non-negative integer")
        if self.image_token_id < 0:
            raise ConfigError("image_token_id must be a non-negative integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2VLConfig:
        text = data.get("text_config")
        vision = data.get("vision_config")
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ConfigError("LFM2-VL requires text_config and vision_config objects")
        text = dict(text)
        text["model_type"] = "lfm2"
        known = {
            "model_type",
            "text_config",
            "vision_config",
            "image_token_id",
            "tie_word_embeddings",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = str(data.get("model_type", "lfm2_vl")).replace("-", "_")
        payload["text_config"] = Lfm2Config.from_dict(text)
        vision = dict(vision)
        if vision.get("model_type") == "siglip2_vision_model":
            vision.setdefault(
                "image_size",
                int(vision.get("num_patches", 256) ** 0.5) * vision.get("patch_size", 16),
            )
            vision["downsample_factor"] = data.get("downsample_factor", 2)
            vision["projection_dim"] = payload["text_config"].hidden_size
            vision["projector_hidden_size"] = data.get("projector_hidden_size", 2560)
            vision["use_projection_layernorm"] = data.get("projector_use_layernorm", True)
            vision["projector_bias"] = data.get("projector_bias", True)
        payload["image_token_id"] = data.get("image_token_id", data.get("image_token_index"))
        payload["vision_config"] = Lfm2VisionConfig.from_dict(vision)
        payload["extra"] = extras(data, known)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid LFM2-VL configuration: {exc}") from exc
