"""LightOnOCR-2: Pixtral vision/projector with the checkpoint's Qwen3 text config."""

from dataclasses import dataclass

from mlx_one.models.language.qwen3.config import Qwen3Config

from .vision_config import VisionConfig


@dataclass
class ModelConfig:
    vision_config: VisionConfig
    text_config: Qwen3Config
    model_type: str = "lightonocr"
    image_token_index: int = 151655
    spatial_merge_size: int = 2
    multimodal_projector_bias: bool = False
    vision_feature_layer: int = -1

    @classmethod
    def from_dict(cls, data):
        return cls(
            VisionConfig.from_dict(data["vision_config"]),
            Qwen3Config.from_dict(data["text_config"]),
            image_token_index=data.get("image_token_id", 151655),
            spatial_merge_size=data.get("spatial_merge_size", 2),
            multimodal_projector_bias=data.get("multimodal_projector_bias", False),
        )
