"""Backend-free configuration for fixed-resolution supervised ViT."""

from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError
from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig


@dataclass(frozen=True)
class ViTConfig:
    backbone: BertEmbeddingConfig
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_labels: int = 1000
    id2label: dict[str, str] = field(default_factory=dict)
    model_type: str = "vit"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backbone, name)

    @classmethod
    def from_dict(cls, data):
        if data.get("model_type") != "vit" or data.get("qkv_bias", True) is not True:
            raise ConfigError("ViT currently supports biased QKV supervised checkpoints")
        image, patch = data.get("image_size", 224), data.get("patch_size", 16)
        if any(type(v) is not int or v < 1 for v in (image, patch)) or image % patch:
            raise ConfigError("ViT image size must be divisible by a positive patch size")
        if data.get("num_channels", 3) != 3:
            raise ConfigError("ViT classification currently requires RGB input")
        labels = data.get("id2label", {})
        count = data.get("num_labels", len(labels) or 1000)
        if (
            type(count) is not int
            or count < 1
            or (labels and set(labels) != {str(i) for i in range(count)})
        ):
            raise ConfigError("ViT labels must form a complete classifier index")
        payload = {
            **data,
            "model_type": "bert",
            "vocab_size": 1,
            "pad_token_id": 0,
            "sentence_max_length": 1,
            "max_position_embeddings": 1,
        }
        return cls(BertEmbeddingConfig.from_dict(payload), image, patch, 3, count, labels)
