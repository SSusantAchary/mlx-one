"""Backend-free config for the pinned original CLIP checkpoint."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CLIPTowerConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_hidden_layers: int = 12
    layer_norm_eps: float = 1e-5
    hidden_act: str = "quick_gelu"
    max_position_embeddings: int = 77
    vocab_size: int = 49408
    eos_token_id: int = 2
    image_size: int = 224
    patch_size: int = 32

    @classmethod
    def from_dict(cls, payload):
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class CLIPConfig:
    model_type: str = "clip"
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    text_config: CLIPTowerConfig = field(default_factory=CLIPTowerConfig)
    vision_config: CLIPTowerConfig = field(
        default_factory=lambda: CLIPTowerConfig(
            hidden_size=768, intermediate_size=3072, num_attention_heads=12
        )
    )

    @classmethod
    def from_dict(cls, payload):
        if payload.get("model_type") != "clip":
            raise ValueError("expected the pinned CLIP architecture")
        return cls(
            projection_dim=payload["projection_dim"],
            logit_scale_init_value=payload["logit_scale_init_value"],
            text_config=CLIPTowerConfig.from_dict(payload["text_config"]),
            vision_config=CLIPTowerConfig.from_dict(payload["vision_config"]),
        )
