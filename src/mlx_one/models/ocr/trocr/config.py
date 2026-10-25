"""Validated configs for the ViT / TrOCR printed and handwritten line checkpoints."""

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class TrOCRConfig:
    encoder: object
    decoder: object
    model_type: str = "trocr"

    @classmethod
    def from_dict(cls, data):
        e = dict(data["encoder"])
        d = dict(data["decoder"])
        if e.get("model_type") != "vit" or d.get("model_type") != "trocr":
            raise ValueError("TrOCR requires the published ViT encoder and TrOCR decoder")
        for cfg, dim, heads in [
            (e, "hidden_size", "num_attention_heads"),
            (d, "d_model", "decoder_attention_heads"),
        ]:
            if cfg[dim] < 1 or cfg[heads] < 1 or cfg[dim] % cfg[heads]:
                raise ValueError("invalid TrOCR attention dimensions")
        if e["image_size"] % e["patch_size"]:
            raise ValueError("TrOCR image size must be divisible by patch size")
        if not d.get("use_learned_position_embeddings", True):
            raise ValueError("sinusoidal TrOCR variants are not implemented")
        d.setdefault("layernorm_embedding", True)
        d.setdefault("scale_embedding", False)
        d.setdefault("decoder_start_token_id", 2)
        d.setdefault("tie_word_embeddings", True)
        e.setdefault("qkv_bias", True)
        return cls(SimpleNamespace(**e), SimpleNamespace(**d))
