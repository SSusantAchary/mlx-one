"""Strict nested weight contract for LFM2-VL."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.weights import weight_contract as text_contract
from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig


def sanitize_weights(weights: dict[str, Any], config: Lfm2VLConfig) -> dict[str, Any]:
    cleaned = {}
    for name, value in weights.items():
        if name.endswith(("position_ids", "rotary_emb.inv_freq")):
            continue
        tied_heads = {"lm_head.weight", "language_model.lm_head.weight"}
        if config.tie_word_embeddings and name in tied_heads:
            continue
        cleaned[name] = value
    return cleaned


def weight_contract(config: Lfm2VLConfig) -> WeightContract:
    expected = {
        f"language_model.{name.removeprefix('model.')}": shape
        for name, shape in text_contract(config.text_config).expected.items()
        if name != "lm_head.weight"
    }
    vision = config.vision_config
    patch_dim = vision.num_channels * vision.patch_size**2
    grid = vision.image_size // vision.patch_size
    expected.update(
        {
            "vision_model.patch_embedding.weight": (vision.hidden_size, patch_dim),
            "vision_model.patch_embedding.bias": (vision.hidden_size,),
            "vision_model.position_embedding.weight": (grid * grid, vision.hidden_size),
            "vision_model.post_layernorm.weight": (vision.hidden_size,),
            "vision_model.post_layernorm.bias": (vision.hidden_size,),
        }
    )
    merged = vision.hidden_size * vision.downsample_factor**2
    if vision.use_projection_layernorm:
        expected["vision_model.projector_norm.weight"] = (merged,)
        expected["vision_model.projector_norm.bias"] = (merged,)
    expected["vision_model.projector.weight"] = (vision.projection_dim, merged)
    expected["vision_model.projector.bias"] = (vision.projection_dim,)
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (config.text_config.vocab_size, config.text_config.hidden_size)
    for index in range(vision.num_hidden_layers):
        prefix = f"vision_model.blocks.{index}"
        for norm in ("layer_norm1", "layer_norm2"):
            expected[f"{prefix}.{norm}.weight"] = (vision.hidden_size,)
            expected[f"{prefix}.{norm}.bias"] = (vision.hidden_size,)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            expected[f"{prefix}.self_attn.{projection}.weight"] = (
                vision.hidden_size,
                vision.hidden_size,
            )
            expected[f"{prefix}.self_attn.{projection}.bias"] = (vision.hidden_size,)
        expected[f"{prefix}.fc1.weight"] = (vision.intermediate_size, vision.hidden_size)
        expected[f"{prefix}.fc1.bias"] = (vision.intermediate_size,)
        expected[f"{prefix}.fc2.weight"] = (vision.hidden_size, vision.intermediate_size)
        expected[f"{prefix}.fc2.bias"] = (vision.hidden_size,)
    return WeightContract(expected)
