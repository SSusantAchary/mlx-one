"""Strict Hugging Face weight contract for Qwen2.5-VL."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.weights import weight_contract as qwen2_weight_contract
from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig


def sanitize_weights(weights: dict[str, Any], config: Qwen2_5_VLConfig) -> dict[str, Any]:
    from mlx_one.models.shared.vlm_weights import canonical_name, patch_weight

    result = {}
    for original, value in weights.items():
        if original.endswith(("rotary_emb.inv_freq", "position_ids")):
            continue
        name = canonical_name(original, "qwen2_5_vl")
        if config.tie_word_embeddings and name.startswith("lm_head."):
            continue
        if name == "visual.patch_embed.proj.weight":
            value = patch_weight(value, config.vision_config.in_channels)
        if name in result:
            raise ValueError(f"duplicate VLM tensor mapping: {name}")
        result[name] = value
    return result


def weight_contract(config: Qwen2_5_VLConfig) -> WeightContract:
    text = config.text_config
    vision = config.vision_config
    dense = Qwen2Config(
        model_type="qwen2_5_vl_text",
        hidden_size=text.hidden_size,
        num_hidden_layers=text.num_hidden_layers,
        intermediate_size=text.intermediate_size,
        num_attention_heads=text.num_attention_heads,
        num_key_value_heads=text.num_key_value_heads,
        vocab_size=text.vocab_size,
        rms_norm_eps=text.rms_norm_eps,
        max_position_embeddings=text.max_position_embeddings,
        rope_theta=text.rope_theta,
        rope_scaling=text.rope_scaling,
        tie_word_embeddings=True,
    )
    expected = {
        f"language_model.{name.removeprefix('model.')}": shape
        for name, shape in qwen2_weight_contract(dense).expected.items()
    }
    patch_dim = (
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size
    )
    merged = vision.hidden_size * vision.spatial_merge_size**2
    expected.update(
        {
            "visual.patch_embed.proj.weight": (vision.hidden_size, patch_dim),
            "visual.merger.ln_q.weight": (vision.hidden_size,),
            "visual.merger.mlp_0.weight": (merged, merged),
            "visual.merger.mlp_0.bias": (merged,),
            "visual.merger.mlp_2.weight": (vision.out_hidden_size, merged),
            "visual.merger.mlp_2.bias": (vision.out_hidden_size,),
        }
    )
    for index in range(vision.depth):
        prefix = f"visual.blocks.{index}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (vision.hidden_size,),
                f"{prefix}.norm2.weight": (vision.hidden_size,),
                f"{prefix}.attn.qkv.weight": (vision.hidden_size * 3, vision.hidden_size),
                f"{prefix}.attn.qkv.bias": (vision.hidden_size * 3,),
                f"{prefix}.attn.proj.weight": (vision.hidden_size, vision.hidden_size),
                f"{prefix}.attn.proj.bias": (vision.hidden_size,),
                f"{prefix}.mlp.gate_proj.weight": (vision.intermediate_size, vision.hidden_size),
                f"{prefix}.mlp.gate_proj.bias": (vision.intermediate_size,),
                f"{prefix}.mlp.up_proj.weight": (vision.intermediate_size, vision.hidden_size),
                f"{prefix}.mlp.up_proj.bias": (vision.intermediate_size,),
                f"{prefix}.mlp.down_proj.weight": (vision.hidden_size, vision.intermediate_size),
                f"{prefix}.mlp.down_proj.bias": (vision.hidden_size,),
            }
        )
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (text.vocab_size, text.hidden_size)
    return WeightContract(expected)
