"""Strict Hugging Face weight contract for Qwen2.5-VL."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.weights import weight_contract as qwen2_weight_contract
from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig


def sanitize_weights(weights: dict[str, Any], config: Qwen2_5_VLConfig) -> dict[str, Any]:
    result = {}
    for name, value in weights.items():
        if (
            name.endswith("rotary_emb.inv_freq")
            or name == "lm_head.weight"
            and config.tie_word_embeddings
        ):
            continue
        if name.startswith("model.language_model."):
            name = name.removeprefix("model.")
        elif name.startswith("model.visual."):
            name = name.removeprefix("model.")
        elif name.startswith("model."):
            name = f"language_model.{name.removeprefix('model.')}"
        name = name.replace("visual.merger.mlp.0.", "visual.merger.mlp_0.")
        name = name.replace("visual.merger.mlp.2.", "visual.merger.mlp_2.")
        result[name] = value
    patch = "visual.patch_embed.proj.weight"
    if patch in result and result[patch].ndim == 5:
        result[patch] = result[patch].reshape(result[patch].shape[0], -1)
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
