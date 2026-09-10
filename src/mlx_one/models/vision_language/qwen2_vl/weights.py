"""Weight conversion contract for Qwen2-VL."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.weights import weight_contract as qwen2_weight_contract
from mlx_one.models.vision_language.qwen2_vl.config import Qwen2VLConfig


def sanitize_weights(weights: dict[str, Any], config: Qwen2VLConfig) -> dict[str, Any]:
    sanitized = {}
    for name, value in weights.items():
        if name.endswith("rotary_emb.inv_freq"):
            continue
        if name.startswith("model.language_model."):
            name = name.removeprefix("model.")
        elif name.startswith("model."):
            name = f"language_model.{name.removeprefix('model.')}"
        name = name.replace("visual.merger.mlp.0.", "visual.merger.mlp_0.")
        name = name.replace("visual.merger.mlp.2.", "visual.merger.mlp_2.")
        sanitized[name] = value
    patch_name = "visual.patch_embed.proj.weight"
    if patch_name in sanitized and sanitized[patch_name].ndim == 5:
        weight = sanitized[patch_name]
        sanitized[patch_name] = weight.reshape(weight.shape[0], -1)
    return sanitized


def weight_contract(config: Qwen2VLConfig) -> WeightContract:
    vision = config.vision_config
    text = config.text_config
    dense = Qwen2Config(
        model_type="qwen2_vl_text",
        hidden_size=text.hidden_size,
        num_hidden_layers=text.num_hidden_layers,
        intermediate_size=text.intermediate_size,
        num_attention_heads=text.num_attention_heads,
        num_key_value_heads=text.num_key_value_heads,
        vocab_size=text.vocab_size,
        rms_norm_eps=text.rms_norm_eps,
        max_position_embeddings=text.max_position_embeddings,
        rope_theta=text.rope_theta,
        tie_word_embeddings=True,
        rope_scaling=text.rope_scaling,
    )
    expected = {
        f"language_model.{name.removeprefix('model.')}": shape
        for name, shape in qwen2_weight_contract(dense).expected.items()
    }
    expected["visual.patch_embed.proj.weight"] = (
        vision.embed_dim,
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size,
    )
    merged_dim = vision.embed_dim * vision.spatial_merge_size**2
    expected.update(
        {
            "visual.merger.ln_q.weight": (vision.embed_dim,),
            "visual.merger.ln_q.bias": (vision.embed_dim,),
            "visual.merger.mlp_0.weight": (merged_dim, merged_dim),
            "visual.merger.mlp_0.bias": (merged_dim,),
            "visual.merger.mlp_2.weight": (vision.hidden_size, merged_dim),
            "visual.merger.mlp_2.bias": (vision.hidden_size,),
        }
    )
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (text.vocab_size, text.hidden_size)
    mlp_hidden = int(vision.embed_dim * vision.mlp_ratio)
    for index in range(vision.depth):
        prefix = f"visual.blocks.{index}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (vision.embed_dim,),
                f"{prefix}.norm1.bias": (vision.embed_dim,),
                f"{prefix}.norm2.weight": (vision.embed_dim,),
                f"{prefix}.norm2.bias": (vision.embed_dim,),
                f"{prefix}.attn.qkv.weight": (vision.embed_dim * 3, vision.embed_dim),
                f"{prefix}.attn.qkv.bias": (vision.embed_dim * 3,),
                f"{prefix}.attn.proj.weight": (vision.embed_dim, vision.embed_dim),
                f"{prefix}.attn.proj.bias": (vision.embed_dim,),
                f"{prefix}.mlp.fc1.weight": (mlp_hidden, vision.embed_dim),
                f"{prefix}.mlp.fc1.bias": (mlp_hidden,),
                f"{prefix}.mlp.fc2.weight": (vision.embed_dim, mlp_hidden),
                f"{prefix}.mlp.fc2.bias": (vision.embed_dim,),
            }
        )
    return WeightContract(expected)
