"""Strict Hugging Face tensor contract for dense Qwen3.5 multimodal models."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.vision_language.qwen3_5.config import Qwen3_5Config


def sanitize_weights(weights: dict[str, Any], config: Qwen3_5Config) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for original, value in weights.items():
        name = original
        if name.endswith(("rotary_emb.inv_freq", "position_ids")):
            continue
        if name == "lm_head.weight" and config.tie_word_embeddings:
            continue
        if name.startswith("model."):
            name = name.removeprefix("model.")
        if name.endswith("linear_attn.conv1d.weight") and value.ndim == 3:
            value = value.squeeze(1)
        if name == "visual.patch_embed.proj.weight" and value.ndim == 5:
            value = value.reshape(value.shape[0], -1)
        sanitized[name] = value
    return sanitized


def weight_contract(config: Qwen3_5Config) -> WeightContract:
    text = config.text_config
    vision = config.vision_config
    expected: dict[str, tuple[int, ...]] = {
        "language_model.embed_tokens.weight": (text.vocab_size, text.hidden_size),
        "language_model.norm.weight": (text.hidden_size,),
        "visual.patch_embed.proj.weight": (
            vision.hidden_size,
            vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size,
        ),
        "visual.patch_embed.proj.bias": (vision.hidden_size,),
        "visual.pos_embed.weight": (vision.num_position_embeddings, vision.hidden_size),
    }
    for index, kind in enumerate(text.layer_types):
        prefix = f"language_model.layers.{index}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (text.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (text.hidden_size,),
                f"{prefix}.mlp.gate_proj.weight": (text.intermediate_size, text.hidden_size),
                f"{prefix}.mlp.up_proj.weight": (text.intermediate_size, text.hidden_size),
                f"{prefix}.mlp.down_proj.weight": (text.hidden_size, text.intermediate_size),
            }
        )
        if kind == "full_attention":
            expected.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": (
                        text.num_attention_heads * text.head_dim * 2,
                        text.hidden_size,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": (
                        text.num_key_value_heads * text.head_dim,
                        text.hidden_size,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": (
                        text.num_key_value_heads * text.head_dim,
                        text.hidden_size,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": (
                        text.hidden_size,
                        text.num_attention_heads * text.head_dim,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": (text.head_dim,),
                    f"{prefix}.self_attn.k_norm.weight": (text.head_dim,),
                }
            )
        else:
            key_dim = text.linear_num_key_heads * text.linear_key_head_dim
            value_dim = text.linear_num_value_heads * text.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            linear = f"{prefix}.linear_attn"
            expected.update(
                {
                    f"{linear}.in_proj_qkv.weight": (conv_dim, text.hidden_size),
                    f"{linear}.in_proj_z.weight": (value_dim, text.hidden_size),
                    f"{linear}.in_proj_b.weight": (
                        text.linear_num_value_heads,
                        text.hidden_size,
                    ),
                    f"{linear}.in_proj_a.weight": (
                        text.linear_num_value_heads,
                        text.hidden_size,
                    ),
                    f"{linear}.conv1d.weight": (conv_dim, text.linear_conv_kernel_dim),
                    f"{linear}.dt_bias": (text.linear_num_value_heads,),
                    f"{linear}.A_log": (text.linear_num_value_heads,),
                    f"{linear}.norm.weight": (text.linear_value_head_dim,),
                    f"{linear}.out_proj.weight": (text.hidden_size, value_dim),
                }
            )
    for index in range(vision.depth):
        prefix = f"visual.blocks.{index}"
        expected.update(
            {
                f"{prefix}.norm1.weight": (vision.hidden_size,),
                f"{prefix}.norm1.bias": (vision.hidden_size,),
                f"{prefix}.norm2.weight": (vision.hidden_size,),
                f"{prefix}.norm2.bias": (vision.hidden_size,),
                f"{prefix}.attn.qkv.weight": (vision.hidden_size * 3, vision.hidden_size),
                f"{prefix}.attn.qkv.bias": (vision.hidden_size * 3,),
                f"{prefix}.attn.proj.weight": (vision.hidden_size, vision.hidden_size),
                f"{prefix}.attn.proj.bias": (vision.hidden_size,),
                f"{prefix}.mlp.linear_fc1.weight": (
                    vision.intermediate_size,
                    vision.hidden_size,
                ),
                f"{prefix}.mlp.linear_fc1.bias": (vision.intermediate_size,),
                f"{prefix}.mlp.linear_fc2.weight": (
                    vision.hidden_size,
                    vision.intermediate_size,
                ),
                f"{prefix}.mlp.linear_fc2.bias": (vision.hidden_size,),
            }
        )
    merged = vision.hidden_size * vision.spatial_merge_size**2
    expected.update(
        {
            "visual.merger.norm.weight": (vision.hidden_size,),
            "visual.merger.norm.bias": (vision.hidden_size,),
            "visual.merger.linear_fc1.weight": (merged, merged),
            "visual.merger.linear_fc1.bias": (merged,),
            "visual.merger.linear_fc2.weight": (vision.out_hidden_size, merged),
            "visual.merger.linear_fc2.bias": (vision.out_hidden_size,),
        }
    )
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (text.vocab_size, text.hidden_size)
    mtp: set[str] = {
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    expected.update(
        {
            "mtp.fc.weight": (text.hidden_size, text.hidden_size * 2),
            "mtp.norm.weight": (text.hidden_size,),
            "mtp.pre_fc_norm_embedding.weight": (text.hidden_size,),
            "mtp.pre_fc_norm_hidden.weight": (text.hidden_size,),
        }
    )
    for index in range(text.mtp_num_hidden_layers):
        prefix = f"mtp.layers.{index}"
        layer = {
            f"{prefix}.input_layernorm.weight": (text.hidden_size,),
            f"{prefix}.post_attention_layernorm.weight": (text.hidden_size,),
            f"{prefix}.mlp.gate_proj.weight": (text.intermediate_size, text.hidden_size),
            f"{prefix}.mlp.up_proj.weight": (text.intermediate_size, text.hidden_size),
            f"{prefix}.mlp.down_proj.weight": (text.hidden_size, text.intermediate_size),
            f"{prefix}.self_attn.q_proj.weight": (
                text.num_attention_heads * text.head_dim * 2,
                text.hidden_size,
            ),
            f"{prefix}.self_attn.k_proj.weight": (
                text.num_key_value_heads * text.head_dim,
                text.hidden_size,
            ),
            f"{prefix}.self_attn.v_proj.weight": (
                text.num_key_value_heads * text.head_dim,
                text.hidden_size,
            ),
            f"{prefix}.self_attn.o_proj.weight": (
                text.hidden_size,
                text.num_attention_heads * text.head_dim,
            ),
            f"{prefix}.self_attn.q_norm.weight": (text.head_dim,),
            f"{prefix}.self_attn.k_norm.weight": (text.head_dim,),
        }
        expected.update(layer)
        mtp.update(layer)
    return WeightContract(expected, optional=frozenset(mtp))
