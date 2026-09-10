"""Strict Hugging Face weight contract for dense LFM2."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.config import Lfm2Config


def sanitize_weights(weights: dict[str, Any], config: Lfm2Config) -> dict[str, Any]:
    sanitized = {}
    for name, value in weights.items():
        if name.endswith(("rotary_emb.inv_freq", "position_ids")):
            continue
        if config.tie_embedding and name == "lm_head.weight":
            continue
        if name.endswith("conv.conv.weight"):
            name = name.removesuffix(".weight")
        if name.endswith("conv.conv") and len(value.shape) == 3:
            if value.shape[-1] == 1:
                value = value[:, :, 0]
            elif value.shape[1] == 1:
                value = value[:, 0, :]
        sanitized[name] = value
    return sanitized


def weight_contract(config: Lfm2Config) -> WeightContract:
    dim = config.hidden_size
    head_dim = dim // config.num_attention_heads
    kv_dim = config.num_key_value_heads * head_dim
    ff = config.adjusted_intermediate_size
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, dim),
        "model.embedding_norm.weight": (dim,),
    }
    if not config.tie_embedding:
        expected["lm_head.weight"] = (config.vocab_size, dim)
    for index, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{index}"
        expected[f"{prefix}.operator_norm.weight"] = (dim,)
        expected[f"{prefix}.ffn_norm.weight"] = (dim,)
        if layer_type in {"full_attention", "sliding_attention"}:
            expected.update(
                {
                    f"{prefix}.self_attn.q_proj.weight": (dim, dim),
                    f"{prefix}.self_attn.k_proj.weight": (kv_dim, dim),
                    f"{prefix}.self_attn.v_proj.weight": (kv_dim, dim),
                    f"{prefix}.self_attn.out_proj.weight": (dim, dim),
                    f"{prefix}.self_attn.q_layernorm.weight": (head_dim,),
                    f"{prefix}.self_attn.k_layernorm.weight": (head_dim,),
                }
            )
        else:
            expected.update(
                {
                    f"{prefix}.conv.in_proj.weight": (3 * dim, dim),
                    f"{prefix}.conv.conv": (dim, config.conv_L_cache),
                    f"{prefix}.conv.out_proj.weight": (dim, dim),
                }
            )
            if config.conv_bias:
                expected.update(
                    {
                        f"{prefix}.conv.in_proj.bias": (3 * dim,),
                        f"{prefix}.conv.conv_bias": (dim,),
                        f"{prefix}.conv.out_proj.bias": (dim,),
                    }
                )
        expected.update(
            {
                f"{prefix}.feed_forward.w1.weight": (ff, dim),
                f"{prefix}.feed_forward.w3.weight": (ff, dim),
                f"{prefix}.feed_forward.w2.weight": (dim, ff),
            }
        )
    return WeightContract(expected)
