"""Hugging Face tensor contract for Apple OpenELM."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.openelm.config import OpenELMConfig


def sanitize_weights(weights: dict[str, Any], config: OpenELMConfig) -> dict[str, Any]:
    sanitized = {
        name: value
        for name, value in weights.items()
        if not name.endswith("pos_embedding.inv_freq")
        and not name.endswith("rotary_emb.inv_freq")
        and not name.endswith("causal_mask")
    }
    if config.share_input_output_layers:
        sanitized.pop("lm_head.weight", None)
    return sanitized


def weight_contract(config: OpenELMConfig) -> WeightContract:
    expected: dict[str, tuple[int, ...]] = {
        "transformer.token_embeddings.weight": (config.vocab_size, config.model_dim),
        "transformer.norm.weight": (config.model_dim,),
    }
    if not config.share_input_output_layers:
        expected["lm_head.weight"] = (config.vocab_size, config.model_dim)
    for index, (query_heads, kv_heads, ffn_dim) in enumerate(
        zip(config.num_query_heads, config.num_kv_heads, config.ffn_dims, strict=True)
    ):
        prefix = f"transformer.layers.{index}"
        qkv_size = (query_heads + 2 * kv_heads) * config.head_dim
        first_ffn_size = ffn_dim * (2 if config.ffn_with_glu else 1)
        expected.update(
            {
                f"{prefix}.attn_norm.weight": (config.model_dim,),
                f"{prefix}.ffn_norm.weight": (config.model_dim,),
                f"{prefix}.attn.qkv_proj.weight": (qkv_size, config.model_dim),
                f"{prefix}.attn.out_proj.weight": (
                    config.model_dim,
                    query_heads * config.head_dim,
                ),
                f"{prefix}.ffn.proj_1.weight": (first_ffn_size, config.model_dim),
                f"{prefix}.ffn.proj_2.weight": (config.model_dim, ffn_dim),
            }
        )
        if config.normalize_qk_projections:
            expected[f"{prefix}.attn.q_norm.weight"] = (config.head_dim,)
            expected[f"{prefix}.attn.k_norm.weight"] = (config.head_dim,)
    return WeightContract(expected)
