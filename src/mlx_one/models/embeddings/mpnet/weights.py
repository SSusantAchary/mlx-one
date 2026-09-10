"""Strict Hugging Face weight contract for MPNet sentence encoders."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.mpnet.config import MPNetEmbeddingConfig


def sanitize_weights(
    weights: dict[str, Any], config: MPNetEmbeddingConfig
) -> dict[str, Any]:
    cleaned = {}
    for name, value in weights.items():
        if name.endswith("embeddings.position_ids"):
            continue
        if name.startswith("mpnet."):
            name = name.removeprefix("mpnet.")
        cleaned[name] = value
    return cleaned


def weight_contract(config: MPNetEmbeddingConfig) -> WeightContract:
    dim = config.hidden_size
    ff = config.intermediate_size
    expected: dict[str, tuple[int, ...]] = {
        "embeddings.word_embeddings.weight": (config.vocab_size, dim),
        "embeddings.position_embeddings.weight": (config.max_position_embeddings, dim),
        "embeddings.LayerNorm.weight": (dim,),
        "embeddings.LayerNorm.bias": (dim,),
        "encoder.relative_attention_bias.weight": (
            config.relative_attention_num_buckets,
            config.num_attention_heads,
        ),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layer.{index}"
        for projection in ("q", "k", "v", "o"):
            expected[f"{prefix}.attention.attn.{projection}.weight"] = (dim, dim)
            expected[f"{prefix}.attention.attn.{projection}.bias"] = (dim,)
        expected[f"{prefix}.attention.LayerNorm.weight"] = (dim,)
        expected[f"{prefix}.attention.LayerNorm.bias"] = (dim,)
        expected[f"{prefix}.intermediate.dense.weight"] = (ff, dim)
        expected[f"{prefix}.intermediate.dense.bias"] = (ff,)
        expected[f"{prefix}.output.dense.weight"] = (dim, ff)
        expected[f"{prefix}.output.dense.bias"] = (dim,)
        expected[f"{prefix}.output.LayerNorm.weight"] = (dim,)
        expected[f"{prefix}.output.LayerNorm.bias"] = (dim,)
    if config.add_pooling_layer:
        expected["pooler.dense.weight"] = (dim, dim)
        expected["pooler.dense.bias"] = (dim,)
    return WeightContract(expected)
