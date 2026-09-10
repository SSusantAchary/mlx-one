"""Strict Hugging Face weight contract for BERT sentence encoders."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig


def sanitize_weights(
    weights: dict[str, Any], config: BertEmbeddingConfig
) -> dict[str, Any]:
    cleaned = {}
    for name, value in weights.items():
        if name.endswith(("embeddings.position_ids", "embeddings.token_type_ids")):
            continue
        if name.startswith("bert."):
            name = name.removeprefix("bert.")
        name = name.replace(".attention.self.", ".attention.self_attn.")
        cleaned[name] = value
    return cleaned


def weight_contract(config: BertEmbeddingConfig) -> WeightContract:
    dim = config.hidden_size
    ff = config.intermediate_size
    expected: dict[str, tuple[int, ...]] = {
        "embeddings.word_embeddings.weight": (config.vocab_size, dim),
        "embeddings.position_embeddings.weight": (config.max_position_embeddings, dim),
        "embeddings.token_type_embeddings.weight": (config.type_vocab_size, dim),
        "embeddings.LayerNorm.weight": (dim,),
        "embeddings.LayerNorm.bias": (dim,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layer.{index}"
        for projection in ("query", "key", "value"):
            expected[f"{prefix}.attention.self_attn.{projection}.weight"] = (dim, dim)
            expected[f"{prefix}.attention.self_attn.{projection}.bias"] = (dim,)
        expected[f"{prefix}.attention.output.dense.weight"] = (dim, dim)
        expected[f"{prefix}.attention.output.dense.bias"] = (dim,)
        expected[f"{prefix}.attention.output.LayerNorm.weight"] = (dim,)
        expected[f"{prefix}.attention.output.LayerNorm.bias"] = (dim,)
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
