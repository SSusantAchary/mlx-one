"""Strict Qwen3 embedding tensor contract."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.language.qwen3.weights import sanitize_weights as sanitize_qwen3
from mlx_one.models.language.qwen3.weights import weight_contract as qwen3_contract


def sanitize_weights(weights: dict[str, Any], config: Qwen3EmbeddingConfig) -> dict[str, Any]:
    # AutoModel publisher bundles omit the CausalLM wrapper's `model.` prefix.
    # Normalize only known backbone roots and reject ambiguous duplicate names.
    renamed = {}
    for name, value in weights.items():
        target = "model." + name if name.startswith(("embed_tokens.", "layers.", "norm.")) else name
        if target in renamed:
            raise ValueError(f"duplicate Qwen3 embedding tensor after normalization: {target}")
        renamed[target] = value
    return sanitize_qwen3(renamed, config.text_config)


def weight_contract(config: Qwen3EmbeddingConfig) -> WeightContract:
    return qwen3_contract(config.text_config)
