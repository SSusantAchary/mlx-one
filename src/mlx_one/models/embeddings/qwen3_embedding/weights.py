"""Strict Qwen3 embedding tensor contract."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.language.qwen3.weights import sanitize_weights as sanitize_qwen3
from mlx_one.models.language.qwen3.weights import weight_contract as qwen3_contract


def sanitize_weights(weights: dict[str, Any], config: Qwen3EmbeddingConfig) -> dict[str, Any]:
    return sanitize_qwen3(weights, config.text_config)


def weight_contract(config: Qwen3EmbeddingConfig) -> WeightContract:
    return qwen3_contract(config.text_config)
