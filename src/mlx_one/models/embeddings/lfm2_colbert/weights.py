"""Strict weight contract for LFM2-ColBERT."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
from mlx_one.models.language.lfm2.weights import sanitize_weights as sanitize_text
from mlx_one.models.language.lfm2.weights import weight_contract as text_contract


def sanitize_weights(weights: dict[str, Any], config: Lfm2ColBERTConfig) -> dict[str, Any]:
    renamed = {}
    for name, value in weights.items():
        if name == "1_Dense.linear.weight":
            name = "projection.weight"
        renamed[name] = value
    return sanitize_text(renamed, config.text_config)


def weight_contract(config: Lfm2ColBERTConfig) -> WeightContract:
    expected = dict(text_contract(config.text_config).expected)
    expected["projection.weight"] = (config.projection_dim, config.text_config.hidden_size)
    return WeightContract(expected)
