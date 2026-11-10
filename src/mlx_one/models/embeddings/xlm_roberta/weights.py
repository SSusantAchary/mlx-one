"""Strict XLM-R mapping; converted BGE-M3 auxiliary heads remain explicit."""

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.embeddings.bert.weights import sanitize_weights as bert_sanitize
from mlx_one.models.embeddings.bert.weights import weight_contract as bert_contract
from mlx_one.models.embeddings.xlm_roberta.config import XLMRobertaConfig


def sanitize_weights(weights: dict[str, Any], config: XLMRobertaConfig) -> dict[str, Any]:
    stripped = {}
    for name, value in weights.items():
        name = name.removeprefix("roberta.")
        if name in stripped:
            raise ValueError("duplicate XLM-R tensor after prefix removal")
        stripped[name] = value
    return bert_sanitize(stripped, config.backbone)


def weight_contract(config: XLMRobertaConfig) -> WeightContract:
    expected = dict(bert_contract(config.backbone).expected)
    width = config.hidden_size
    if config.task == "reranking":
        expected.update(
            {
                "classifier.dense.weight": (width, width),
                "classifier.dense.bias": (width,),
                "classifier.out_proj.weight": (1, width),
                "classifier.out_proj.bias": (1,),
            }
        )
    if config.multi_representation:
        expected.update(
            {
                "sparse_linear.weight": (1, width),
                "sparse_linear.bias": (1,),
                "colbert_linear.weight": (config.colbert_dim, width),
                "colbert_linear.bias": (config.colbert_dim,),
            }
        )
    return WeightContract(expected)
