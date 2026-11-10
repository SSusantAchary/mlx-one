"""Backend-free XLM-R profile configuration with explicit head ownership."""

from dataclasses import dataclass
from typing import Any

from mlx_one.core.config import ConfigError
from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig


@dataclass(frozen=True)
class XLMRobertaConfig:
    """Reuse BERT's compatible encoder contracts, with XLM-R position/head rules."""

    backbone: BertEmbeddingConfig
    task: str = "embedding"
    multi_representation: bool = False
    colbert_dim: int = 1024
    model_type: str = "xlm-roberta"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backbone, name)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "XLMRobertaConfig":
        if data.get("model_type") != "xlm-roberta":
            raise ConfigError("XLM-R config requires model_type=xlm-roberta")
        if data.get("is_decoder") or data.get("add_cross_attention"):
            raise ConfigError("XLM-R retrieval supports bidirectional encoder checkpoints only")
        task = data.get("task", "embedding")
        if task not in {"embedding", "reranking"}:
            raise ConfigError("XLM-R task must be embedding or reranking")
        if task == "reranking" and data.get("num_labels", len(data.get("id2label", {}))) != 1:
            raise ConfigError("XLM-R reranking requires one output logit")
        pad = data.get("pad_token_id", 1)
        capacity = data.get("max_position_embeddings", 514) - pad - 1
        payload = {
            **data,
            "model_type": "bert",
            "task": "embedding",
            "type_vocab_size": 1,
            "pad_token_id": pad,
            "layer_norm_eps": data.get("layer_norm_eps", 1e-5),
            "sentence_max_length": data.get("sentence_max_length", capacity),
            "add_pooling_layer": False,
            "pooling": "cls",
        }
        if data.get("type_vocab_size", 1) != 1 or payload["sentence_max_length"] > capacity:
            raise ConfigError("unsupported XLM-R type vocabulary or position capacity")
        multi = data.get("multi_representation", False)
        if type(multi) is not bool or (multi and task != "embedding"):
            raise ConfigError("multi-representation requires an embedding profile")
        dim = data.get("colbert_dim", data.get("hidden_size", 1024))
        if type(dim) is not int or dim < 1:
            raise ConfigError("colbert_dim must be a positive integer")
        return cls(BertEmbeddingConfig.from_dict(payload), task, multi, dim)
