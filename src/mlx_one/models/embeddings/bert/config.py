"""Configuration for BERT-compatible Sentence Transformer encoders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.shared.encoder import validate_dropout


@dataclass(frozen=True)
class BertEmbeddingConfig:
    model_type: str = "bert"
    vocab_size: int = 30522
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 1536
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    sentence_max_length: int = 256
    add_pooling_layer: bool = True
    normalize_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "bert":
            raise ConfigError(
                f"BertEmbeddingConfig cannot represent model_type={self.model_type!r}"
            )
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "type_vocab_size",
            "sentence_max_length",
        ):
            require_positive(name, getattr(self, name))
        require_positive("initializer_range", self.initializer_range)
        require_positive("layer_norm_eps", self.layer_norm_eps)
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )
        validate_dropout("hidden_dropout_prob", self.hidden_dropout_prob)
        validate_dropout(
            "attention_probs_dropout_prob", self.attention_probs_dropout_prob
        )
        if self.hidden_act not in {"gelu", "gelu_new"}:
            raise ConfigError(f"unsupported BERT activation: {self.hidden_act}")
        if self.position_embedding_type != "absolute":
            raise ConfigError("only absolute BERT position embeddings are supported")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ConfigError("pad_token_id must be within the vocabulary")
        if self.sentence_max_length > self.max_position_embeddings:
            raise ConfigError("sentence_max_length cannot exceed max_position_embeddings")
        for name in ("add_pooling_layer", "normalize_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BertEmbeddingConfig:
        data = dict(data)
        if "sentence_max_length" not in data and "max_seq_length" in data:
            data["sentence_max_length"] = data["max_seq_length"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known | {"max_seq_length"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid BERT embedding configuration: {exc}") from exc
