"""Configuration for MPNet Sentence Transformer encoders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_divisible, require_positive
from mlx_one.models.shared.encoder import validate_dropout


def mpnet_relative_position_bucket(
    relative_position: int, *, num_buckets: int = 32, max_distance: int = 128
) -> int:
    """Scalar reference for MPNet's bidirectional relative-position buckets."""

    if num_buckets < 4 or num_buckets % 2:
        raise ConfigError("num_buckets must be an even integer of at least four")
    half = num_buckets // 2
    distance = -relative_position
    result = half if distance < 0 else 0
    distance = abs(distance)
    max_exact = half // 2
    if max_distance <= max_exact:
        raise ConfigError("max_distance must exceed the exact bucket range")
    if distance < max_exact:
        return result + distance
    bucket = max_exact + int(
        math.log(distance / max_exact) / math.log(max_distance / max_exact)
        * (half - max_exact)
    )
    return result + min(bucket, half - 1)


@dataclass(frozen=True)
class MPNetEmbeddingConfig:
    model_type: str = "mpnet"
    vocab_size: int = 30527
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    sentence_max_length: int = 384
    add_pooling_layer: bool = True
    normalize_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "mpnet":
            raise ConfigError(
                f"MPNetEmbeddingConfig cannot represent model_type={self.model_type!r}"
            )
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
            "sentence_max_length",
        ):
            require_positive(name, getattr(self, name))
        require_positive("initializer_range", self.initializer_range)
        require_positive("layer_norm_eps", self.layer_norm_eps)
        require_divisible(
            "hidden_size", self.hidden_size, "num_attention_heads", self.num_attention_heads
        )
        if self.relative_attention_num_buckets < 4:
            raise ConfigError("relative_attention_num_buckets must be at least four")
        if self.relative_attention_max_distance <= self.relative_attention_num_buckets // 4:
            raise ConfigError("relative_attention_max_distance is too small")
        require_divisible(
            "relative_attention_num_buckets", self.relative_attention_num_buckets,
            "bidirectional bucket groups", 2,
        )
        validate_dropout("hidden_dropout_prob", self.hidden_dropout_prob)
        validate_dropout(
            "attention_probs_dropout_prob", self.attention_probs_dropout_prob
        )
        if self.hidden_act not in {"gelu", "gelu_new"}:
            raise ConfigError(f"unsupported MPNet activation: {self.hidden_act}")
        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ConfigError(f"{name} must be within the vocabulary")
        if self.sentence_max_length + self.pad_token_id + 1 > self.max_position_embeddings:
            raise ConfigError("sentence_max_length exceeds MPNet position embedding capacity")
        for name in ("add_pooling_layer", "normalize_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MPNetEmbeddingConfig:
        data = dict(data)
        if "sentence_max_length" not in data and "max_seq_length" in data:
            data["sentence_max_length"] = data["max_seq_length"]
        known = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        payload = {key: value for key, value in data.items() if key in known}
        payload["extra"] = extras(data, known | {"max_seq_length"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid MPNet embedding configuration: {exc}") from exc
