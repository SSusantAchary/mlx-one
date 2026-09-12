"""Configuration for Qwen3-Embedding-0.6B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_positive
from mlx_one.models.language.qwen3.config import Qwen3Config


@dataclass(frozen=True)
class Qwen3EmbeddingConfig:
    model_type: str = "qwen3_embedding"
    text_config: Qwen3Config = field(
        default_factory=lambda: Qwen3Config(
            model_type="qwen3",
            hidden_size=1024,
            num_hidden_layers=28,
            intermediate_size=3072,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151669,
            max_position_embeddings=32768,
            tie_word_embeddings=True,
        )
    )
    min_dimension: int = 32
    max_dimension: int = 1024
    max_seq_length: int = 32768
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    normalize_embeddings: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen3_embedding":
            raise ConfigError(f"invalid Qwen3 embedding model_type: {self.model_type!r}")
        for name in ("min_dimension", "max_dimension", "max_seq_length"):
            require_positive(name, getattr(self, name))
        if self.min_dimension > self.max_dimension:
            raise ConfigError("min_dimension cannot exceed max_dimension")
        if self.max_dimension > self.text_config.hidden_size:
            raise ConfigError("max_dimension cannot exceed text hidden_size")
        if self.max_seq_length > self.text_config.max_position_embeddings:
            raise ConfigError("max_seq_length cannot exceed the Qwen3 context length")
        if not self.text_config.tie_word_embeddings:
            raise ConfigError("Qwen3 embedding requires tied word embeddings")
        for name in ("bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer")
            if not 0 <= value < self.text_config.vocab_size:
                raise ConfigError(f"{name} must be within the vocabulary")
        _validate_official_qwen3_settings(self.text_config)
        if not isinstance(self.normalize_embeddings, bool):
            raise ConfigError("normalize_embeddings must be a boolean")

    def validate_dimensions(self, dimensions: int | None) -> int:
        value = self.max_dimension if dimensions is None else dimensions
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("embedding dimensions must be an integer")
        if not self.min_dimension <= value <= self.max_dimension:
            raise ValueError(
                f"embedding dimensions must be within [{self.min_dimension}, {self.max_dimension}]"
            )
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3EmbeddingConfig:
        outer_type = data.get("model_type", "qwen3_embedding")
        if outer_type not in {"qwen3", "qwen3_embedding"}:
            raise ConfigError(f"invalid Qwen3 embedding model_type: {outer_type!r}")
        nested = data.get("text_config", data)
        if not isinstance(nested, Mapping):
            raise ConfigError("text_config must be an object")
        text = dict(nested)
        if "text_config" in data and text.get("model_type", "qwen3") != "qwen3":
            raise ConfigError("Qwen3 embedding text_config model_type must be 'qwen3'")
        text["model_type"] = "qwen3"
        known = {
            "model_type",
            "text_config",
            "min_dimension",
            "max_dimension",
            "max_seq_length",
            "bos_token_id",
            "eos_token_id",
            "normalize_embeddings",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = "qwen3_embedding"
        payload["text_config"] = Qwen3Config.from_dict(text)
        raw_dimension = data.get("word_embedding_dimension", text["hidden_size"])
        if isinstance(raw_dimension, bool) or not isinstance(raw_dimension, int):
            raise ConfigError("word_embedding_dimension must be an integer")
        payload.setdefault("max_dimension", raw_dimension)
        payload.setdefault("min_dimension", min(32, payload["max_dimension"]))
        payload.setdefault(
            "max_seq_length",
            int(data.get("max_seq_length", text["max_position_embeddings"])),
        )
        payload.setdefault("bos_token_id", int(text.get("bos_token_id", 151643)))
        payload.setdefault("eos_token_id", int(text.get("eos_token_id", 151643)))
        payload["extra"] = extras(data, known | set(text) | {"word_embedding_dimension"})
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3 embedding configuration: {exc}") from exc


def _validate_official_qwen3_settings(config: Qwen3Config) -> None:
    settings = config.extra
    if settings.get("hidden_act", "silu") != "silu":
        raise ConfigError("Qwen3 retrieval models require hidden_act='silu'")
    if settings.get("attention_bias", False) is not False:
        raise ConfigError("Qwen3 retrieval models require bias-free attention")
    dropout = settings.get("attention_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or dropout != 0:
        raise ConfigError("Qwen3 retrieval models require zero attention dropout")
    if settings.get("use_sliding_window", False) is not False:
        raise ConfigError("Qwen3 retrieval models do not support sliding-window attention")
    if settings.get("sliding_window") is not None:
        raise ConfigError("Qwen3 retrieval models require sliding_window=null")
