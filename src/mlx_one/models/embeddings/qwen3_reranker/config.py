"""Configuration for Qwen3-Reranker-0.6B."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_positive
from mlx_one.models.embeddings.qwen3_embedding.config import (
    _validate_official_qwen3_settings,
)
from mlx_one.models.language.qwen3.config import Qwen3Config


@dataclass(frozen=True)
class Qwen3RerankerConfig:
    model_type: str = "qwen3_reranker"
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
            max_position_embeddings=40960,
            tie_word_embeddings=True,
        )
    )
    max_seq_length: int = 40960
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type != "qwen3_reranker":
            raise ConfigError(f"invalid Qwen3 reranker model_type: {self.model_type!r}")
        require_positive("max_seq_length", self.max_seq_length)
        if self.max_seq_length > self.text_config.max_position_embeddings:
            raise ConfigError("max_seq_length cannot exceed the Qwen3 context length")
        if not self.text_config.tie_word_embeddings:
            raise ConfigError("Qwen3 reranker requires tied word embeddings")
        for name in ("bos_token_id", "eos_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer")
            if not 0 <= value < self.text_config.vocab_size:
                raise ConfigError(f"{name} must be within the vocabulary")
        _validate_official_qwen3_settings(self.text_config)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Qwen3RerankerConfig:
        outer_type = data.get("model_type", "qwen3_reranker")
        if outer_type not in {"qwen3", "qwen3_reranker"}:
            raise ConfigError(f"invalid Qwen3 reranker model_type: {outer_type!r}")
        nested = data.get("text_config", data)
        if not isinstance(nested, Mapping):
            raise ConfigError("text_config must be an object")
        text = dict(nested)
        if "text_config" in data and text.get("model_type", "qwen3") != "qwen3":
            raise ConfigError("Qwen3 reranker text_config model_type must be 'qwen3'")
        text["model_type"] = "qwen3"
        known = {
            "model_type",
            "text_config",
            "max_seq_length",
            "bos_token_id",
            "eos_token_id",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = "qwen3_reranker"
        payload["text_config"] = Qwen3Config.from_dict(text)
        payload.setdefault("max_seq_length", int(text["max_position_embeddings"]))
        payload.setdefault("bos_token_id", int(text.get("bos_token_id", 151643)))
        payload.setdefault("eos_token_id", int(text.get("eos_token_id", 151645)))
        payload["extra"] = extras(data, known | set(text))
        try:
            return cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid Qwen3 reranker configuration: {exc}") from exc
