"""Configuration for the LFM2 ColBERT late-interaction encoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mlx_one.core.config import ConfigError, extras, require_positive
from mlx_one.models.language.lfm2.config import Lfm2Config


@dataclass(frozen=True)
class Lfm2ColBERTConfig:
    model_type: str
    text_config: Lfm2Config
    projection_dim: int = 128
    normalize: bool = True
    query_max_length: int = 32
    document_max_length: int = 512
    query_prefix: str | None = None
    document_prefix: str | None = None
    do_query_expansion: bool = False
    attend_to_expansion_tokens: bool = False
    skiplist_words: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"lfm2_colbert", "lfm2-colbert"}:
            raise ConfigError(f"Lfm2ColBERTConfig cannot represent model_type={self.model_type!r}")
        for name in ("projection_dim", "query_max_length", "document_max_length"):
            require_positive(name, getattr(self, name))
        if not isinstance(self.normalize, bool):
            raise ConfigError("normalize must be a boolean")
        for name in ("do_query_expansion", "attend_to_expansion_tokens"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigError(f"{name} must be a boolean")
        if (self.query_prefix is None) != (self.document_prefix is None):
            raise ConfigError("query and document prefixes must be supplied together")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (self.query_prefix, self.document_prefix)
        ):
            raise ConfigError("ColBERT prefixes must be nonempty strings")
        if self.query_prefix is not None and "sliding_attention" in self.text_config.layer_types:
            raise ConfigError("publisher ColBERT recipe does not admit sliding attention")
        if not isinstance(self.skiplist_words, (tuple, list)) or any(
            not isinstance(w, str) for w in self.skiplist_words
        ):
            raise ConfigError("skiplist_words must be a sequence of strings")
        if self.do_query_expansion and self.query_prefix is None:
            raise ConfigError("query expansion requires the checkpoint prefix recipe")
        object.__setattr__(self, "skiplist_words", tuple(self.skiplist_words))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Lfm2ColBERTConfig:
        nested = data.get("text_config", data)
        if not isinstance(nested, Mapping):
            raise ConfigError("text_config must be an object")
        text = dict(nested)
        text["model_type"] = "lfm2"
        known = {
            "model_type",
            "text_config",
            "projection_dim",
            "normalize",
            "query_max_length",
            "document_max_length",
            "query_prefix",
            "document_prefix",
            "do_query_expansion",
            "attend_to_expansion_tokens",
            "skiplist_words",
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = str(data.get("model_type", "lfm2_colbert")).replace("-", "_")
        payload["text_config"] = Lfm2Config.from_dict(text)
        payload["extra"] = extras(data, known | set(text))
        return cls(**payload)
