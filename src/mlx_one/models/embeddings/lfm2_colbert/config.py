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
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.model_type not in {"lfm2_colbert", "lfm2-colbert"}:
            raise ConfigError(f"Lfm2ColBERTConfig cannot represent model_type={self.model_type!r}")
        for name in ("projection_dim", "query_max_length", "document_max_length"):
            require_positive(name, getattr(self, name))
        if not isinstance(self.normalize, bool):
            raise ConfigError("normalize must be a boolean")

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
        }
        payload = {key: value for key, value in data.items() if key in known}
        payload["model_type"] = str(data.get("model_type", "lfm2_colbert")).replace("-", "_")
        payload["text_config"] = Lfm2Config.from_dict(text)
        payload["extra"] = extras(data, known | set(text))
        return cls(**payload)
