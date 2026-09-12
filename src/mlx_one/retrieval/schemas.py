"""JSON-serializable native retrieval results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class EmbeddingResult:
    texts: tuple[str, ...]
    embeddings: tuple[tuple[float, ...], ...]
    dimensions: int
    input_type: Literal["query", "document"]
    model: str
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class RerankItem:
    index: int
    document: str
    score: float
    probability: float


@dataclass(frozen=True)
class RerankResult:
    query: str
    items: tuple[RerankItem, ...]
    model: str
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
