"""Typed native text-generation contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class TextGenerationOptions:
    max_tokens: int = 128
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int = 0
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            raise ValueError("top_k must be null or positive")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise ValueError("top_p must be a number")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be within (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        stop = tuple(self.stop)
        if any(not isinstance(value, str) or not value for value in stop):
            raise ValueError("stop sequences must be non-empty strings")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))
        object.__setattr__(self, "stop", stop)


@dataclass(frozen=True)
class GenerationChunk:
    token_id: int | None
    text: str
    generated_tokens: int
    finish_reason: Literal["stop", "length"] | None = None


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    prompt_tokens: int
    generation_tokens: int
    finish_reason: Literal["stop", "length"]
    model: str
    revision: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
