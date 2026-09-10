"""Typed public contracts for native ASR transcription."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class WhisperDecodeOptions:
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    beam_size: int | None = 5
    best_of: int | None = 5
    patience: float = 1.0
    length_penalty: float | None = None
    compression_ratio_threshold: float | None = 2.4
    logprob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    suppress_blank: bool = True
    suppress_tokens: tuple[int, ...] | None = None
    max_initial_timestamp: float | None = 1.0
    max_tokens: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.temperatures)
        if not temperatures or any(value < 0 for value in temperatures):
            raise ValueError("temperatures must contain non-negative values")
        if self.beam_size is not None and self.beam_size < 1:
            raise ValueError("beam_size must be positive")
        if self.best_of is not None and self.best_of < 1:
            raise ValueError("best_of must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "temperatures", temperatures)
        if self.suppress_tokens is not None:
            object.__setattr__(self, "suppress_tokens", tuple(self.suppress_tokens))


@dataclass(frozen=True)
class TranscriptionWord:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class TranscriptionSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: tuple[TranscriptionWord, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    segments: tuple[TranscriptionSegment, ...]
    duration: float
    model: str
    revision: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
