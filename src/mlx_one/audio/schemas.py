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
    without_timestamps: bool = False

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


@dataclass(frozen=True)
class ASRDecodeOptions:
    """Greedy decoding for non-Whisper ASR; no Whisper fallback metrics."""

    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True)
class SpeechTranscript:
    """Non-Whisper transcript: unavailable timestamps/confidence remain unset."""

    text: str
    duration: float
    model: str
    revision: str | None
    sample_rate: int
    tokens: tuple[int, ...] = ()
    language: str | None = None
    confidence: float | None = None
    timestamps: tuple[dict[str, Any], ...] | None = None
    task: str = "transcribe"
    timings: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class QwenASROptions(ASRDecodeOptions):
    """Qwen greedy budget and system context; not Whisper initial-prompt tokens."""

    context: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.context, str):
            raise ValueError("Qwen context must be a string")


@dataclass(frozen=True)
class AlignmentUnit:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class AudioAlignment:
    units: tuple[AlignmentUnit, ...]
    text: str
    language: str
    duration: float
    model: str
    revision: str | None
    timestamp_tick_seconds: float
    sample_rate: int = 16000
    task: Literal["align"] = "align"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class AudioStreamEvent:
    """Complete current hypothesis, never an append-only text delta."""

    kind: Literal["partial", "final"]
    text: str
    sequence: int
    audio_seconds: float
    model: str
    revision: str | None
    language: str | None = None
    confidence: float | None = None
    timestamps: tuple[dict[str, Any], ...] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
