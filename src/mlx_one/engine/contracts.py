"""Backend-neutral contracts for the native mlx-one execution engine."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Any, Literal, Protocol, runtime_checkable


class RequestKind(str, Enum):
    TEXT = "text"
    VISION = "vision"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    TRANSCRIPTION = "transcription"


class SequenceState(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            SequenceState.FINISHED,
            SequenceState.CANCELLED,
            SequenceState.TIMED_OUT,
            SequenceState.FAILED,
        }


@dataclass(frozen=True)
class CacheCapabilities:
    batchable: bool = False
    trimmable: bool = False
    quantizable: bool = False
    sliding_window: bool = False
    block_compatible: bool = False


@dataclass(frozen=True)
class RunnerCapabilities:
    request_kinds: frozenset[RequestKind]
    batchable: bool = False
    chunked_prefill: bool = False
    speculative: bool = False
    modalities: frozenset[str] = frozenset({"text"})


@dataclass(frozen=True)
class EngineRequest:
    """Normalized request accepted by an internal model runner."""

    kind: RequestKind
    model: str
    payload: Mapping[str, Any]
    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex}")
    created_at: float = field(default_factory=time.monotonic)
    deadline: float | None = None
    priority: int = 0
    max_tokens: int = 0
    cache_key: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("engine request model cannot be empty")
        if not self.request_id:
            raise ValueError("engine request ID cannot be empty")
        if self.max_tokens < 0:
            raise ValueError("engine request max_tokens cannot be negative")
        if self.deadline is not None and self.deadline < self.created_at:
            raise ValueError("engine request deadline precedes its arrival")


@dataclass
class SequenceContext:
    """Mutable lifecycle state owned by the scheduler for one request."""

    request: EngineRequest
    state: SequenceState = SequenceState.WAITING
    prompt_tokens: tuple[int, ...] = ()
    computed_tokens: int = 0
    generated_tokens: list[int] = field(default_factory=list)
    cache: Any | None = None
    cancel: Event = field(default_factory=Event)
    error: BaseException | None = None

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(len(self.prompt_tokens) - self.computed_tokens, 0)

    def transition(self, target: SequenceState) -> None:
        allowed = {
            SequenceState.WAITING: {
                SequenceState.PREFILL,
                SequenceState.CANCELLED,
                SequenceState.TIMED_OUT,
                SequenceState.FAILED,
            },
            SequenceState.PREFILL: {
                SequenceState.PREFILL,
                SequenceState.DECODING,
                SequenceState.CANCELLED,
                SequenceState.TIMED_OUT,
                SequenceState.FAILED,
            },
            SequenceState.DECODING: {
                SequenceState.DECODING,
                SequenceState.FINISHED,
                SequenceState.CANCELLED,
                SequenceState.TIMED_OUT,
                SequenceState.FAILED,
            },
        }
        if self.state.terminal or target not in allowed.get(self.state, set()):
            raise ValueError(f"invalid sequence transition: {self.state.value} -> {target.value}")
        self.state = target


@dataclass(frozen=True)
class ExecutionBatch:
    sequences: tuple[SequenceContext, ...]
    phase: Literal["prefill", "decode", "encode"]
    token_budget: int

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("execution batch cannot be empty")
        if self.token_budget < 1:
            raise ValueError("execution token budget must be positive")


@dataclass(frozen=True)
class EngineMetrics:
    request_id: str
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    generated_tokens: int = 0
    context_used: int = 0
    reasoning_tokens: int = 0
    queue_ms: float | None = None
    tokenization_ms: float | None = None
    prefill_ms: float | None = None
    ttft_ms: float | None = None
    decode_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    inter_token_latency_ms: float | None = None
    active_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    cache_bytes: int | None = None
    multimodal_memory_bytes: int | None = None
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    draft_acceptance_ratio: float | None = None
    verification_ms: float | None = None
    effective_tokens_per_second: float | None = None
    prefix_cache_hit: bool = False
    reused_prompt_tokens: int = 0
    terminal_state: SequenceState | None = None
    finish_reason: Literal["stop", "length"] | None = None
    failure: EngineError | None = None


@dataclass(frozen=True)
class EngineError:
    message: str
    type: str
    code: str | int


@dataclass(frozen=True)
class EngineEvent:
    request_id: str
    text: str = ""
    reasoning_content: str = ""
    token_ids: tuple[int, ...] = ()
    state: SequenceState | None = None
    finish_reason: Literal["stop", "length"] | None = None
    metrics: EngineMetrics | None = None
    error: EngineError | None = None


@runtime_checkable
class ModelRunner(Protocol):
    @property
    def capabilities(self) -> RunnerCapabilities: ...

    def prepare(self, request: EngineRequest) -> SequenceContext: ...

    def make_cache(self) -> Any: ...

    def prefill(self, batch: ExecutionBatch) -> Sequence[Any]: ...

    def decode(self, batch: ExecutionBatch) -> Sequence[Any]: ...

    def finalize(self, sequence: SequenceContext) -> None: ...


@runtime_checkable
class TaskRunner(Protocol):
    """Non-autoregressive runner contract for encoder and transcription tasks."""

    @property
    def capabilities(self) -> RunnerCapabilities: ...

    def execute(self, request: EngineRequest) -> Any: ...


@runtime_checkable
class EngineRuntime(Protocol):
    def submit(self, request: EngineRequest) -> Any: ...

    def cancel(self, request_id: str) -> bool: ...

    def close(self) -> None: ...
