"""Backend-neutral contracts for the native mlx-one execution engine."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
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


class Phase(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            Phase.FINISHED,
            Phase.CANCELLED,
            Phase.TIMED_OUT,
            Phase.FAILED,
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
    family: str = "unknown"
    supported_cache_backends: tuple[str, ...] = ("dense",)
    supported_state_kinds: tuple[str, ...] = ()
    supports_batched_prefill: bool = False
    supports_batched_decode: bool = False
    supports_mixed_prefill_decode: bool = False
    supports_paged_attention: bool = False
    supports_exact_prefix_restore: bool = False
    supports_128k: bool = False


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
class SequenceState:
    """Actor-owned committed state. Generated IDs may include one uncomputed token."""

    request_id: str
    input_ids: tuple[int, ...]
    prompt_len: int
    computed_len: int = 0
    generated_ids: list[int] = field(default_factory=list)
    phase: Phase = Phase.WAITING
    cache_handle_id: str | None = None
    deadline_ns: int | None = None
    tenant_namespace: str = "private"
    rng_seed: int | None = None
    request: EngineRequest | None = None
    cache: Any | None = None
    cancel: Event = field(default_factory=Event)
    error: BaseException | None = None
    model_epoch: int = 0
    last_event_no: int = -1

    def __post_init__(self) -> None:
        if not self.request_id or self.prompt_len != len(self.input_ids):
            raise ValueError("sequence requires an ID and matching prompt length")
        if not 0 <= self.computed_len <= self.prompt_len + len(self.generated_ids):
            raise ValueError("invalid committed token boundary")

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(self.prompt_len - self.computed_len, 0)

    def transition(self, target: Phase) -> None:
        if target is self.phase:
            return
        allowed = {
            Phase.WAITING: {Phase.PREFILL},
            Phase.PREFILL: {Phase.DECODE, Phase.PAUSED, Phase.FINISHED},
            Phase.DECODE: {Phase.PAUSED, Phase.FINISHED},
            Phase.PAUSED: {Phase.PREFILL, Phase.DECODE},
        }
        failures = {Phase.CANCELLED, Phase.TIMED_OUT, Phase.FAILED}
        if self.phase.terminal or target not in allowed.get(self.phase, set()) | failures:
            raise ValueError(f"invalid sequence transition: {self.phase.value} -> {target.value}")
        self.phase = target

    def commit(self, count: int) -> None:
        if self.phase not in {Phase.PREFILL, Phase.DECODE} or count < 1:
            raise ValueError("commit requires active execution and a positive token count")
        if self.computed_len + count > self.prompt_len + len(self.generated_ids):
            raise ValueError("commit exceeds available input tokens")
        self.computed_len += count

    def accept_event(self, *, request_id: str, model_epoch: int, event_no: int) -> bool:
        """Ignore stale lifecycle/transport events without touching committed state."""
        if (request_id != self.request_id or model_epoch != self.model_epoch
                or event_no <= self.last_event_no or self.phase.terminal):
            return False
        self.last_event_no = event_no
        return True


@dataclass(frozen=True)
class ScheduledSequence:
    request_id: str
    phase: Literal["prefill", "decode"]
    start_pos: int
    input_token_count: int
    block_table_id: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id or self.phase not in {"prefill", "decode"}:
            raise ValueError("invalid scheduled request or phase")
        if self.start_pos < 0 or self.input_token_count < 1:
            raise ValueError("scheduled positions and token counts must be valid")
        if self.phase == "decode" and self.input_token_count != 1:
            raise ValueError("decode schedules exactly one input token")


@dataclass(frozen=True)
class BatchPlan:
    sequences: tuple[ScheduledSequence, ...]
    total_scheduled_tokens: int
    reserved_new_blocks: int = 0
    execution_mode: Literal["dense", "paged", "hybrid"] = "dense"

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("batch plan cannot be empty")
        ids = [item.request_id for item in self.sequences]
        if len(set(ids)) != len(ids):
            raise ValueError("batch request IDs must be unique")
        if self.total_scheduled_tokens != sum(s.input_token_count for s in self.sequences):
            raise ValueError("batch total must equal scheduled token counts")
        if self.reserved_new_blocks < 0 or self.execution_mode not in {"dense", "paged", "hybrid"}:
            raise ValueError("invalid batch reservations or execution mode")


@dataclass(frozen=True)
class RunnerOutput:
    request_ids: tuple[str, ...]
    logits: Any
    committed_token_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("runner output request IDs must be unique")
        if len(self.request_ids) != len(self.committed_token_counts):
            raise ValueError("runner output mappings must align")
        if any(count < 0 for count in self.committed_token_counts):
            raise ValueError("committed counts cannot be negative")
        if len(self.logits) != len(self.request_ids):
            raise ValueError("runner logits must align with request IDs")


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
    terminal_state: Phase | None = None
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
    state: Phase | None = None
    finish_reason: Literal["stop", "length"] | None = None
    metrics: EngineMetrics | None = None
    error: EngineError | None = None


@runtime_checkable
class ModelRunner(Protocol):
    def capabilities(self) -> RunnerCapabilities: ...

    def prepare(self, request: EngineRequest) -> SequenceState: ...

    def make_cache(self) -> Any: ...

    def prefill(self, plan: BatchPlan) -> RunnerOutput: ...

    def decode(self, plan: BatchPlan) -> RunnerOutput: ...

    def finalize(self, sequence: SequenceState) -> None: ...

    def release(self) -> None: ...


@runtime_checkable
class TaskRunner(Protocol):
    """Non-autoregressive runner contract for encoder and transcription tasks."""

    def capabilities(self) -> RunnerCapabilities: ...

    def execute(self, request: EngineRequest) -> Any: ...


@runtime_checkable
class EngineRuntime(Protocol):
    def submit(self, request: EngineRequest) -> Any: ...

    def cancel(self, request_id: str) -> bool: ...

    def close(self) -> None: ...


class TaskKind(str, Enum):
    """Canonical service tasks; registry aliases remain available in introspection."""

    TEXT_GENERATION = "text-generation"
    VISION_LANGUAGE = "vision-language-generation"
    EMBEDDING = "embedding"
    LATE_INTERACTION = "late-interaction"
    RERANKING = "reranking"
    TRANSCRIPTION = "transcription"
    ALIGNMENT = "alignment"
    LANGUAGE_DETECTION = "language-detection"
    AUDIO_LANGUAGE = "audio-language"
    SPEECH_GENERATION = "speech-generation"
    SPEECH_TO_SPEECH = "speech-to-speech"
    IMAGE_SEGMENTATION = "image-segmentation"
    MASK_GENERATION = "mask-generation"
    VIDEO_SEGMENTATION = "video-segmentation"


def canonical_task(task: str | TaskKind) -> TaskKind:
    return TaskKind({"asr": "transcription", "transcribe": "transcription",
                     "align": "alignment", "audio.transcribe": "transcription",
                     "audio.align": "alignment", "audio.detect_language": "language-detection",
                     "audio.generate_speech": "speech-generation",
                     "audio.speech_to_speech": "speech-to-speech"}.get(task, task))


class StageKind(str, Enum):
    ENCODE = "encode"
    PREFILL = "prefill"
    DECODE = "decode"
    ALIGN = "align"
    SCORE = "score"
    FRAME = "frame"
    CODEC = "codec"
    PREDICT = "predict"
    JOINT = "joint"


class RequestPhase(str, Enum):
    """Task-neutral lifecycle, distinct from task-local execution stages."""

    QUEUED = "queued"
    PREPARING = "preparing"
    ADMITTED = "admitted"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {self.FINISHED, self.CANCELLED, self.TIMED_OUT, self.FAILED}

    def validate_transition(self, target: RequestPhase) -> None:
        if self == target:
            return
        allowed = {self.QUEUED: {self.PREPARING}, self.PREPARING: {self.ADMITTED},
                   self.ADMITTED: {self.RUNNING}, self.RUNNING: {self.PAUSED, self.FINISHED},
                   self.PAUSED: {self.RUNNING}}
        failures = {self.CANCELLED, self.TIMED_OUT, self.FAILED}
        if self.terminal or target not in allowed.get(self, set()) | failures:
            raise ValueError(f"invalid request transition: {self.value} -> {target.value}")


def request_phase(phase: Phase) -> RequestPhase:
    """Compatibility mapping: existing generation events and Phase values stay intact."""
    legacy = {Phase.WAITING: RequestPhase.QUEUED, Phase.PREFILL: RequestPhase.RUNNING,
              Phase.DECODE: RequestPhase.RUNNING}
    return legacy[phase] if phase in legacy else RequestPhase(phase.value)


@dataclass(frozen=True)
class TaskSpec:
    task: TaskKind
    runner_key: str
    stage_kinds: tuple[StageKind, ...]
    batch_policy: str
    output_schema: str
    speculation_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskKind):
            raise ValueError("task must be canonical")
        if not self.runner_key or not self.batch_policy or not self.output_schema:
            raise ValueError("task requires runner, batching and output schema identities")
        if not isinstance(self.stage_kinds, tuple) or not self.stage_kinds or any(
                not isinstance(stage, StageKind) for stage in self.stage_kinds):
            raise ValueError("task requires immutable typed stages")
        if len(set(self.stage_kinds)) != len(self.stage_kinds):
            raise ValueError("duplicate task stages")
        if not isinstance(self.speculation_modes, tuple) or (
                len(set(self.speculation_modes)) != len(self.speculation_modes)):
            raise ValueError("speculation modes must be an immutable unique tuple")


@dataclass(frozen=True)
class LayerExecutionSpec:
    layer_id: int
    state_kind: str
    query_heads: int | None
    kv_heads: int | None
    key_dim: int | None
    value_dim: int | None
    position_policy: str
    window_tokens: int | None = None

    def __post_init__(self) -> None:
        kinds = {"full_kv", "sliding_kv", "cross_kv", "conv", "recurrent", "encoder_features",
                 "predictor", "tracking", "codec"}
        if self.layer_id < 0 or self.state_kind not in kinds or not self.position_policy:
            raise ValueError("invalid layer identity, state kind or position policy")
        dimensions = (self.query_heads, self.kv_heads, self.key_dim, self.value_dim)
        if any(d is not None and (not isinstance(d, int) or isinstance(d, bool) or d < 1)
               for d in dimensions):
            raise ValueError("layer dimensions must be positive integers")
        if self.state_kind in {"full_kv", "sliding_kv", "cross_kv"}:
            if any(d is None for d in dimensions):
                raise ValueError("attention requires explicit per-layer head dimensions")
            if self.query_heads % self.kv_heads:
                raise ValueError("query heads must be divisible by KV heads")
        if (self.state_kind == "sliding_kv" and self.window_tokens is None) or (
                self.window_tokens is not None and self.window_tokens < 1):
            raise ValueError("sliding state requires a positive window")


@dataclass(frozen=True)
class BatchCompatibilityKey:
    """All execution-altering identity; RNG and output limits remain request-local."""

    model_instance: str
    model_epoch: int
    adapter_version: str
    task: TaskKind
    stage: StageKind
    precision_policy: str
    state_schema: str
    position_policy: str
    computation_identity: str

    def __post_init__(self) -> None:
        if self.model_epoch < 0 or not all((self.model_instance, self.adapter_version,
                self.precision_policy, self.state_schema, self.position_policy,
                self.computation_identity)):
            raise ValueError("batch compatibility requires complete versioned identity")
        if not isinstance(self.task, TaskKind) or not isinstance(self.stage, StageKind):
            raise ValueError("batch compatibility requires canonical task and stage")


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    compatibility: BatchCompatibilityKey
    real_input_cost: int
    output_bound: int
    estimated_transient_bytes: int
    cost_unit: Literal["tokens", "samples", "pixels", "frames", "service_us"]
    payload: Any = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("work requires request ownership")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (
                self.real_input_cost, self.output_bound, self.estimated_transient_bytes)):
            raise ValueError("work costs must be nonnegative integers")
        if self.cost_unit not in {"tokens", "samples", "pixels", "frames", "service_us"}:
            raise ValueError("invalid task cost unit")


@runtime_checkable
class ExecutionAdapter(Protocol):
    """Shared description, costing and ownership; no task is forced to decode."""

    def describe(self, bundle: object) -> object: ...
    def prepare(self, payload: object) -> object: ...
    def estimate(self, prepared: object, plan: object) -> object: ...
    def pack(self, work: tuple[WorkItem, ...]) -> object: ...
    def execute(self, packed: object, state: object) -> object: ...
    def unpack(self, output: object) -> tuple[object, ...]: ...


class CausalAdapter(ExecutionAdapter, Protocol):
    def prefill(self, packed: object, state: object) -> object: ...
    def decode(self, packed: object, state: object) -> object: ...


class EncoderAdapter(ExecutionAdapter, Protocol):
    def encode(self, packed: object) -> object: ...


class EncoderDecoderAdapter(EncoderAdapter, Protocol):
    def decode(self, packed: object, state: object, encoded: object) -> object: ...


class TransducerAdapter(EncoderAdapter, Protocol):
    def predict(self, packed: object, state: object) -> object: ...
    def joint(self, encoded: object, predicted: object) -> object: ...


class AlignmentAdapter(ExecutionAdapter, Protocol):
    def align(self, packed: object) -> object: ...


class MultimodalAdapter(CausalAdapter, Protocol):
    def encode_media(self, prepared: object) -> object: ...


class TrackingAdapter(ExecutionAdapter, Protocol):
    def frame(self, packed: object, state: object) -> object: ...
