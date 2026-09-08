"""Versioned, backend-free schemas for mlx-one workflows."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeAlias

from mlx_one.schema_migrations import migrate_record

SCHEMA_VERSION = "1.0"

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class Modality(str, Enum):
    """Model input/output modality."""

    TEXT = "text"
    VISION_LANGUAGE = "vision-language"
    ASR = "asr"
    TTS = "tts"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class ModelSource(str, Enum):
    """Location from which model metadata was inspected."""

    HUGGING_FACE = "hugging-face"
    LOCAL = "local"


class CapabilityHintStatus(str, Enum):
    """Non-authoritative availability state for an adapter candidate."""

    CANDIDATE = "candidate"
    BACKEND_UNAVAILABLE = "backend-unavailable"
    UNKNOWN = "unknown"


class Operation(str, Enum):
    """Lifecycle operation represented by a run."""

    INSPECT = "inspect"
    PLAN = "plan"
    TRAIN = "train"
    CONVERT = "convert"
    VERIFY = "verify"
    EVALUATE = "evaluate"
    COMPARE = "compare"
    BENCHMARK = "benchmark"
    MERGE = "merge"
    SHIP = "ship"


class CachePolicy(str, Enum):
    """Permitted interaction with local and remote caches."""

    USE = "use"
    REFRESH = "refresh"
    OFFLINE = "offline"
    DISABLED = "disabled"


class RunStatus(str, Enum):
    """Terminal or intermediate run state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompatibilityStatus(str, Enum):
    """Evidence-backed compatibility classification."""

    OFFICIALLY_VERIFIED = "officially-verified"
    COMMUNITY_VERIFIED = "community-verified"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CapabilityStatus(str, Enum):
    """Operation-level support state used by the capability registry."""

    CANDIDATE = "candidate"
    UPSTREAM_DOCUMENTED = "upstream-documented"
    INTEGRATION_TESTED = "integration-tested"
    HARDWARE_VERIFIED = "hardware-verified"
    QUALITY_VERIFIED = "quality-verified"
    UNSUPPORTED = "unsupported"
    DEPRECATED = "deprecated"


class MetricProvenance(str, Enum):
    """How a reported metric value was obtained."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    NOT_AVAILABLE = "not-available"


class DatasetFormat(str, Enum):
    """Supported local text dataset representation."""

    AUTO = "auto"
    JSON = "json"
    JSONL = "jsonl"


class DatasetLayout(str, Enum):
    """Logical record layout used for supervised text data."""

    AUTO = "auto"
    INSTRUCTION = "instruction"
    MESSAGES = "messages"
    PROMPT_COMPLETION = "prompt-completion"
    TEXT = "text"


class Confidence(str, Enum):
    """Confidence assigned to a plan or compatibility claim."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WorkloadKind(str, Enum):
    """Text workload being planned or calibrated."""

    INFERENCE = "inference"
    TRAIN = "train"


class TrainingMethod(str, Enum):
    """Weight-update strategy for a training workload."""

    AUTO = "auto"
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    SCRATCH = "scratch"


class ResumeSemantics(str, Enum):
    """State guarantee offered by a training checkpoint."""

    ADAPTER_ONLY = "adapter-only"
    EXACT = "exact"


class Precision(str, Enum):
    """Storage or compute precision used for planning."""

    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class FitStatus(str, Enum):
    """Relationship between an estimate range and a hardware budget."""

    COMFORTABLE = "comfortable"
    POSSIBLE = "possible"
    RISKY = "risky"
    DOES_NOT_FIT = "does-not-fit"
    UNKNOWN = "unknown"


class CalibrationStatus(str, Enum):
    """Terminal status of one calibration matrix cell."""

    COMPLETED = "completed"
    SKIPPED_OVER_BUDGET = "skipped-over-budget"
    INVALID = "invalid"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _require_non_negative(value: int | float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} cannot be negative")


def _validate_timestamp(value: str | None, field_name: str) -> None:
    if value is None:
        return
    _require_text(value, field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a UTC offset")


def _json_mapping(value: Mapping[str, JSONValue], field_name: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) for key in result):
        raise TypeError(f"{field_name} keys must be strings")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain finite JSON values") from exc
    return result


def _string_mapping(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    result = dict(value)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in result.items()):
        raise TypeError(f"{field_name} keys and values must be strings")
    return result


def _enum(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _version(data: Mapping[str, Any]) -> str:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
    return version


def _serialize(value: object) -> JSONValue:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")


class SchemaMixin:
    """Stable JSON serialization shared by all schema records."""

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-compatible dictionary."""
        result = _serialize(self)
        if not isinstance(result, dict):  # pragma: no cover - dataclass invariant
            raise TypeError("schema did not serialize to an object")
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return deterministic JSON with finite numeric values."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


@dataclass(frozen=True, kw_only=True)
class ModelDimensions(SchemaMixin):
    """Normalized transformer dimensions used by memory planning."""

    hidden_size: int | None = None
    layer_count: int | None = None
    attention_heads: int | None = None
    key_value_heads: int | None = None
    head_dimension: int | None = None
    intermediate_size: int | None = None
    vocabulary_size: int | None = None

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            _require_non_negative(value, item.name)
            if value == 0:
                raise ValueError(f"{item.name} must be greater than zero")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelDimensions:
        """Construct normalized model dimensions."""
        return cls(**dict(data))


@dataclass(frozen=True, kw_only=True)
class ModelSpec(SchemaMixin):
    """Identity and static metadata for a model artifact."""

    schema_version: str = SCHEMA_VERSION
    model_id: str
    revision: str
    modality: Modality = Modality.UNKNOWN
    architectures: tuple[str, ...] = ()
    model_type: str | None = None
    base_models: tuple[str, ...] = ()
    library_name: str | None = None
    pipeline_tag: str | None = None
    parameter_count: int | None = None
    parameter_count_source: str | None = None
    weight_format: str | None = None
    weight_bytes: int | None = None
    dtype: str | None = None
    quantization: Mapping[str, JSONValue] = field(default_factory=dict)
    tokenizer_id: str | None = None
    processor_id: str | None = None
    license: str | None = None
    gated: bool = False
    private: bool = False
    disabled: bool = False
    requires_remote_code: bool = False
    dimensions: ModelDimensions = field(default_factory=ModelDimensions)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.model_id, "model_id")
        _require_text(self.revision, "revision")
        _require_non_negative(self.parameter_count, "parameter_count")
        _require_non_negative(self.weight_bytes, "weight_bytes")
        object.__setattr__(self, "modality", _enum(Modality, self.modality, "modality"))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "base_models", tuple(self.base_models))
        if not isinstance(self.dimensions, ModelDimensions):
            raise TypeError("dimensions must be a ModelDimensions record")
        object.__setattr__(
            self, "quantization", _json_mapping(self.quantization or {}, "quantization")
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        """Validate and construct a model specification."""
        payload = dict(data)
        _version(payload)
        payload["architectures"] = tuple(payload.get("architectures", ()))
        payload["base_models"] = tuple(payload.get("base_models", ()))
        if isinstance(payload.get("dimensions"), Mapping):
            payload["dimensions"] = ModelDimensions.from_dict(payload["dimensions"])
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> ModelSpec:
        """Construct a model specification from JSON."""
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class HardwareSpec(SchemaMixin):
    """Hardware and operating environment used by a workload."""

    schema_version: str = SCHEMA_VERSION
    profile_id: str
    platform: str
    architecture: str
    chip: str
    memory_bytes: int | None = None
    accelerator: str | None = None
    accelerator_count: int = 1
    accelerator_memory_bytes: int | None = None
    process_memory_budget_bytes: int | None = None
    system_reserve_bytes: int | None = None
    form_factor: str | None = None
    cpu_core_count: int | None = None
    gpu_core_count: int | None = None
    unified_memory: bool = False
    reference_device: bool = False
    profile_version: str = "1"
    provenance: str = "user"
    os_version: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        for field_name in ("profile_id", "platform", "architecture", "chip"):
            _require_text(getattr(self, field_name), field_name)
        _require_non_negative(self.memory_bytes, "memory_bytes")
        _require_non_negative(self.accelerator_memory_bytes, "accelerator_memory_bytes")
        _require_non_negative(self.process_memory_budget_bytes, "process_memory_budget_bytes")
        _require_non_negative(self.system_reserve_bytes, "system_reserve_bytes")
        _require_non_negative(self.cpu_core_count, "cpu_core_count")
        _require_non_negative(self.gpu_core_count, "gpu_core_count")
        if self.accelerator_count < 1:
            raise ValueError("accelerator_count must be at least 1")
        if self.cpu_core_count == 0 or self.gpu_core_count == 0:
            raise ValueError("core counts must be greater than zero")
        if self.memory_bytes is not None:
            budget = self.process_memory_budget_bytes
            reserve = self.system_reserve_bytes
            if budget is not None and budget > self.memory_bytes:
                raise ValueError("process_memory_budget_bytes cannot exceed memory_bytes")
            if reserve is not None and reserve > self.memory_bytes:
                raise ValueError("system_reserve_bytes cannot exceed memory_bytes")
            if budget is not None and reserve is not None and budget + reserve > self.memory_bytes:
                raise ValueError("process memory budget plus system reserve exceeds memory_bytes")
        _require_text(self.profile_version, "profile_version")
        _require_text(self.provenance, "provenance")
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareSpec:
        """Validate and construct a hardware specification."""
        payload = dict(data)
        _version(payload)
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> HardwareSpec:
        """Construct a hardware specification from JSON."""
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class WorkloadSpec(SchemaMixin):
    """Inputs that materially affect a text-workload memory estimate."""

    schema_version: str = SCHEMA_VERSION
    kind: WorkloadKind
    method: TrainingMethod | None = None
    precision: Precision = Precision.AUTO
    context_length: int = 2048
    batch_size: int = 1
    generation_length: int = 32
    lora_rank: int = 8
    lora_layers: int = 16
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        object.__setattr__(self, "kind", _enum(WorkloadKind, self.kind, "kind"))
        object.__setattr__(self, "precision", _enum(Precision, self.precision, "precision"))
        method = self.method
        if method is not None:
            method = _enum(TrainingMethod, method, "method")
            object.__setattr__(self, "method", method)
        if self.kind is WorkloadKind.INFERENCE and method is not None:
            raise ValueError("method is only valid for train workloads")
        if self.kind is WorkloadKind.TRAIN and method is None:
            raise ValueError("train workloads require a method")
        for field_name in (
            "context_length",
            "batch_size",
            "generation_length",
            "lora_rank",
            "lora_layers",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        _require_text(self.optimizer, "optimizer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkloadSpec:
        """Validate and construct a workload specification."""
        payload = dict(data)
        _version(payload)
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> WorkloadSpec:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class MemoryEstimate(SchemaMixin):
    """An uncertainty-aware peak-memory estimate for one workload."""

    schema_version: str = SCHEMA_VERSION
    workload: WorkloadSpec
    lower_bytes: int | None
    central_bytes: int | None
    upper_bytes: int | None
    budget_bytes: int | None
    fit: FitStatus
    confidence: Confidence
    components: Mapping[str, JSONValue] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    calibration_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        if not isinstance(self.workload, WorkloadSpec):
            raise TypeError("workload must be a WorkloadSpec")
        for field_name in ("lower_bytes", "central_bytes", "upper_bytes", "budget_bytes"):
            _require_non_negative(getattr(self, field_name), field_name)
        values = (self.lower_bytes, self.central_bytes, self.upper_bytes)
        if all(value is not None for value in values):
            lower, central, upper = values
            if not lower <= central <= upper:
                raise ValueError("memory estimate must satisfy lower <= central <= upper")
        object.__setattr__(self, "fit", _enum(FitStatus, self.fit, "fit"))
        object.__setattr__(self, "confidence", _enum(Confidence, self.confidence, "confidence"))
        object.__setattr__(self, "components", _json_mapping(self.components or {}, "components"))
        for field_name in ("assumptions", "warnings", "calibration_references"):
            values_tuple = tuple(getattr(self, field_name))
            if any(not isinstance(item, str) or not item.strip() for item in values_tuple):
                raise ValueError(f"{field_name} must contain non-empty strings")
            object.__setattr__(self, field_name, values_tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemoryEstimate:
        payload = dict(data)
        _version(payload)
        if isinstance(payload.get("workload"), Mapping):
            payload["workload"] = WorkloadSpec.from_dict(payload["workload"])
        for name in ("assumptions", "warnings", "calibration_references"):
            payload[name] = tuple(payload.get(name, ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> MemoryEstimate:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class PlanResult(SchemaMixin):
    """A metadata-only workload recommendation and its alternatives."""

    schema_version: str = SCHEMA_VERSION
    model: ModelSpec
    hardware: HardwareSpec
    requested_workload: WorkloadSpec
    recommended_workload: WorkloadSpec
    estimate: MemoryEstimate
    alternatives: tuple[MemoryEstimate, ...] = ()
    planned_at: str = ""

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if not isinstance(self.hardware, HardwareSpec):
            raise TypeError("hardware must be a HardwareSpec")
        if not isinstance(self.requested_workload, WorkloadSpec):
            raise TypeError("requested_workload must be a WorkloadSpec")
        if not isinstance(self.recommended_workload, WorkloadSpec):
            raise TypeError("recommended_workload must be a WorkloadSpec")
        if not isinstance(self.estimate, MemoryEstimate):
            raise TypeError("estimate must be a MemoryEstimate")
        if self.estimate.workload != self.recommended_workload:
            raise ValueError("estimate workload must match recommended_workload")
        if any(not isinstance(item, MemoryEstimate) for item in self.alternatives):
            raise TypeError("alternatives must contain MemoryEstimate records")
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        timestamp = self.planned_at or _utc_now()
        _validate_timestamp(timestamp, "planned_at")
        object.__setattr__(self, "planned_at", timestamp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanResult:
        payload = dict(data)
        _version(payload)
        if isinstance(payload.get("model"), Mapping):
            payload["model"] = ModelSpec.from_dict(payload["model"])
        if isinstance(payload.get("hardware"), Mapping):
            payload["hardware"] = HardwareSpec.from_dict(payload["hardware"])
        for name in ("requested_workload", "recommended_workload"):
            if isinstance(payload.get(name), Mapping):
                payload[name] = WorkloadSpec.from_dict(payload[name])
        if isinstance(payload.get("estimate"), Mapping):
            payload["estimate"] = MemoryEstimate.from_dict(payload["estimate"])
        payload["alternatives"] = tuple(
            item if isinstance(item, MemoryEstimate) else MemoryEstimate.from_dict(item)
            for item in payload.get("alternatives", ())
        )
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> PlanResult:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class RunSpec(SchemaMixin):
    """Reproducible inputs and policies for one lifecycle operation."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    operation: Operation
    model: ModelSpec
    profile: str | None = None
    hardware: HardwareSpec | None = None
    dataset_revisions: Mapping[str, str] = field(default_factory=dict)
    generation: Mapping[str, JSONValue] = field(default_factory=dict)
    seed: int = 0
    limits: Mapping[str, JSONValue] = field(default_factory=dict)
    cache_policy: CachePolicy = CachePolicy.USE
    parent_run_ids: tuple[str, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.run_id, "run_id")
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if self.hardware is not None and not isinstance(self.hardware, HardwareSpec):
            raise TypeError("hardware must be a HardwareSpec")
        object.__setattr__(self, "operation", _enum(Operation, self.operation, "operation"))
        object.__setattr__(
            self, "cache_policy", _enum(CachePolicy, self.cache_policy, "cache_policy")
        )
        object.__setattr__(
            self,
            "dataset_revisions",
            _string_mapping(self.dataset_revisions or {}, "dataset_revisions"),
        )
        object.__setattr__(self, "generation", _json_mapping(self.generation or {}, "generation"))
        object.__setattr__(self, "limits", _json_mapping(self.limits or {}, "limits"))
        parents = tuple(self.parent_run_ids)
        if any(not isinstance(item, str) or not item.strip() for item in parents):
            raise ValueError("parent_run_ids must contain non-empty strings")
        if self.run_id in parents:
            raise ValueError("a run cannot be its own parent")
        object.__setattr__(self, "parent_run_ids", parents)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))
        timestamp = self.created_at or _utc_now()
        _validate_timestamp(timestamp, "created_at")
        object.__setattr__(self, "created_at", timestamp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunSpec:
        """Validate and construct a run specification."""
        payload = migrate_record("run", data)
        _version(payload)
        if isinstance(payload.get("model"), Mapping):
            payload["model"] = ModelSpec.from_dict(payload["model"])
        if isinstance(payload.get("hardware"), Mapping):
            payload["hardware"] = HardwareSpec.from_dict(payload["hardware"])
        payload["parent_run_ids"] = tuple(payload.get("parent_run_ids", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> RunSpec:
        """Construct a run specification from JSON."""
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class Failure(SchemaMixin):
    """Structured failure retained in a partial or failed run."""

    code: str
    message: str
    task: str | None = None
    retryable: bool = False
    details: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.code, "code")
        _require_text(self.message, "message")
        object.__setattr__(self, "details", _json_mapping(self.details or {}, "details"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Failure:
        """Construct a structured failure."""
        return cls(**dict(data))


@dataclass(frozen=True, kw_only=True)
class TaskResult(SchemaMixin):
    """Normalized result for one evaluation or benchmark task."""

    task: str
    metrics: Mapping[str, JSONValue]
    sample_count: int | None = None
    failures: tuple[Failure, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.task, "task")
        _require_non_negative(self.sample_count, "sample_count")
        if any(not isinstance(item, Failure) for item in self.failures):
            raise TypeError("failures must contain Failure records")
        object.__setattr__(self, "metrics", _json_mapping(self.metrics, "metrics"))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskResult:
        """Construct a normalized task result."""
        payload = dict(data)
        payload["failures"] = tuple(
            item if isinstance(item, Failure) else Failure.from_dict(item)
            for item in payload.get("failures", ())
        )
        return cls(**payload)


@dataclass(frozen=True, kw_only=True)
class ArtifactRef(SchemaMixin):
    """Reference to a local or remote artifact produced by a run."""

    schema_version: str = SCHEMA_VERSION
    uri: str
    kind: str
    sha256: str | None = None
    media_type: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.uri, "uri")
        _require_text(self.kind, "kind")
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRef:
        """Construct an artifact reference."""
        return cls(**migrate_record("artifact", data))


@dataclass(frozen=True, kw_only=True)
class MetricValue(SchemaMixin):
    """A metric whose value cannot be confused with its provenance."""

    value: JSONScalar = None
    provenance: MetricProvenance
    unit: str | None = None
    protocol: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provenance", _enum(MetricProvenance, self.provenance, "provenance")
        )
        if self.provenance is MetricProvenance.NOT_AVAILABLE and self.value is not None:
            raise ValueError("not-available metrics cannot contain a value")
        if self.provenance is not MetricProvenance.NOT_AVAILABLE and self.value is None:
            raise ValueError("measured and estimated metrics require a value")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        for name in ("unit", "protocol"):
            value = getattr(self, name)
            if value is not None:
                _require_text(value, name)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricValue:
        return cls(**dict(data))


@dataclass(frozen=True, kw_only=True)
class DatasetSpec(SchemaMixin):
    """Immutable description of local supervised or evaluation data."""

    schema_version: str = SCHEMA_VERSION
    dataset_id: str
    source: str
    revision: str
    split: str = "train"
    format: DatasetFormat = DatasetFormat.AUTO
    layout: DatasetLayout = DatasetLayout.AUTO
    columns: Mapping[str, str] = field(default_factory=dict)
    response_only: bool = True
    private: bool = True
    sha256: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        for name in ("dataset_id", "source", "revision", "split"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "format", _enum(DatasetFormat, self.format, "format"))
        object.__setattr__(self, "layout", _enum(DatasetLayout, self.layout, "layout"))
        object.__setattr__(self, "columns", _string_mapping(self.columns or {}, "columns"))
        if self.sha256 is not None:
            digest = self.sha256.lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("sha256 must be a 64-character hexadecimal digest")
            object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetSpec:
        payload = migrate_record("dataset", data)
        _version(payload)
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> DatasetSpec:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class TrainConfig(SchemaMixin):
    """Backend-neutral configuration for the supported text SFT subset."""

    schema_version: str = SCHEMA_VERSION
    output_dir: str
    method: TrainingMethod = TrainingMethod.LORA
    max_seq_length: int = 2048
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    learning_rate: float = 2e-4
    optimizer: str = "adamw"
    seed: int = 42
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_layers: int = 16
    target_modules: tuple[str, ...] = ()
    gradient_checkpointing: bool = False
    save_steps: int = 100
    eval_steps: int = 0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.output_dir, "output_dir")
        _require_text(self.optimizer, "optimizer")
        object.__setattr__(self, "method", _enum(TrainingMethod, self.method, "method"))
        if self.method not in {TrainingMethod.LORA, TrainingMethod.QLORA}:
            raise ValueError("text SFT currently supports only lora and qlora")
        for name in (
            "max_seq_length",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "lora_rank",
            "lora_layers",
            "save_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.eval_steps < 0:
            raise ValueError("eval_steps cannot be negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be a positive finite number")
        if not math.isfinite(self.lora_dropout) or not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        modules = tuple(self.target_modules)
        if any(not isinstance(item, str) or not item.strip() for item in modules):
            raise ValueError("target_modules must contain non-empty strings")
        object.__setattr__(self, "target_modules", modules)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        payload = dict(data)
        _version(payload)
        payload["target_modules"] = tuple(payload.get("target_modules", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> TrainConfig:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class CheckpointMetadata(SchemaMixin):
    """Lineage and resumability metadata stored beside adapter weights."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    base_model_id: str
    base_revision: str
    method: TrainingMethod
    step: int
    adapter: ArtifactRef
    config_sha256: str
    resumable_state: tuple[str, ...] = ()
    resume_semantics: ResumeSemantics = ResumeSemantics.ADAPTER_ONLY
    state_artifacts: tuple[ArtifactRef, ...] = ()
    created_at: str = ""

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        for name in ("run_id", "base_model_id", "base_revision"):
            _require_text(getattr(self, name), name)
        object.__setattr__(self, "method", _enum(TrainingMethod, self.method, "method"))
        if self.step < 0:
            raise ValueError("step cannot be negative")
        if not isinstance(self.adapter, ArtifactRef):
            raise TypeError("adapter must be an ArtifactRef")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.config_sha256):
            raise ValueError("config_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "config_sha256", self.config_sha256.lower())
        object.__setattr__(
            self,
            "resume_semantics",
            _enum(ResumeSemantics, self.resume_semantics, "resume_semantics"),
        )
        states = tuple(self.resumable_state)
        if any(not isinstance(item, str) or not item.strip() for item in states):
            raise ValueError("resumable_state must contain non-empty strings")
        object.__setattr__(self, "resumable_state", states)
        if any(not isinstance(item, ArtifactRef) for item in self.state_artifacts):
            raise TypeError("state_artifacts must contain ArtifactRef records")
        state_artifacts = tuple(self.state_artifacts)
        object.__setattr__(self, "state_artifacts", state_artifacts)
        exact_state = {"adapter-weights", "optimizer", "scheduler", "rng", "step"}
        if self.resume_semantics is ResumeSemantics.EXACT and not exact_state.issubset(states):
            raise ValueError(
                "exact resume requires adapter, optimizer, scheduler, RNG, and step state"
            )
        timestamp = self.created_at or _utc_now()
        _validate_timestamp(timestamp, "created_at")
        object.__setattr__(self, "created_at", timestamp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointMetadata:
        payload = migrate_record("checkpoint", data)
        _version(payload)
        if isinstance(payload.get("adapter"), Mapping):
            payload["adapter"] = ArtifactRef.from_dict(payload["adapter"])
        payload["resumable_state"] = tuple(payload.get("resumable_state", ()))
        payload["state_artifacts"] = tuple(
            item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
            for item in payload.get("state_artifacts", ())
        )
        return cls(**payload)


@dataclass(frozen=True, kw_only=True)
class CapabilitySpec(SchemaMixin):
    """Evidence-scoped support declaration for one model operation."""

    schema_version: str = SCHEMA_VERSION
    model: ModelSpec
    backend: str
    operation: Operation
    status: CapabilityStatus
    constraints: Mapping[str, JSONValue] = field(default_factory=dict)
    hardware_profile_ids: tuple[str, ...] = ()
    evidence: tuple[ArtifactRef, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        _require_text(self.backend, "backend")
        object.__setattr__(self, "operation", _enum(Operation, self.operation, "operation"))
        object.__setattr__(self, "status", _enum(CapabilityStatus, self.status, "status"))
        object.__setattr__(
            self, "constraints", _json_mapping(self.constraints or {}, "constraints")
        )
        profiles = tuple(self.hardware_profile_ids)
        notes = tuple(self.notes)
        if any(not isinstance(item, str) or not item.strip() for item in profiles):
            raise ValueError("hardware_profile_ids must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in notes):
            raise ValueError("notes must contain non-empty strings")
        if any(not isinstance(item, ArtifactRef) for item in self.evidence):
            raise TypeError("evidence must contain ArtifactRef records")
        object.__setattr__(self, "hardware_profile_ids", profiles)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "notes", notes)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilitySpec:
        payload = migrate_record("capability", data)
        _version(payload)
        if isinstance(payload.get("model"), Mapping):
            payload["model"] = ModelSpec.from_dict(payload["model"])
        payload["hardware_profile_ids"] = tuple(payload.get("hardware_profile_ids", ()))
        payload["evidence"] = tuple(
            item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
            for item in payload.get("evidence", ())
        )
        payload["notes"] = tuple(payload.get("notes", ()))
        return cls(**payload)


@dataclass(frozen=True, kw_only=True)
class SoftwareProvenance(SchemaMixin):
    """Software environment needed to interpret or reproduce a run."""

    python_version: str
    packages: Mapping[str, str]
    os_version: str | None = None
    git_commit: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.python_version, "python_version")
        object.__setattr__(self, "packages", _string_mapping(self.packages, "packages"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SoftwareProvenance:
        """Construct software provenance."""
        return cls(**dict(data))


@dataclass(frozen=True, kw_only=True)
class CalibrationRecord(SchemaMixin):
    """Sanitized evidence produced by one calibration matrix cell."""

    schema_version: str = SCHEMA_VERSION
    matrix_id: str
    cell_id: str
    status: CalibrationStatus
    model: ModelSpec
    hardware: HardwareSpec
    workload: WorkloadSpec
    software: SoftwareProvenance
    metrics: Mapping[str, JSONValue] = field(default_factory=dict)
    memory: Mapping[str, JSONValue] = field(default_factory=dict)
    environment: Mapping[str, JSONValue] = field(default_factory=dict)
    failure: Failure | None = None
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.matrix_id, "matrix_id")
        _require_text(self.cell_id, "cell_id")
        object.__setattr__(self, "status", _enum(CalibrationStatus, self.status, "status"))
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if not isinstance(self.hardware, HardwareSpec):
            raise TypeError("hardware must be a HardwareSpec")
        if not isinstance(self.workload, WorkloadSpec):
            raise TypeError("workload must be a WorkloadSpec")
        if not isinstance(self.software, SoftwareProvenance):
            raise TypeError("software must be SoftwareProvenance")
        if self.failure is not None and not isinstance(self.failure, Failure):
            raise TypeError("failure must be a Failure record")
        object.__setattr__(self, "metrics", _json_mapping(self.metrics or {}, "metrics"))
        object.__setattr__(self, "memory", _json_mapping(self.memory or {}, "memory"))
        object.__setattr__(
            self, "environment", _json_mapping(self.environment or {}, "environment")
        )
        _validate_timestamp(self.started_at, "started_at")
        _validate_timestamp(self.ended_at, "ended_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationRecord:
        payload = dict(data)
        _version(payload)
        constructors = {
            "model": ModelSpec.from_dict,
            "hardware": HardwareSpec.from_dict,
            "workload": WorkloadSpec.from_dict,
            "software": SoftwareProvenance.from_dict,
            "failure": Failure.from_dict,
        }
        for name, constructor in constructors.items():
            if isinstance(payload.get(name), Mapping):
                payload[name] = constructor(payload[name])
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> CalibrationRecord:
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class RunResult(SchemaMixin):
    """Versioned outcome, evidence, and provenance for a run."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    status: RunStatus
    spec: RunSpec
    metrics: Mapping[str, JSONValue] = field(default_factory=dict)
    task_results: tuple[TaskResult, ...] = ()
    failures: tuple[Failure, ...] = ()
    timings: Mapping[str, JSONValue] = field(default_factory=dict)
    memory: Mapping[str, JSONValue] = field(default_factory=dict)
    software: SoftwareProvenance | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    started_at: str | None = None
    ended_at: str | None = None

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        _require_text(self.run_id, "run_id")
        if not isinstance(self.spec, RunSpec):
            raise TypeError("spec must be a RunSpec")
        if self.run_id != self.spec.run_id:
            raise ValueError("run_id must match spec.run_id")
        if self.software is not None and not isinstance(self.software, SoftwareProvenance):
            raise TypeError("software must be SoftwareProvenance")
        if any(not isinstance(item, TaskResult) for item in self.task_results):
            raise TypeError("task_results must contain TaskResult records")
        if any(not isinstance(item, Failure) for item in self.failures):
            raise TypeError("failures must contain Failure records")
        if any(not isinstance(item, ArtifactRef) for item in self.artifacts):
            raise TypeError("artifacts must contain ArtifactRef records")
        object.__setattr__(self, "status", _enum(RunStatus, self.status, "status"))
        object.__setattr__(self, "metrics", _json_mapping(self.metrics or {}, "metrics"))
        object.__setattr__(self, "timings", _json_mapping(self.timings or {}, "timings"))
        object.__setattr__(self, "memory", _json_mapping(self.memory or {}, "memory"))
        object.__setattr__(self, "task_results", tuple(self.task_results))
        object.__setattr__(self, "failures", tuple(self.failures))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        _validate_timestamp(self.started_at, "started_at")
        _validate_timestamp(self.ended_at, "ended_at")
        if self.started_at and self.ended_at:
            started = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            ended = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
            if ended < started:
                raise ValueError("ended_at cannot be earlier than started_at")
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float) and not math.isfinite(metric):
                raise ValueError(f"metric {metric_name!r} must be finite")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunResult:
        """Validate and construct a run result."""
        payload = migrate_record("run_result", data)
        _version(payload)
        if isinstance(payload.get("spec"), Mapping):
            payload["spec"] = RunSpec.from_dict(payload["spec"])
        payload["task_results"] = tuple(
            item if isinstance(item, TaskResult) else TaskResult.from_dict(item)
            for item in payload.get("task_results", ())
        )
        payload["failures"] = tuple(
            item if isinstance(item, Failure) else Failure.from_dict(item)
            for item in payload.get("failures", ())
        )
        payload["artifacts"] = tuple(
            item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
            for item in payload.get("artifacts", ())
        )
        if isinstance(payload.get("software"), Mapping):
            payload["software"] = SoftwareProvenance.from_dict(payload["software"])
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> RunResult:
        """Construct a run result from JSON."""
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class CapabilityHint(SchemaMixin):
    """Cautious adapter hint that does not assert verified compatibility."""

    adapter: str
    status: CapabilityHintStatus
    operations: tuple[Operation, ...] = ()
    installed_version: str | None = None
    reason: str

    def __post_init__(self) -> None:
        _require_text(self.adapter, "adapter")
        _require_text(self.reason, "reason")
        object.__setattr__(
            self,
            "status",
            _enum(CapabilityHintStatus, self.status, "status"),
        )
        object.__setattr__(
            self,
            "operations",
            tuple(_enum(Operation, item, "operations") for item in self.operations),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityHint:
        """Construct an adapter capability hint."""
        payload = dict(data)
        payload["operations"] = tuple(payload.get("operations", ()))
        return cls(**payload)


@dataclass(frozen=True, kw_only=True)
class InspectionResult(SchemaMixin):
    """Normalized metadata and cautious capability hints for one model."""

    schema_version: str = SCHEMA_VERSION
    model: ModelSpec
    source: ModelSource
    requested_revision: str | None = None
    capabilities: tuple[CapabilityHint, ...] = ()
    warnings: tuple[str, ...] = ()
    inspected_at: str = ""

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if any(not isinstance(item, CapabilityHint) for item in self.capabilities):
            raise TypeError("capabilities must contain CapabilityHint records")
        if any(not isinstance(item, str) or not item.strip() for item in self.warnings):
            raise ValueError("warnings must contain non-empty strings")
        object.__setattr__(self, "source", _enum(ModelSource, self.source, "source"))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        timestamp = self.inspected_at or _utc_now()
        _validate_timestamp(timestamp, "inspected_at")
        object.__setattr__(self, "inspected_at", timestamp)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InspectionResult:
        """Validate and construct an inspection result."""
        payload = dict(data)
        _version(payload)
        if isinstance(payload.get("model"), Mapping):
            payload["model"] = ModelSpec.from_dict(payload["model"])
        payload["capabilities"] = tuple(
            item if isinstance(item, CapabilityHint) else CapabilityHint.from_dict(item)
            for item in payload.get("capabilities", ())
        )
        payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> InspectionResult:
        """Construct an inspection result from JSON."""
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, kw_only=True)
class CompatibilityResult(SchemaMixin):
    """Compatibility claim scoped to model, workload, backend, and hardware."""

    schema_version: str = SCHEMA_VERSION
    model: ModelSpec
    hardware: HardwareSpec
    operation: Operation
    backend: str
    status: CompatibilityStatus
    confidence: Confidence
    workload: WorkloadSpec | None = None
    estimated_peak_memory_bytes: int | None = None
    reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    evidence: tuple[ArtifactRef, ...] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _version({"schema_version": self.schema_version})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("model must be a ModelSpec")
        if not isinstance(self.hardware, HardwareSpec):
            raise TypeError("hardware must be a HardwareSpec")
        if self.workload is not None and not isinstance(self.workload, WorkloadSpec):
            raise TypeError("workload must be a WorkloadSpec")
        _require_text(self.backend, "backend")
        _require_non_negative(self.estimated_peak_memory_bytes, "estimated_peak_memory_bytes")
        if any(not isinstance(item, str) or not item.strip() for item in self.reasons):
            raise ValueError("reasons must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.recommendations):
            raise ValueError("recommendations must contain non-empty strings")
        if any(not isinstance(item, ArtifactRef) for item in self.evidence):
            raise TypeError("evidence must contain ArtifactRef records")
        object.__setattr__(self, "operation", _enum(Operation, self.operation, "operation"))
        object.__setattr__(self, "status", _enum(CompatibilityStatus, self.status, "status"))
        object.__setattr__(self, "confidence", _enum(Confidence, self.confidence, "confidence"))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "recommendations", tuple(self.recommendations))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata or {}, "metadata"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompatibilityResult:
        """Validate and construct a compatibility result."""
        payload = dict(data)
        _version(payload)
        if isinstance(payload.get("model"), Mapping):
            payload["model"] = ModelSpec.from_dict(payload["model"])
        if isinstance(payload.get("hardware"), Mapping):
            payload["hardware"] = HardwareSpec.from_dict(payload["hardware"])
        if isinstance(payload.get("workload"), Mapping):
            payload["workload"] = WorkloadSpec.from_dict(payload["workload"])
        payload["evidence"] = tuple(
            item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
            for item in payload.get("evidence", ())
        )
        payload["reasons"] = tuple(payload.get("reasons", ()))
        payload["recommendations"] = tuple(payload.get("recommendations", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> CompatibilityResult:
        """Construct a compatibility result from JSON."""
        return cls.from_dict(json.loads(value))
