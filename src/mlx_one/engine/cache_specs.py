"""Stable cache topology, configuration, identity, and lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

CacheBackend = Literal["dense", "block", "paged", "hybrid"]
CACHE_BACKENDS = ("dense", "block", "paged", "hybrid")


class CacheKind(str, Enum):
    ATTENTION_KV = "attention_kv"
    ROTATING_KV = "rotating_kv"
    QUANTIZED_KV = "quantized_kv"
    RECURRENT_STATE = "recurrent_state"
    CONV_STATE = "conv_state"
    CROSS_ATTENTION = "cross_attention"
    MODEL_DEFINED = "model_defined"


@dataclass(frozen=True)
class LayerCacheSpec:
    layer_id: int
    kind: CacheKind
    num_kv_heads: int | None = None
    key_head_dim: int | None = None
    value_head_dim: int | None = None
    dtype: str | None = None
    max_window_tokens: int | None = None
    quantizable: bool = False
    shareable_prefix: bool = True
    trimmable: bool = True
    batchable: bool = True
    block_compatible: bool = False
    state_shapes: tuple[tuple[int, ...], ...] = ()
    checkpoint_semantics: str | None = None
    num_query_heads: int | None = None
    position_policy: str = "rope"
    state_dtypes: tuple[str, ...] = ()

    def bytes_per_token(self, *, key_bits: int = 16, value_bits: int = 16) -> int:
        if self.kind not in {
            CacheKind.ATTENTION_KV,
            CacheKind.ROTATING_KV,
            CacheKind.QUANTIZED_KV,
        }:
            return 0
        if self.num_kv_heads is None or self.key_head_dim is None:
            return 0
        value_dim = self.value_head_dim or self.key_head_dim
        key_bytes = (self.num_kv_heads * self.key_head_dim * key_bits + 7) // 8
        value_bytes = (self.num_kv_heads * value_dim * value_bits + 7) // 8
        return key_bytes + value_bytes


@dataclass(frozen=True)
class LayerStateSpec:
    """Runtime state schema; implementation ability is separate from qualification."""

    layer_id: int
    kind: Literal["full_kv", "sliding_kv", "conv", "recurrent", "cross_kv", "unknown"]
    num_kv_heads: int | None = None
    key_head_dim: int | None = None
    value_head_dim: int | None = None
    dtype: str | None = None
    window_tokens: int | None = None
    state_shapes: tuple[tuple[int, ...], ...] = ()
    checkpoint_semantics: str | None = None
    can_batch: bool = False
    can_page: bool = False
    can_snapshot: bool = False
    can_quantize: bool = False
    can_share_prefix: bool = False
    state_dtypes: tuple[str, ...] = ()

    @classmethod
    def from_cache_spec(cls, spec: LayerCacheSpec) -> LayerStateSpec:
        kinds = {
            CacheKind.ATTENTION_KV: "full_kv",
            CacheKind.QUANTIZED_KV: "full_kv",
            CacheKind.ROTATING_KV: "sliding_kv",
            CacheKind.CONV_STATE: "conv",
            CacheKind.RECURRENT_STATE: "recurrent",
            CacheKind.CROSS_ATTENTION: "cross_kv",
        }
        # Block compatibility describes storage, not direct page-table attention.
        kind = kinds.get(spec.kind, "unknown")
        return cls(
            spec.layer_id, kind, spec.num_kv_heads, spec.key_head_dim,
            spec.value_head_dim or spec.key_head_dim, None, spec.max_window_tokens,
            state_shapes=spec.state_shapes, checkpoint_semantics=spec.checkpoint_semantics,
            state_dtypes=spec.state_dtypes,
            can_batch=spec.batchable and kind == "full_kv",
            can_snapshot=spec.shareable_prefix and kind == "full_kv",
            can_quantize=spec.quantizable and kind == "full_kv",
            can_share_prefix=spec.shareable_prefix and kind == "full_kv",
        )

    def to_cache_spec(self) -> LayerCacheSpec:
        kinds = {
            "full_kv": CacheKind.ATTENTION_KV,
            "sliding_kv": CacheKind.ROTATING_KV,
            "conv": CacheKind.CONV_STATE,
            "recurrent": CacheKind.RECURRENT_STATE,
            "cross_kv": CacheKind.CROSS_ATTENTION,
            "unknown": CacheKind.MODEL_DEFINED,
        }
        return LayerCacheSpec(
            self.layer_id, kinds[self.kind], self.num_kv_heads, self.key_head_dim,
            self.value_head_dim, self.dtype, self.window_tokens,
            quantizable=self.can_quantize, shareable_prefix=self.can_share_prefix,
            trimmable=False, batchable=self.can_batch, block_compatible=False,
            state_shapes=self.state_shapes, checkpoint_semantics=self.checkpoint_semantics,
            state_dtypes=self.state_dtypes,
        )


@dataclass(frozen=True)
class CachePlan:
    model_id: str
    revision: str | None
    architecture: str
    layer_specs: tuple[LayerCacheSpec, ...]
    max_context_tokens: int
    block_size_tokens: int = 32

    @property
    def state_specs(self) -> tuple[LayerStateSpec, ...]:
        return tuple(LayerStateSpec.from_cache_spec(spec) for spec in self.layer_specs)

    @property
    def block_compatible(self) -> bool:
        return bool(self.layer_specs) and all(item.block_compatible for item in self.layer_specs)

    @property
    def prefix_shareable(self) -> bool:
        return bool(self.layer_specs) and all(item.shareable_prefix for item in self.layer_specs)

    def bytes_per_token(self, *, key_bits: int = 16, value_bits: int = 16) -> int:
        return sum(
            item.bytes_per_token(key_bits=key_bits, value_bits=value_bits)
            for item in self.layer_specs
        )

    def carry_bytes(self) -> int:
        """Declared private carry storage; unknown floating types use FP32 bounds."""
        import math

        return sum(math.prod(shape) * (2 if index < len(spec.state_dtypes)
                   and spec.state_dtypes[index] in {"float16","bfloat16"} else 4)
                   for spec in self.layer_specs
                   if spec.kind in {CacheKind.CONV_STATE,CacheKind.RECURRENT_STATE}
                   for index,shape in enumerate(spec.state_shapes))

    @property
    def topology_digest(self) -> str:
        payload = {
            "architecture": self.architecture,
            "layers": [asdict(item) for item in self.layer_specs],
            "context": self.max_context_tokens,
            "block": self.block_size_tokens,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class CacheConfig:
    backend: CacheBackend = "dense"
    memory_budget_bytes: int | None = None
    prefix_cache_enabled: bool = True
    prefix_cache_budget_bytes: int = 512 * 1024**2
    block_size_tokens: int = 32
    key_type: str = "f16"
    value_type: str = "f16"
    ssd_cache_directory: str | None = None
    ssd_cache_budget_bytes: int = 0
    ssd_staging_budget_bytes: int = 256 * 1024**2
    ssd_max_entries: int = 128

    def __post_init__(self) -> None:
        if (self.ssd_cache_budget_bytes<0 or self.ssd_staging_budget_bytes<1
                or self.ssd_max_entries<1 or bool(self.ssd_cache_directory) !=
                (self.ssd_cache_budget_bytes>0)):
            raise ValueError("SSD prefixes require explicit directory and positive budgets")
        if self.backend not in CACHE_BACKENDS:
            raise ValueError("cache backend must be " + ", ".join(CACHE_BACKENDS))
        if self.memory_budget_bytes is not None and self.memory_budget_bytes < 1:
            raise ValueError("cache memory budget must be positive")
        if self.prefix_cache_budget_bytes < 0:
            raise ValueError("prefix cache budget cannot be negative")
        if self.block_size_tokens not in {16, 32, 64, 128}:
            raise ValueError("cache block size must be 16, 32, 64, or 128")
        if self.key_type not in {"f16", "bf16", "q4_0", "q8_0"} or self.value_type not in {
            "f16", "bf16", "q4_0", "q8_0"
        }:
            raise ValueError("unsupported configured cache precision")


@dataclass(frozen=True)
class CacheFingerprint:
    model_id: str
    revision: str | None
    model_identity: str
    adapter_identity: str | None
    tokenizer_identity: str
    template_identity: str
    topology_digest: str
    key_type: str
    value_type: str
    context_tokens: int
    rope_identity: str
    namespace: str = "private"
    special_tokens_identity: str = ""
    rendering_identity: str = ""
    multimodal_identity: str | None = None
    execution_identity: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class ReservationState(str, Enum):
    REQUESTED = "requested"
    GRANTED = "granted"
    COMMITTED = "committed"
    RELEASED = "released"


@dataclass
class CacheReservation:
    reservation_id: str
    handle_id: str
    requested_bytes: int
    requested_tokens: int
    state: ReservationState = ReservationState.REQUESTED
    committed_bytes: int = 0
    manager_id: str | None = None


@dataclass
class CacheHandle:
    handle_id: str
    request_id: str
    sequence_id: str
    plan: CachePlan
    bundle: Any
    backend: CacheBackend
    token_count: int = 0
    computed_tokens: int = 0
    prefix_tokens_reused: int = 0
    reserved_bytes: int = 0
    closed: bool = False
    reservation: CacheReservation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CacheMemoryStats:
    budget_bytes: int | None
    logical_bytes: int
    allocated_bytes: int
    physical_bytes: int
    prefix_bytes: int
    reserved_bytes: int
    pinned_bytes: int
    evictable_bytes: int
    active_handles: int
    block_total: int = 0
    block_free: int = 0
    block_live: int = 0
    block_prefix: int = 0
    block_shared: int = 0
    retained_free_bytes: int = 0
    unique_live_physical_bytes: int = 0
    shared_page_savings_bytes: int = 0
    prefix_metadata_bytes: int = 0
    prefix_entries: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    prefix_evictions: int = 0
    prefix_tokens_reused: int = 0
    admission_failures: int = 0
    prefix_lookup_time_ms: float = 0.0
    allocation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, int | None | float]:
        result: dict[str, int | None | float] = asdict(self)
        lookups = self.prefix_hits + self.prefix_misses
        result["prefix_hit_rate"] = self.prefix_hits / lookups if lookups else 0.0
        return result
