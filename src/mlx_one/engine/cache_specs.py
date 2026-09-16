"""Stable cache topology, configuration, identity, and lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


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
class CachePlan:
    model_id: str
    revision: str | None
    architecture: str
    layer_specs: tuple[LayerCacheSpec, ...]
    max_context_tokens: int
    block_size_tokens: int = 32

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
    backend: Literal["dense", "block"] = "dense"
    memory_budget_bytes: int | None = None
    prefix_cache_enabled: bool = True
    prefix_cache_budget_bytes: int = 512 * 1024**2
    block_size_tokens: int = 32

    def __post_init__(self) -> None:
        if self.backend not in {"dense", "block"}:
            raise ValueError("cache backend must be dense or block")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes < 1:
            raise ValueError("cache memory budget must be positive")
        if self.prefix_cache_budget_bytes < 0:
            raise ValueError("prefix cache budget cannot be negative")
        if self.block_size_tokens not in {16, 32, 64, 128}:
            raise ValueError("cache block size must be 16, 32, 64, or 128")


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
    namespace: str = "global"

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


@dataclass
class CacheHandle:
    handle_id: str
    request_id: str
    sequence_id: str
    plan: CachePlan
    bundle: Any
    backend: Literal["dense", "block"]
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
