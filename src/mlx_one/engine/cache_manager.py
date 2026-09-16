"""Central cache ownership, prefix reuse, accounting, and reservations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from mlx_one.core.cache import configure_kv_caches
from mlx_one.engine.block_cache import BlockPool
from mlx_one.engine.block_kv import BlockKVCache
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_errors import (
    CacheCapacityError,
    CacheOwnershipError,
    CacheReservationError,
    CacheTopologyUnsupported,
    KVQuantizationUnsupported,
    PrefixReuseUnsupported,
)
from mlx_one.engine.cache_planner import build_cache_plan
from mlx_one.engine.cache_specs import (
    CacheConfig,
    CacheFingerprint,
    CacheHandle,
    CacheMemoryStats,
    CacheReservation,
    ReservationState,
)
from mlx_one.engine.prefix_cache import BlockPrefixIndex, PrefixCache


class CacheManager:
    """Model-scoped owner of all cache state and capacity decisions."""

    def __init__(self, bundle: Any, config: CacheConfig | None = None) -> None:
        self.bundle = bundle
        self.config = config or CacheConfig()
        self.plan = build_cache_plan(bundle, block_size_tokens=self.config.block_size_tokens)
        capacity = self.config.memory_budget_bytes or (2**63 - 1)
        self.block_pool = BlockPool(
            capacity,
            block_tokens=self.config.block_size_tokens,
        )
        self.snapshot_prefix = PrefixCache(self.config.prefix_cache_budget_bytes)
        self.block_prefix = BlockPrefixIndex(
            self.config.prefix_cache_budget_bytes,
            block_tokens=self.config.block_size_tokens,
        )
        self._handles: dict[str, CacheHandle] = {}
        self._reservations: dict[str, CacheReservation] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._admission_failures = 0
        self._prefix_lookup_time_ms = 0.0
        self._allocation_time_ms = 0.0

    def fingerprint(
        self,
        *,
        key_type: str = "f16",
        value_type: str = "f16",
        context_tokens: int | None = None,
        namespace: str = "global",
    ) -> CacheFingerprint:
        config = getattr(self.bundle.model, "config", None)
        rope = {
            name: getattr(getattr(config, "text_config", config), name, None)
            for name in ("rope_theta", "rope_scaling", "rope_freq_constant", "rope_max_length")
        }
        template = getattr(getattr(self.bundle, "chat_template", None), "template", None)
        return CacheFingerprint(
            model_id=str(self.bundle.model_id),
            revision=self.bundle.revision,
            model_identity=_model_identity(self.bundle),
            adapter_identity=getattr(self.bundle, "adapter_identity", None),
            tokenizer_identity=(
                getattr(self.bundle, "tokenizer_identity", None)
                or _object_identity(self.bundle.tokenizer)
            ),
            template_identity=_digest(template or ""),
            topology_digest=self.plan.topology_digest,
            key_type=key_type,
            value_type=value_type,
            context_tokens=context_tokens or self.plan.max_context_tokens,
            rope_identity=_digest(json.dumps(rope, sort_keys=True, default=str)),
            namespace=namespace,
        )

    def create_handle(
        self,
        request_id: str,
        *,
        sequence_id: str | None = None,
        key_type: str = "f16",
        value_type: str = "f16",
    ) -> CacheHandle:
        started = time.perf_counter()
        with self._lock:
            self._ensure_open()
            backend = self.config.backend
            if backend == "block":
                self._validate_block_backend(key_type, value_type)
                entries = tuple(BlockKVCache(self.block_pool) for _ in self.plan.layer_specs)
            else:
                entries = configure_kv_caches(
                    tuple(self.bundle.model.make_cache()), key_type, value_type
                )
            handle_id = f"cache-{uuid.uuid4().hex}"
            handle = CacheHandle(
                handle_id,
                request_id,
                sequence_id or request_id,
                self.plan,
                CacheBundle(entries),
                backend,
            )
            handle.metadata.update({"key_type": key_type, "value_type": value_type})
            elapsed = (time.perf_counter() - started) * 1000
            handle.metadata["allocation_time_ms"] = elapsed
            self._allocation_time_ms += elapsed
            self._handles[handle_id] = handle
            return handle

    def estimate_growth(
        self,
        handle: CacheHandle,
        *,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> int:
        self._validate_handle(handle)
        uncached = max(prompt_tokens - handle.prefix_tokens_reused, 0)
        tokens = uncached + max(max_new_tokens, 0)
        key_type = str(handle.metadata.get("key_type", "f16"))
        value_type = str(handle.metadata.get("value_type", "f16"))
        key_bits = _cache_bits(key_type)
        value_bits = _cache_bits(value_type)
        if handle.backend == "block":
            width = self.config.block_size_tokens
            tokens = ((tokens + width - 1) // width) * width
        elif tokens:
            step = 256
            tokens = ((tokens + step - 1) // step) * step
        return self.plan.bytes_per_token(key_bits=key_bits, value_bits=value_bits) * tokens

    def reserve(
        self,
        handle: CacheHandle,
        *,
        tokens: int,
        requested_bytes: int | None = None,
    ) -> CacheReservation:
        with self._lock:
            self._validate_handle(handle)
            if (
                handle.reservation is not None
                and handle.reservation.state is not ReservationState.RELEASED
            ):
                raise CacheReservationError("cache handle already owns a reservation")
            amount = requested_bytes
            if amount is None:
                amount = self.estimate_growth(handle, prompt_tokens=tokens, max_new_tokens=0)
            if amount < 0:
                raise ValueError("reserved cache bytes cannot be negative")
            available = self.available_bytes()
            if available is not None and amount > available:
                self._admission_failures += 1
                budget = self.config.memory_budget_bytes or 0
                if amount > budget:
                    raise CacheCapacityError(
                        "request exceeds the configured cache budget",
                        retryable=False,
                    )
                raise CacheCapacityError("request does not fit the available cache budget")
            reservation = CacheReservation(
                f"reservation-{uuid.uuid4().hex}", handle.handle_id, amount, tokens
            )
            reservation.state = ReservationState.GRANTED
            handle.reservation = reservation
            handle.reserved_bytes = amount
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def commit(self, reservation: CacheReservation, allocated_bytes: int = 0) -> None:
        with self._lock:
            current = self._require_reservation(reservation)
            if current.state is not ReservationState.GRANTED:
                raise CacheReservationError("only a granted reservation may be committed")
            if allocated_bytes < 0 or allocated_bytes > current.requested_bytes:
                raise CacheReservationError("committed bytes exceed the reservation")
            current.committed_bytes = allocated_bytes
            current.state = ReservationState.COMMITTED
            handle = self._handles[current.handle_id]
            handle.reserved_bytes = max(current.requested_bytes - allocated_bytes, 0)

    def release_reservation(self, reservation: CacheReservation) -> None:
        with self._lock:
            current = self._require_reservation(reservation)
            if current.state is ReservationState.RELEASED:
                raise CacheReservationError("cache reservation was released twice")
            current.state = ReservationState.RELEASED
            handle = self._handles.get(current.handle_id)
            if handle is not None:
                handle.reserved_bytes = 0
                handle.reservation = None
            self._reservations.pop(current.reservation_id, None)

    def adopt_prefix(
        self,
        handle: CacheHandle,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> int:
        started = time.perf_counter()
        try:
            with self._lock:
                self._validate_handle(handle)
                if not self.config.prefix_cache_enabled:
                    return 0
                if handle.backend == "dense":
                    if not self.plan.prefix_shareable:
                        raise PrefixReuseUnsupported(
                            "hybrid model prefix restoration is unqualified"
                        )
                    match = self.snapshot_prefix.longest_prefix(
                        (fingerprint,), tokens, minimum_tokens=minimum_tokens
                    )
                    if match is None:
                        return 0
                    matched_tokens, bundle = match
                    handle.bundle.release()
                    handle.bundle = bundle
                    reused = len(matched_tokens)
                else:
                    match = self.block_prefix.lookup(
                        fingerprint, tokens, minimum_tokens=minimum_tokens
                    )
                    if match is None:
                        return 0
                    if len(match.layer_references) != len(handle.bundle.entries):
                        for layer in match.layer_references:
                            for reference in layer:
                                reference.close()
                        raise CacheTopologyUnsupported(
                            "prefix layer count does not match cache plan"
                        )
                    for cache, references in zip(
                        handle.bundle.entries, match.layer_references, strict=True
                    ):
                        cache.adopt(references)
                    reused = match.token_count
                    handle.metadata["shared_blocks"] = match.block_count
                handle.prefix_tokens_reused = reused
                handle.computed_tokens = reused
                return reused
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            with self._lock:
                self._prefix_lookup_time_ms += elapsed
                if self._handles.get(handle.handle_id) is handle:
                    handle.metadata["prefix_lookup_time_ms"] = elapsed

    def preview_prefix(
        self,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> int:
        if not self.config.prefix_cache_enabled or not self.plan.prefix_shareable:
            return 0
        if self.config.backend == "block":
            return self.block_prefix.longest_prefix_tokens(
                fingerprint, tokens, minimum_tokens=minimum_tokens
            )
        return self.snapshot_prefix.longest_prefix_tokens(
            (fingerprint,), tokens, minimum_tokens=minimum_tokens
        )

    def publish_prefix(
        self,
        handle: CacheHandle,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        *,
        maximum_entries: int | None = None,
    ) -> bool:
        with self._lock:
            self._validate_handle(handle)
            if not self.config.prefix_cache_enabled or not tokens:
                return False
            if handle.backend == "dense":
                if not self.plan.prefix_shareable:
                    return False
                stored = self.snapshot_prefix.put((fingerprint,), tokens, handle.bundle)
                if stored and maximum_entries is not None:
                    self.snapshot_prefix.limit_entries(maximum_entries)
                return stored
            layers = tuple(cache.full_block_refs for cache in handle.bundle.entries)
            return self.block_prefix.publish(fingerprint, tokens, layers) > 0

    def release(self, handle: CacheHandle) -> None:
        with self._lock:
            self._validate_handle(handle)
            if handle.reservation is not None:
                self.release_reservation(handle.reservation)
            handle.bundle.release()
            handle.closed = True
            self._handles.pop(handle.handle_id, None)

    def available_bytes(self) -> int | None:
        if self.config.memory_budget_bytes is None:
            return None
        stats = self.stats()
        return max(
            self.config.memory_budget_bytes
            - stats.physical_bytes
            - stats.reserved_bytes,
            0,
        )

    def stats(self) -> CacheMemoryStats:
        with self._lock:
            handles = tuple(self._handles.values())
            logical = sum(item.bundle.nbytes for item in handles)
            allocated = sum(item.bundle.allocated_nbytes for item in handles)
            reserved = sum(item.reserved_bytes for item in handles)
            if self.config.backend == "block":
                blocks = self.block_pool.stats()
                prefix = self.block_prefix.stats()
                physical = blocks["physical_bytes"]
                pinned = blocks["pinned_bytes"]
            else:
                blocks = {
                    "block_total": 0,
                    "block_free": 0,
                    "block_live": 0,
                    "block_prefix": 0,
                    "block_shared": 0,
                }
                prefix = self.snapshot_prefix.stats()
                physical = allocated + prefix["bytes"]
                pinned = prefix["bytes"]
            return CacheMemoryStats(
                budget_bytes=self.config.memory_budget_bytes,
                logical_bytes=logical,
                allocated_bytes=allocated,
                physical_bytes=physical,
                prefix_bytes=prefix["bytes"],
                reserved_bytes=reserved,
                pinned_bytes=pinned,
                evictable_bytes=prefix["bytes"],
                active_handles=len(handles),
                block_total=blocks["block_total"],
                block_free=blocks["block_free"],
                block_live=blocks["block_live"],
                block_prefix=blocks["block_prefix"],
                block_shared=blocks["block_shared"],
                prefix_entries=prefix["entries"],
                prefix_hits=prefix["hits"],
                prefix_misses=prefix["misses"],
                prefix_evictions=prefix["evictions"],
                prefix_tokens_reused=prefix.get("tokens_reused", 0),
                admission_failures=self._admission_failures,
                prefix_lookup_time_ms=self._prefix_lookup_time_ms,
                allocation_time_ms=self._allocation_time_ms,
            )

    def clear_prefixes(self) -> None:
        with self._lock:
            self.snapshot_prefix.clear()
            self.block_prefix.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for handle in tuple(self._handles.values()):
                self.release(handle)
            self.clear_prefixes()
            self.block_pool.clear()
            self._closed = True

    def _validate_block_backend(self, key_type: str, value_type: str) -> None:
        if key_type != "f16" or value_type != "f16":
            raise KVQuantizationUnsupported("block KV currently supports f16 keys and values only")
        if self.plan.architecture != "qwen2" or not self.plan.block_compatible:
            raise CacheTopologyUnsupported(
                "block KV is currently qualified only for dense Qwen2/Qwen2.5 text models"
            )

    def _validate_handle(self, handle: CacheHandle) -> None:
        self._ensure_open()
        if handle.closed or self._handles.get(handle.handle_id) is not handle:
            raise CacheOwnershipError("cache handle is closed or owned by another manager")

    def _require_reservation(self, reservation: CacheReservation) -> CacheReservation:
        current = self._reservations.get(reservation.reservation_id)
        if current is not reservation:
            raise CacheReservationError("unknown cache reservation")
        return current

    def _ensure_open(self) -> None:
        if self._closed:
            raise CacheOwnershipError("cache manager is closed")


def _cache_bits(value: str) -> int:
    return {"q4_0": 4, "q8_0": 8}.get(value, 16)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _object_identity(value: Any) -> str:
    return _digest(f"{type(value).__module__}.{type(value).__qualname__}:{id(value)}")


def _model_identity(bundle: Any) -> str:
    root = Path(getattr(bundle, "path", "."))
    files = []
    if root.is_dir():
        for pattern in ("config.json", "*.safetensors", "*.safetensors.index.json"):
            for path in sorted(root.glob(pattern)):
                stat = path.stat()
                files.append((path.name, stat.st_size, stat.st_mtime_ns))
    payload = {
        "model": str(bundle.model_id),
        "revision": bundle.revision,
        "path": str(root.resolve()),
        "instance": id(bundle.model),
        "files": files,
        "config": repr(getattr(bundle.model, "config", None)),
    }
    return _digest(json.dumps(payload, sort_keys=True, default=str))
