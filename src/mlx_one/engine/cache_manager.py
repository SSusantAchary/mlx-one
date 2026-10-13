"""Central cache ownership, prefix reuse, accounting, and reservations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
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
        self._last_state_observations: list[dict[str, Any]] = []
        self._last_precision_configuration: dict[str, str] | None = None
        self.config = config or CacheConfig()
        self.weight_identity = _model_identity(bundle)
        self._model_identity = _digest(self.weight_identity + str(id(bundle.model)))
        self.plan = build_cache_plan(bundle, block_size_tokens=self.config.block_size_tokens)
        capacity = self.config.memory_budget_bytes or (2**63 - 1)
        from mlx_one.engine.arena import SegmentedPageArena

        pool_type = SegmentedPageArena if self.config.backend == "paged" else BlockPool
        self.block_pool = pool_type(
            capacity,
            block_tokens=self.config.block_size_tokens,
        )
        self.snapshot_prefix = PrefixCache(self.config.prefix_cache_budget_bytes)
        self.block_prefix = BlockPrefixIndex(
            self.config.prefix_cache_budget_bytes,
            block_tokens=self.config.block_size_tokens,
        )
        from mlx_one.engine.components import HybridPrefixIndex

        self.hybrid_prefix = HybridPrefixIndex(self.config.prefix_cache_budget_bytes,
                                              self.config.block_size_tokens)
        self._handles: dict[str, CacheHandle] = {}
        self._reservations: dict[str, CacheReservation] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._transient_bytes = 0
        self._owner_token = uuid.uuid4().hex
        self._admission_failures = 0
        self._prefix_lookup_time_ms = 0.0
        self._allocation_time_ms = 0.0
        self.harness_only = False
        self.pressure = None
        parameters = getattr(bundle.model, "parameters", None)
        from mlx_one.engine.cache import array_nbytes

        self.model_bytes = array_nbytes(parameters()) if callable(parameters) else 0

    def fingerprint(
        self,
        *,
        key_type: str = "f16",
        value_type: str = "f16",
        context_tokens: int | None = None,
        namespace: str | None = None,
        rendering_options: dict[str, Any] | None = None,
        multimodal_identity: str | None = None,
    ) -> CacheFingerprint:
        from mlx_one.engine.reporting import tensor_dtypes

        config = getattr(self.bundle.model, "config", None)
        rope = {
            name: getattr(getattr(config, "text_config", config), name, None)
            for name in ("rope_theta", "rope_scaling", "rope_freq_constant", "rope_max_length")
        }
        template = getattr(getattr(self.bundle, "chat_template", None), "template", None)
        return CacheFingerprint(
            model_id=str(self.bundle.model_id),
            revision=self.bundle.revision,
            model_identity=self._model_identity,
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
            namespace=namespace or f"private:{uuid.uuid4().hex}",
            special_tokens_identity=_digest(json.dumps({
                name: getattr(self.bundle.tokenizer, name, None)
                for name in ("bos_token_id", "eos_token_id", "pad_token_id", "special_tokens_map",
                             "added_tokens_decoder", "special_tokens")},
                sort_keys=True, default=str)),
            rendering_identity=_digest(json.dumps(
                rendering_options or {}, sort_keys=True, default=str)),
            multimodal_identity=multimodal_identity,
            execution_identity=_digest(json.dumps({
                "mode": getattr(self, "mode", None),
                "weight_dtypes": tensor_dtypes(self.bundle.model.parameters())
                    if callable(getattr(self.bundle.model, "parameters", None)) else (),
                "eos": getattr(self.bundle, "eos_token_ids", ()),
            }, sort_keys=True)),
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
            if backend == "paged":
                from mlx_one.engine.paged import PagedKVCache

                if key_type != value_type or key_type not in {"f16", "bf16", "q4_0", "q8_0"}:
                    raise KVQuantizationUnsupported("paged KV requires matched precision")
                bits = {"q4_0": 4, "q8_0": 8}.get(key_type)
                if bits is not None and not self.harness_only:
                    from mlx_one.engine.selection import qualification_matches

                    if not qualification_matches(self, "quantized_paged_kv"):
                        raise KVQuantizationUnsupported(
                            "quantized paged KV requires quality evidence")
                entries = tuple(PagedKVCache(self.block_pool, layer_id=spec.layer_id,
                                            dtype="bfloat16" if key_type == "bf16" else "float16",
                                            bits=bits) for spec in self.plan.layer_specs)
            elif backend == "block":
                self._validate_block_backend(key_type, value_type)
                entries = tuple(BlockKVCache(self.block_pool, layer_id=spec.layer_id)
                                for spec in self.plan.layer_specs)
            elif backend == "hybrid":
                from mlx_one.engine.components import hybrid_components

                if key_type not in {"f16", "bf16"} or key_type != value_type:
                    raise KVQuantizationUnsupported("hybrid batching requires floating KV")
                entries = hybrid_components(self.bundle, key_type)
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
            handle.metadata.update({"key_type": key_type, "value_type": value_type,
                                    "manager_id": self._owner_token})
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
        if handle.backend in {"block", "paged"}:
            width = self.config.block_size_tokens * (16 if handle.backend == "paged" else 1)
            tokens = ((tokens + width - 1) // width) * width
        elif tokens:
            step = 256
            tokens = ((tokens + step - 1) // step) * step
        amount = self.plan.bytes_per_token(key_bits=key_bits, value_bits=value_bits) * tokens
        if handle.backend == "hybrid":
            config = self.bundle.model.config
            amount += sum(1 for s in self.plan.layer_specs if s.kind.value == "conv_state") * (
                config.conv_L_cache - 1) * config.hidden_size * 4
        if handle.backend == "paged" and key_bits < 16:
            # FP16 scales+biases per 32 scalars plus floating tails and sealing overlap.
            amount = sum((s.num_kv_heads or 0) * tokens * sum(
                ((dim + 31) // 32) * (32 * key_bits // 8 + 4)
                for dim in (s.key_head_dim or 0, s.value_head_dim or s.key_head_dim or 0))
                for s in self.plan.layer_specs)
            amount += self.plan.bytes_per_token() * self.config.block_size_tokens * 16
        return amount

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
            if self.pressure is not None:
                self.pressure.shed(self)
                stats = self.stats()
                self.pressure.admission(amount,
                                        owner_physical=stats.physical_bytes + self.model_bytes,
                                        outstanding=stats.reserved_bytes)
            available = self.available_bytes()
            if available is not None and amount > available:
                self.clear_prefixes()
                self.block_pool.discard_free()
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
            reservation.manager_id = self._owner_token
            reservation.state = ReservationState.GRANTED
            handle.reservation = reservation
            handle.reserved_bytes = amount
            handle.metadata["reservation_base_bytes"] = handle.bundle.allocated_nbytes
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

    def reconcile(self, handle: CacheHandle) -> None:
        with self._lock:
            if handle.reservation is not None:
                current = handle.reservation
                base = int(handle.metadata.get("reservation_base_bytes", 0))
                allocated = max(handle.bundle.allocated_nbytes - base, 0)
                current.committed_bytes = min(allocated, current.requested_bytes)
                handle.reserved_bytes = max(current.requested_bytes - allocated, 0)

    @contextmanager
    def transient_reservation(self, amount: int):
        with self._lock:
            if self.pressure is not None:
                self.pressure.shed(self)
                stats = self.stats()
                self.pressure.forward(amount,
                                      owner_physical=stats.physical_bytes + self.model_bytes,
                                      outstanding=stats.reserved_bytes)
            available = self.available_bytes()
            if available is not None and amount > available:
                self.clear_prefixes()
                self.block_pool.discard_free()
                available = self.available_bytes()
            if available is not None and amount > available:
                raise CacheCapacityError("temporary dense batch buffers exceed cache budget",
                                         retryable=False)
            self._transient_bytes += amount
        try:
            yield
        finally:
            with self._lock:
                self._transient_bytes -= amount

    def release_reservation(self, reservation: CacheReservation) -> None:
        with self._lock:
            if reservation.manager_id != self._owner_token:
                raise CacheReservationError("reservation belongs to another manager")
            if reservation.state is ReservationState.RELEASED:
                return
            current = self._require_reservation(reservation)
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
                handle.metadata["fingerprint"] = fingerprint
                if fingerprint.multimodal_identity is not None:
                    raise PrefixReuseUnsupported("multimodal prefix restoration is unavailable")
                if not self.config.prefix_cache_enabled:
                    return 0
                if handle.backend == "hybrid":
                    match = self.hybrid_prefix.lookup(fingerprint, tokens)
                    if match is None or match[0] < minimum_tokens:
                        if match is not None:
                            match[1].release()
                        return 0
                    reused, replacement = match
                    old = handle.bundle
                    handle.bundle = replacement
                    old.release()
                elif handle.backend == "dense":
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
                    staged = []
                    try:
                        for cache, references in zip(handle.bundle.entries, match.layer_references,
                                                     strict=True):
                            draft = (cache.empty() if callable(getattr(cache, "empty", None))
                                     else BlockKVCache(self.block_pool, layer_id=cache.layer_id))
                            draft.adopt(references)
                            staged.append(draft)
                    except BaseException:
                        for layer in match.layer_references:
                            for reference in layer:
                                if not reference.closed:
                                    reference.close()
                        raise
                    old = handle.bundle
                    handle.bundle = CacheBundle(tuple(staged))
                    old.release()
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
        if not self.config.prefix_cache_enabled:
            return 0
        if self.config.backend == "hybrid":
            length = self.hybrid_prefix.preview(fingerprint, tokens)
            return length if length >= minimum_tokens else 0
        if not self.plan.prefix_shareable:
            return 0
        if self.config.backend in {"block", "paged"}:
            return self.block_prefix.longest_prefix_tokens(
                fingerprint, tokens, minimum_tokens=minimum_tokens
            )
        return self.snapshot_prefix.longest_prefix_tokens(
            (fingerprint,), tokens, minimum_tokens=minimum_tokens
        )

    def stage_prefix(self, handle, fingerprint, tokens):
        with self._lock:
            self._validate_handle(handle)
            if (handle.backend != "dense" or not self.plan.prefix_shareable
                    or not self.config.prefix_cache_enabled
                    or "staged_prefix" in handle.metadata
                    or any(c.offset != len(tokens) for c in handle.bundle.entries)):
                return False
            amount = handle.bundle.allocated_nbytes * 2
            if amount > self.config.prefix_cache_budget_bytes:
                handle.metadata["publication_rejections"] = 1
                return False
            resources = ExitStack()
            try:
                resources.enter_context(self.transient_reservation(amount))
                snapshot = handle.bundle.snapshot()
                resources.callback(snapshot.release)
                import mlx.core as mx

                tensors = []
                for entry in snapshot.entries:
                    for value in vars(entry).values():
                        if getattr(value, "dtype", None) is not None:
                            tensors.append(value)
                mx.eval(*tensors)
                handle.metadata["staged_prefix"] = (fingerprint, tokens, snapshot, resources)
                return True
            except BaseException:
                resources.close()
                raise

    def publish_staged_prefix(self, handle, *, maximum_entries=None):
        with self._lock:
            self._validate_handle(handle)
            staged = handle.metadata.pop("staged_prefix", None)
            if staged is None:
                return False
            fingerprint, tokens, snapshot, resources = staged
            try:
                callback = handle.metadata.get("is_cancelled")
                if (handle.metadata.get("cancelled") or callable(callback) and callback()
                        or any(c.offset != len(tokens) for c in snapshot.entries)):
                    return False
                stored = self.snapshot_prefix.put((fingerprint,), tokens, snapshot)
                if stored and maximum_entries is not None:
                    self.snapshot_prefix.limit_entries(maximum_entries)
                handle.metadata["prefix_publications"] = int(stored)
                return stored
            finally:
                resources.close()

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
            cancellation = handle.metadata.get("is_cancelled")
            if (handle.metadata.get("cancelled") or (callable(cancellation) and cancellation())
                    or fingerprint.multimodal_identity is not None
                    or any(int(cache.offset) != len(tokens) for cache in handle.bundle.entries)):
                return False
            handle.computed_tokens = len(tokens)
            if handle.backend == "hybrid":
                return self.publish_hybrid_boundary(handle, tokens)
            if handle.backend == "dense":
                if not self.plan.prefix_shareable:
                    return False
                stored = self.snapshot_prefix.put((fingerprint,), tokens, handle.bundle)
                if stored and maximum_entries is not None:
                    self.snapshot_prefix.limit_entries(maximum_entries)
                return stored
            layers = tuple(cache.full_block_refs for cache in handle.bundle.entries)
            return self.block_prefix.publish(fingerprint, tokens, layers) > 0

    def publish_hybrid_boundary(self, handle, prompt_ids):
        with self._lock:
            self._validate_handle(handle)
            callback = handle.metadata.get("is_cancelled")
            offset = handle.bundle.offset
            fp = handle.metadata.get("fingerprint")
            if (handle.backend != "hybrid" or not self.config.prefix_cache_enabled or fp is None
                    or fp.multimodal_identity is not None or not offset
                    or offset % self.config.block_size_tokens or offset > len(prompt_ids)
                    or (callable(callback) and callback()) or handle.metadata.get("cancelled")
                    or any(c.offset != offset for c in handle.bundle.entries)):
                return False
            return self.hybrid_prefix.put(fp, tuple(prompt_ids[:offset]), handle.bundle)

    def release(self, handle: CacheHandle) -> None:
        with self._lock:
            if handle.closed and handle.metadata.get("manager_id") == self._owner_token:
                return
            self._validate_handle(handle)
            staged = handle.metadata.pop("staged_prefix", None)
            if staged is not None:
                staged[3].close()
            if handle.reservation is not None:
                self.release_reservation(handle.reservation)
            from mlx_one.engine.reporting import describe_cache

            self._last_state_observations = describe_cache(handle.bundle.entries, self.plan)
            self._last_precision_configuration = {
                "key_type": str(handle.metadata["key_type"]),
                "value_type": str(handle.metadata["value_type"]),
            }
            handle.bundle.release()
            handle.closed = True
            self._handles.pop(handle.handle_id, None)

    def precision_configuration(self) -> dict[str, str] | None:
        with self._lock:
            configurations = {(str(h.metadata["key_type"]), str(h.metadata["value_type"]))
                              for h in self._handles.values()}
            if len(configurations) == 1:
                key, value = next(iter(configurations))
                return {"key_type": key, "value_type": value}
            if configurations:
                return {"key_type": "mixed", "value_type": "mixed"}
            return self._last_precision_configuration

    def state_observations(self) -> list[dict[str, Any]]:
        from mlx_one.engine.reporting import describe_cache

        with self._lock:
            live = [layer for handle in self._handles.values()
                    for layer in describe_cache(handle.bundle.entries, self.plan)]
            return live or list(self._last_state_observations)

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
            reserved = sum(item.reserved_bytes for item in handles) + self._transient_bytes
            if self.config.backend in {"block", "paged"}:
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
                prefix = (self.hybrid_prefix.stats() if self.config.backend == "hybrid"
                          else self.snapshot_prefix.stats())
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
                evictable_bytes=blocks.get("evictable_bytes", prefix["bytes"]),
                active_handles=len(handles),
                block_total=blocks["block_total"],
                block_free=blocks["block_free"],
                block_live=blocks["block_live"],
                block_prefix=blocks["block_prefix"],
                block_shared=blocks["block_shared"],
                retained_free_bytes=blocks.get("retained_free_bytes", 0),
                unique_live_physical_bytes=blocks.get("live_bytes", allocated),
                shared_page_savings_bytes=blocks.get("shared_page_savings_bytes", 0),
                prefix_metadata_bytes=prefix.get("metadata_bytes", 0),
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
            self.hybrid_prefix.clear()

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
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files.append((path.name, digest.hexdigest()))
    payload = {
        "model": str(bundle.model_id),
        "revision": bundle.revision,
        "path": str(root.resolve()),
        "files": files,
        "config": repr(getattr(bundle.model, "config", None)),
    }
    return _digest(json.dumps(payload, sort_keys=True, default=str))
