"""Bounded, reference-counted cache block storage."""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.cache_errors import CacheCapacityError, CacheCorruptionError


class BlockId(int):
    """Opaque epoch/slot/generation identity; integers are never recycled."""

    @property
    def epoch(self) -> int:
        return int(self) >> 96

    @property
    def slot(self) -> int:
        return (int(self) >> 32) & ((1 << 64) - 1)

    @property
    def generation(self) -> int:
        return int(self) & ((1 << 32) - 1)


@dataclass(frozen=True)
class BlockLayout:
    version: int = 1
    layer_id: int = 0
    page_tokens: int = 32
    num_kv_heads: int | None = None
    head_dim: int | None = None
    key_dtype: str = "float16"
    value_dtype: str = "float16"
    quantization_bits: int | None = None
    quantization_group_size: int = 32
    kernel_version: str | None = None


def _signature(value: Any) -> tuple[object, ...]:
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, *(_signature(item) for item in value))
    return (
        type(value).__name__,
        tuple(getattr(value, "shape", ())),
        str(getattr(value, "dtype", "")),
    )


def _copy_state(target: Any, source: Any) -> Any:
    """Copy into retained tensor storage, falling back to the supplied object."""

    if isinstance(target, tuple) and isinstance(source, tuple):
        return tuple(_copy_state(left, right) for left, right in zip(target, source, strict=True))
    if isinstance(target, list) and isinstance(source, list):
        return [_copy_state(left, right) for left, right in zip(target, source, strict=True)]
    try:
        target[...] = source
    except (TypeError, ValueError, AttributeError):
        return source
    return target


@dataclass
class CacheBlock:
    block_id: BlockId
    token_count: int
    state: Any
    references: int = 1
    pins: int = 0
    immutable: bool = False
    last_access: float = 0.0
    signature: tuple[object, ...] = ()
    layout: BlockLayout | None = None

    @property
    def nbytes(self) -> int:
        return array_nbytes(self.state)


class BlockRef:
    """One deterministic owner of a physical block."""

    def __init__(self, pool: BlockPool, block_id: BlockId, *, acquired: bool = False) -> None:
        self.pool = pool
        self.block_id = block_id
        self.closed = False
        if acquired:
            self.pool.acquire(block_id)

    @property
    def block(self) -> CacheBlock:
        if self.closed:
            raise CacheCorruptionError("block reference is closed")
        return self.pool.get(self.block_id)

    def fork(self) -> BlockRef:
        if self.closed:
            raise CacheCorruptionError("cannot fork a closed block reference")
        return BlockRef(self.pool, self.block_id, acquired=True)

    def pin(self) -> None:
        self.pool.pin(self.block_id)

    def unpin(self) -> None:
        self.pool.unpin(self.block_id)

    def close(self) -> None:
        if self.closed:
            raise CacheCorruptionError("block reference was released twice")
        self.closed = True
        self.pool.release(self.block_id)


class BlockPool:
    """Own physical tensor blocks independently of request page tables."""

    def __init__(
        self,
        capacity_bytes: int,
        *,
        block_tokens: int = 32,
        retain_free: bool = True,
    ) -> None:
        if capacity_bytes < 1 or block_tokens < 1:
            raise ValueError("block capacity and token width must be valid")
        self.capacity_bytes = capacity_bytes
        self.block_tokens = block_tokens
        self.retain_free = retain_free
        self._blocks: dict[BlockId, CacheBlock] = {}
        self._free: dict[tuple[object, ...], deque[CacheBlock]] = defaultdict(deque)
        self._free_ids: deque[BlockId] = deque()
        self._next_id = 0
        self._epoch = uuid.uuid4().int
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(block.nbytes for block in self._blocks.values())

    @property
    def allocated_nbytes(self) -> int:
        with self._lock:
            return self.nbytes + sum(
                block.nbytes for blocks in self._free.values() for block in blocks
            )

    def allocate(
        self, state: Any, token_count: int, *, layout: BlockLayout | None = None
    ) -> BlockId:
        if token_count < 1 or token_count > self.block_tokens:
            raise ValueError("block token count exceeds the configured block width")
        signature = (layout, _signature(state))
        size = array_nbytes(state)
        with self._lock:
            free = self._free.get(signature)
            if free:
                block = free.popleft()
                if not free:
                    self._free.pop(signature, None)
                block.block_id = self._identity(block.block_id.slot)
                block.token_count = token_count
                block.references = 1
                block.pins = 0
                block.immutable = False
                block.last_access = time.monotonic()
                block.state = _copy_state(block.state, state)
                self._blocks[block.block_id] = block
                return block.block_id
            self._discard_free_for(size)
            if self.allocated_nbytes + size > self.capacity_bytes:
                raise CacheCapacityError("KV block pool is full")
            slot = self._free_ids.popleft().slot if self._free_ids else self._next_id
            if slot == self._next_id:
                self._next_id += 1
            block_id = self._identity(slot)
            self._blocks[block_id] = CacheBlock(
                block_id,
                token_count,
                state,
                last_access=time.monotonic(),
                signature=signature,
                layout=layout,
            )
            return block_id

    def allocate_ref(self, state: Any, token_count: int, *, layout=None) -> BlockRef:
        return BlockRef(self, self.allocate(state, token_count, layout=layout))

    def acquire(self, block_id: int) -> None:
        with self._lock:
            block = self._require(block_id)
            block.references += 1
            block.last_access = time.monotonic()

    def release(self, block_id: int) -> None:
        with self._lock:
            block = self._require(block_id)
            if block.references < 1:
                raise CacheCorruptionError("KV block reference count became negative")
            block.references -= 1
            if block.references == 0 and block.pins == 0:
                self._reclaim(block)

    def pin(self, block_id: int) -> None:
        with self._lock:
            block = self._require(block_id)
            block.pins += 1
            block.last_access = time.monotonic()

    def unpin(self, block_id: int) -> None:
        with self._lock:
            block = self._require(block_id)
            if block.pins < 1:
                raise CacheCorruptionError("KV block pin count became negative")
            block.pins -= 1
            if block.pins == 0 and block.references == 0:
                self._reclaim(block)

    def seal(self, block_id: int) -> None:
        with self._lock:
            block = self._require(block_id)
            if block.token_count != self.block_tokens:
                raise CacheCorruptionError("only complete KV blocks may be sealed")
            block.immutable = True

    def get(self, block_id: int) -> CacheBlock:
        with self._lock:
            block = self._require(block_id)
            block.last_access = time.monotonic()
            return block

    def update(self, block_id: int, state: Any, token_count: int) -> None:
        if token_count < 1 or token_count > self.block_tokens:
            raise ValueError("block token count exceeds the configured block width")
        with self._lock:
            block = self._require(block_id)
            if block.immutable:
                raise CacheCorruptionError("cannot mutate an immutable KV block")
            if (block.layout, _signature(state)) != block.signature:
                raise CacheCorruptionError("KV block storage signature changed")
            old_size = block.nbytes
            new_size = array_nbytes(state)
            if self.allocated_nbytes - old_size + new_size > self.capacity_bytes:
                raise CacheCapacityError("KV block pool is full")
            block.state = state
            block.token_count = token_count
            block.last_access = time.monotonic()

    def copy_on_write(self, block_id: int, state: Any, token_count: int) -> BlockId:
        with self._lock:
            block = self._require(block_id)
            if block.references == 1 and block.pins == 0 and not block.immutable:
                self.update(block_id, state, token_count)
                return BlockId(block_id)
            replacement = self.allocate(state, token_count, layout=block.layout)
            self.release(block_id)
            return replacement

    def stats(self) -> dict[str, int]:
        with self._lock:
            free = sum(len(items) for items in self._free.values())
            return {
                "block_total": len(self._blocks) + free,
                "block_free": free,
                "block_live": sum(block.references > 0 for block in self._blocks.values()),
                "block_prefix": sum(block.pins > 0 for block in self._blocks.values()),
                "block_shared": sum(block.references > 1 for block in self._blocks.values()),
                "block_pinned": sum(block.pins > 0 for block in self._blocks.values()),
                "physical_bytes": self.allocated_nbytes,
                "live_bytes": self.nbytes,
                "retained_free_bytes": self.allocated_nbytes - self.nbytes,
                "evictable_bytes": sum(block.nbytes for block in self._blocks.values()
                                       if block.references == 0 and block.pins == 1),
                "shared_page_savings_bytes": sum(
                    block.nbytes * max(block.references - 1, 0) for block in self._blocks.values()
                ),
                "pinned_bytes": sum(
                    block.nbytes for block in self._blocks.values() if block.pins
                ),
            }

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()
            self._free.clear()
            self._free_ids.clear()
            self._next_id = 0
            self._epoch = uuid.uuid4().int

    def _identity(self, slot: int) -> BlockId:
        self._generation += 1
        return BlockId((self._epoch << 96) | (slot << 32) | self._generation)

    def discard_free(self) -> None:
        with self._lock:
            self._free.clear()

    def _require(self, block_id: int) -> CacheBlock:
        try:
            return self._blocks[BlockId(block_id)]
        except KeyError as exc:
            raise CacheCorruptionError(f"unknown or freed KV block {block_id}") from exc

    def _reclaim(self, block: CacheBlock) -> None:
        del self._blocks[block.block_id]
        if self.retain_free:
            self._free[block.signature].append(block)
        else:
            self._free_ids.append(block.block_id)

    def _discard_free_for(self, requested_bytes: int) -> None:
        while self.allocated_nbytes + requested_bytes > self.capacity_bytes and self._free:
            signature = next(iter(self._free))
            block = self._free[signature].popleft()
            self._free_ids.append(block.block_id)
            if not self._free[signature]:
                del self._free[signature]


class BlockAllocator(BlockPool):
    """Compatibility name for the original non-retaining allocator."""

    def __init__(self, capacity_bytes: int, *, block_tokens: int = 256) -> None:
        super().__init__(capacity_bytes, block_tokens=block_tokens, retain_free=False)


@dataclass
class PageTable:
    allocator: BlockPool
    block_ids: list[int]

    def clone(self) -> PageTable:
        acquired = []
        try:
            for block_id in self.block_ids:
                self.allocator.acquire(block_id)
                acquired.append(block_id)
            return PageTable(self.allocator, list(self.block_ids))
        except BaseException:
            for block_id in acquired:
                self.allocator.release(block_id)
            raise

    def release(self) -> None:
        for block_id in self.block_ids:
            self.allocator.release(block_id)
        self.block_ids.clear()

    @property
    def token_count(self) -> int:
        return sum(self.allocator.get(block_id).token_count for block_id in self.block_ids)
