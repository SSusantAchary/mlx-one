"""Bounded, reference-counted cache block storage."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.cache_errors import CacheCapacityError, CacheCorruptionError


class BlockId(int):
    """Opaque physical block identifier."""


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

    def allocate(self, state: Any, token_count: int) -> BlockId:
        if token_count < 1 or token_count > self.block_tokens:
            raise ValueError("block token count exceeds the configured block width")
        signature = _signature(state)
        size = array_nbytes(state)
        with self._lock:
            free = self._free.get(signature)
            if free:
                block = free.popleft()
                if not free:
                    self._free.pop(signature, None)
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
            block_id = self._free_ids.popleft() if self._free_ids else BlockId(self._next_id)
            if int(block_id) == self._next_id:
                self._next_id += 1
            self._blocks[block_id] = CacheBlock(
                block_id,
                token_count,
                state,
                last_access=time.monotonic(),
                signature=signature,
            )
            return block_id

    def allocate_ref(self, state: Any, token_count: int) -> BlockRef:
        return BlockRef(self, self.allocate(state, token_count))

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
            if _signature(state) != block.signature:
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
            replacement = self.allocate(state, token_count)
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
        for block_id in self.block_ids:
            self.allocator.acquire(block_id)
        return PageTable(self.allocator, list(self.block_ids))

    def release(self) -> None:
        for block_id in self.block_ids:
            self.allocator.release(block_id)
        self.block_ids.clear()

    @property
    def token_count(self) -> int:
        return sum(self.allocator.get(block_id).token_count for block_id in self.block_ids)
