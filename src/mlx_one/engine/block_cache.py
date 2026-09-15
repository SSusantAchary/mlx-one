"""Reference-counted token-block allocator for block-KV feasibility work."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from mlx_one.engine.cache import array_nbytes


@dataclass
class CacheBlock:
    block_id: int
    token_count: int
    state: Any
    references: int = 1

    @property
    def nbytes(self) -> int:
        return array_nbytes(self.state)


class BlockAllocator:
    """Own physical blocks independently from per-sequence page tables."""

    def __init__(self, capacity_bytes: int, *, block_tokens: int = 256) -> None:
        if capacity_bytes < 1 or block_tokens < 1:
            raise ValueError("block capacity and token width must be positive")
        self.capacity_bytes = capacity_bytes
        self.block_tokens = block_tokens
        self._blocks: dict[int, CacheBlock] = {}
        self._free_ids: deque[int] = deque()
        self._next_id = 0
        self._lock = threading.RLock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(block.nbytes for block in self._blocks.values())

    def allocate(self, state: Any, token_count: int) -> int:
        if token_count < 1 or token_count > self.block_tokens:
            raise ValueError("block token count exceeds the configured block width")
        size = array_nbytes(state)
        with self._lock:
            if self.nbytes + size > self.capacity_bytes:
                raise MemoryError("KV block pool is full")
            block_id = self._free_ids.popleft() if self._free_ids else self._next_id
            if block_id == self._next_id:
                self._next_id += 1
            self._blocks[block_id] = CacheBlock(block_id, token_count, state)
            return block_id

    def acquire(self, block_id: int) -> None:
        with self._lock:
            self._blocks[block_id].references += 1

    def release(self, block_id: int) -> None:
        with self._lock:
            block = self._blocks[block_id]
            block.references -= 1
            if block.references < 0:
                raise RuntimeError("KV block reference count became negative")
            if block.references == 0:
                del self._blocks[block_id]
                self._free_ids.append(block_id)

    def get(self, block_id: int) -> CacheBlock:
        with self._lock:
            return self._blocks[block_id]

    def copy_on_write(self, block_id: int, state: Any, token_count: int) -> int:
        with self._lock:
            block = self._blocks[block_id]
            if block.references == 1:
                old_size = block.nbytes
                new_size = array_nbytes(state)
                if self.nbytes - old_size + new_size > self.capacity_bytes:
                    raise MemoryError("KV block pool is full")
                block.state = state
                block.token_count = token_count
                return block_id
            self.release(block_id)
            return self.allocate(state, token_count)

    def clear(self) -> None:
        with self._lock:
            self._free_ids.extend(self._blocks)
            self._blocks.clear()


@dataclass
class PageTable:
    allocator: BlockAllocator
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
