"""Versioned, bounded segments and persistent page-table chunks for direct KV."""
from __future__ import annotations

import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field

from mlx_one.engine.block_cache import BlockPool, CacheBlock, _signature
from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.cache_errors import CacheCapacityError, CacheCorruptionError

ARENA_VERSION = "segmented-arena-v1"
SEGMENT_PAGES = 16


@dataclass
class ArenaSegment:
    segment_id: int
    layout: object
    state: tuple
    signature: tuple
    occupied: set = field(default_factory=set)
    version: int = 0


class SegmentedPageArena(BlockPool):
    """Own tensors once per 16-page segment; BlockRef retains page ownership.

    Tensor replacement is evaluated before publication. Page slices are adapters,
    never additional owners in physical accounting. Execution leases retain old
    segment versions until their evaluated dispatch completes.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments = {}
        self.locations = {}
        self._next_segment = 0
        self.table_update_bytes = 0
        self._tables = weakref.WeakSet()
        self.replacement_peak_bytes = 0
        self.segment_update_bytes = 0
        self._leases = []

    @property
    def table_bytes(self):
        tensors = {id(x): x for table in self._tables for x in table.chunks.values()}
        return sum(array_nbytes(x) for x in tensors.values())

    @property
    def allocated_nbytes(self):
        return sum(array_nbytes(s.state) for s in self.segments.values()) + self.table_bytes

    @property
    def nbytes(self):
        return sum(array_nbytes(s.state) for s in self.segments.values() if s.occupied)

    def allocate(self, state, token_count, *, layout=None):
        import mlx.core as mx

        if layout is None or not 0 < token_count <= self.block_tokens:
            raise ValueError("arena allocation requires a valid layout and token count")
        signature = (layout, _signature(state))
        with self._lock:
            segment = next((s for s in self.segments.values()
                            if s.signature == signature and len(s.occupied) < SEGMENT_PAGES), None)
            fresh = segment is None
            if fresh:
                size = array_nbytes(state) * SEGMENT_PAGES
                self.discard_free()
                if self.allocated_nbytes + size > self.capacity_bytes:
                    raise CacheCapacityError("segmented KV arena is full")
                segment = ArenaSegment(self._next_segment, layout, tuple(
                    mx.zeros((SEGMENT_PAGES, *x.shape[1:]), dtype=x.dtype) for x in state),
                    signature)
            slot = next(i for i in range(SEGMENT_PAGES) if i not in segment.occupied)
            staged = self._stage(segment, slot, state)
            block_id = self._identity(segment.segment_id * SEGMENT_PAGES + slot)
            if fresh:
                self._next_segment += 1
                self.segments[segment.segment_id] = segment
            segment.occupied.add(slot)
            self.locations[block_id] = (segment.segment_id, slot)
            self._blocks[block_id] = CacheBlock(block_id, token_count, (),
                                                last_access=time.monotonic(),
                                                signature=signature, layout=layout)
            self._publish(segment, staged)
            return block_id

    def _stage(self, segment, slot, values):
        import mlx.core as mx

        self.replacement_peak_bytes = max(self.replacement_peak_bytes, array_nbytes(segment.state))
        staged = tuple(mx.zeros(x.shape, dtype=x.dtype) for x in segment.state)
        for target, old, value in zip(staged, segment.state, values, strict=True):
            target[...] = old
            target[slot:slot + 1] = value
        mx.eval(*staged)
        return staged

    def _publish(self, segment, staged):
        segment.state, segment.version = staged, segment.version + 1
        self.segment_update_bytes += array_nbytes(staged)
        for identity, (sid, slot) in self.locations.items():
            if sid == segment.segment_id:
                self._blocks[identity].state = tuple(x[slot:slot + 1] for x in staged)

    def store(self, block_id):
        with self._lock:
            block = self._require(block_id)
            sid, slot = self.locations[block_id]
            segment = self.segments[sid]
            self._publish(segment, self._stage(segment, slot, block.state))

    def address(self, reference):
        block = reference.block
        if reference.pool is not self or block.block_id not in self.locations:
            raise CacheCorruptionError("page is not owned by this arena")
        sid, slot = self.locations[block.block_id]
        segment = self.segments[sid]
        if segment.layout != block.layout or slot not in segment.occupied:
            raise CacheCorruptionError("stale arena address or layout")
        return sid, slot

    @contextmanager
    def lease(self):
        with self._lock:
            versions = tuple(s.state for s in self.segments.values() if s.occupied)
            self._leases.append(versions)
        try:
            yield versions
        finally:
            with self._lock:
                self._leases[:] = [item for item in self._leases if item is not versions]

    def _reclaim(self, block):
        del self._blocks[block.block_id]
        sid, slot = self.locations.pop(block.block_id)
        self.segments[sid].occupied.remove(slot)
        if not self.retain_free and not self.segments[sid].occupied:
            del self.segments[sid]

    def discard_free(self):
        with self._lock:
            for sid in [sid for sid, s in self.segments.items() if not s.occupied]:
                del self.segments[sid]

    def clear(self):
        with self._lock:
            super().clear()
            self.segments.clear()
            self.locations.clear()
            self._tables.clear()

    def stats(self):
        result = super().stats()
        payload = sum(b.nbytes for b in self._blocks.values())
        current = {id(x) for s in self.segments.values() for x in s.state}
        old = {id(x): x for lease in self._leases for state in lease for x in state
               if id(x) not in current}
        result.update(arena_version=ARENA_VERSION, segment_count=len(self.segments),
                      segment_capacity_pages=SEGMENT_PAGES,
                      page_payload_bytes=payload,
                      unused_segment_bytes=self.allocated_nbytes - self.table_bytes - payload,
                      table_bytes=self.table_bytes, table_update_bytes=self.table_update_bytes,
                      segment_update_bytes=self.segment_update_bytes,
                      replacement_overlap_peak_bytes=self.replacement_peak_bytes,
                      replacement_overlap_bytes=sum(array_nbytes(x) for x in old.values()))
        return result


class PersistentPageTable:
    """Immutable GPU chunks: clone shares them; editing copies only one chunk."""
    def __init__(self, arena):
        self.arena, self.chunks, self.entries = arena, {}, {}
        arena._tables.add(self)

    def fork(self):
        result = PersistentPageTable(self.arena)
        result.chunks, result.entries = dict(self.chunks), dict(self.entries)
        # Shared tensors are counted once by their owning original table. Logical
        # descriptors on a fork do not allocate GPU storage until edited.
        return result

    def update(self, index, reference):
        import mlx.core as mx

        sid, slot = self.arena.address(reference)
        block = reference.block
        entry = (sid, slot, index * self.arena.block_tokens, block.token_count)
        if self.entries.get(index) == entry:
            return
        staged = {**self.entries, index: entry}
        chunk = index // SEGMENT_PAGES
        value = mx.array([staged.get(i, (-1, -1, 0, 0)) for i in range(
            chunk * SEGMENT_PAGES, (chunk + 1) * SEGMENT_PAGES)], dtype=mx.int32)
        mx.eval(value)
        self.entries = staged
        self.chunks[chunk] = value
        self.arena.table_update_bytes += SEGMENT_PAGES * 4 * 4

    def truncate(self, length):
        import mlx.core as mx

        for index in sorted([i for i in self.entries if i >= length], reverse=True):
            del self.entries[index]
        for chunk in list(self.chunks):
            if chunk * SEGMENT_PAGES >= length:
                del self.chunks[chunk]
        if length % SEGMENT_PAGES:
            chunk = length // SEGMENT_PAGES
            value = mx.array([self.entries.get(i, (-1, -1, 0, 0)) for i in range(
                chunk * SEGMENT_PAGES, (chunk + 1) * SEGMENT_PAGES)], dtype=mx.int32)
            mx.eval(value)
            self.chunks[chunk] = value
            self.arena.table_update_bytes += SEGMENT_PAGES * 4 * 4

    def validate(self, references):
        if len(references) != len(self.entries):
            raise CacheCorruptionError("page table length mismatch")
        for index, ref in enumerate(references):
            sid, slot = self.arena.address(ref)
            if self.entries[index] != (sid, slot, index * self.arena.block_tokens,
                                       ref.block.token_count):
                raise CacheCorruptionError("page table address mismatch")
