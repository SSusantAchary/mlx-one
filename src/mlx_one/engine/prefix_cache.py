"""Byte-bounded, process-local prefix cache with deterministic LRU eviction."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from mlx_one.engine.block_cache import BlockId, BlockPool, BlockRef
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_specs import CacheFingerprint


@dataclass(frozen=True)
class PrefixCacheKey:
    compatibility: tuple[object, ...]
    token_digest: str
    token_count: int

    @classmethod
    def create(
        cls, compatibility: tuple[object, ...], tokens: tuple[int, ...]
    ) -> PrefixCacheKey:
        encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
        return cls(compatibility, hashlib.sha256(encoded).hexdigest(), len(tokens))


@dataclass
class PrefixCacheEntry:
    key: PrefixCacheKey
    tokens: tuple[int, ...]
    cache: CacheBundle

    @property
    def nbytes(self) -> int:
        return self.cache.nbytes


class PrefixCache:
    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("prefix cache capacity cannot be negative")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[PrefixCacheKey, PrefixCacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(entry.nbytes for entry in self._entries.values())

    def put(
        self,
        compatibility: tuple[object, ...],
        tokens: tuple[int, ...],
        cache: CacheBundle,
    ) -> bool:
        if not tokens or self.capacity_bytes == 0:
            return False
        entry = PrefixCacheEntry(
            PrefixCacheKey.create(compatibility, tokens), tokens, cache.snapshot()
        )
        if entry.nbytes > self.capacity_bytes:
            return False
        with self._lock:
            previous = self._entries.pop(entry.key, None)
            if previous is not None:
                previous.cache.release()
            self._entries[entry.key] = entry
            self._evict()
        return True

    def longest_prefix(
        self,
        compatibility: tuple[object, ...],
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> tuple[tuple[int, ...], CacheBundle] | None:
        if minimum_tokens < 0:
            raise ValueError("minimum prefix tokens cannot be negative")
        with self._lock:
            matches = [
                entry
                for entry in self._entries.values()
                if entry.key.compatibility == compatibility
                and len(entry.tokens) >= minimum_tokens
                and len(entry.tokens) < len(tokens)
                and tokens[: len(entry.tokens)] == entry.tokens
            ]
            if not matches:
                self.misses += 1
                return None
            selected = max(matches, key=lambda item: len(item.tokens))
            self._entries.move_to_end(selected.key)
            self.hits += 1
            return selected.tokens, selected.cache.snapshot()

    def longest_prefix_tokens(
        self,
        compatibility: tuple[object, ...],
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> int:
        """Return a reusable token count without cloning state or updating metrics."""

        with self._lock:
            return max(
                (
                    len(entry.tokens)
                    for entry in self._entries.values()
                    if entry.key.compatibility == compatibility
                    and len(entry.tokens) >= minimum_tokens
                    and len(entry.tokens) < len(tokens)
                    and tokens[: len(entry.tokens)] == entry.tokens
                ),
                default=0,
            )

    def clear(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.cache.release()
            self._entries.clear()

    def limit_entries(self, maximum: int) -> None:
        if maximum < 1:
            raise ValueError("prefix cache entry limit must be positive")
        with self._lock:
            while len(self._entries) > maximum:
                _, entry = self._entries.popitem(last=False)
                entry.cache.release()
                self.evictions += 1

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.nbytes,
                "capacity_bytes": self.capacity_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def _evict(self) -> None:
        while self.nbytes > self.capacity_bytes and self._entries:
            _, entry = self._entries.popitem(last=False)
            entry.cache.release()
            self.evictions += 1


@dataclass(frozen=True)
class BlockLink:
    pool: BlockPool
    block_id: BlockId


@dataclass
class BlockPrefixEntry:
    key: str
    fingerprint: CacheFingerprint
    token_count: int
    layer_blocks: tuple[tuple[BlockLink, ...], ...]
    created_at: float
    last_access: float
    hit_count: int = 0

    @property
    def block_count(self) -> int:
        return sum(len(layer) for layer in self.layer_blocks)


@dataclass(frozen=True)
class BlockPrefixMatch:
    token_count: int
    layer_references: tuple[tuple[BlockRef, ...], ...]
    block_count: int


class BlockPrefixIndex:
    """Byte-bounded longest-prefix index over complete token blocks."""

    def __init__(self, capacity_bytes: int, *, block_tokens: int) -> None:
        if capacity_bytes < 0:
            raise ValueError("prefix cache capacity cannot be negative")
        self.capacity_bytes = capacity_bytes
        self.block_tokens = block_tokens
        self._entries: OrderedDict[str, BlockPrefixEntry] = OrderedDict()
        self._pin_owners: dict[tuple[int, int], int] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.tokens_reused = 0

    @property
    def nbytes(self) -> int:
        with self._lock:
            total = 0
            seen: set[tuple[int, int]] = set()
            for entry in self._entries.values():
                for layer in entry.layer_blocks:
                    for link in layer:
                        identity = (id(link.pool), int(link.block_id))
                        if identity not in seen:
                            total += link.pool.get(link.block_id).nbytes
                            seen.add(identity)
            return total

    def lookup(
        self,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> BlockPrefixMatch | None:
        maximum = ((len(tokens) - 1) // self.block_tokens) * self.block_tokens
        if maximum < max(minimum_tokens, self.block_tokens):
            with self._lock:
                self.misses += 1
            return None
        selected: BlockPrefixEntry | None = None
        parent = fingerprint.digest
        with self._lock:
            for start in range(0, maximum, self.block_tokens):
                parent = _block_hash(parent, tokens[start : start + self.block_tokens])
                candidate = self._entries.get(parent)
                if candidate is None:
                    break
                if candidate.fingerprint == fingerprint and candidate.token_count >= minimum_tokens:
                    selected = candidate
            if selected is None:
                self.misses += 1
                return None
            acquired: list[BlockRef] = []
            try:
                layers = []
                for layer in selected.layer_blocks:
                    refs = tuple(
                        BlockRef(link.pool, link.block_id, acquired=True) for link in layer
                    )
                    acquired.extend(refs)
                    layers.append(refs)
            except Exception:
                for ref in reversed(acquired):
                    ref.close()
                raise
            selected.hit_count += 1
            selected.last_access = time.monotonic()
            self._entries.move_to_end(selected.key)
            self.hits += 1
            self.tokens_reused += selected.token_count
            return BlockPrefixMatch(
                selected.token_count,
                tuple(layers),
                selected.block_count,
            )

    def longest_prefix_tokens(
        self,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        *,
        minimum_tokens: int = 1,
    ) -> int:
        maximum = ((len(tokens) - 1) // self.block_tokens) * self.block_tokens
        parent = fingerprint.digest
        selected = 0
        with self._lock:
            for start in range(0, maximum, self.block_tokens):
                parent = _block_hash(parent, tokens[start : start + self.block_tokens])
                candidate = self._entries.get(parent)
                if candidate is None:
                    break
                if candidate.fingerprint == fingerprint and candidate.token_count >= minimum_tokens:
                    selected = candidate.token_count
        return selected

    def publish(
        self,
        fingerprint: CacheFingerprint,
        tokens: tuple[int, ...],
        layer_blocks: tuple[tuple[BlockRef, ...], ...],
    ) -> int:
        if not layer_blocks or self.capacity_bytes == 0:
            return 0
        complete = min(len(layer) for layer in layer_blocks)
        complete = min(complete, len(tokens) // self.block_tokens)
        if complete < 1:
            return 0
        parent = fingerprint.digest
        published = 0
        with self._lock:
            for index in range(complete):
                start = index * self.block_tokens
                parent = _block_hash(parent, tokens[start : start + self.block_tokens])
                if parent in self._entries:
                    self._entries[parent].last_access = time.monotonic()
                    self._entries.move_to_end(parent)
                    continue
                links = tuple(
                    tuple(BlockLink(ref.pool, ref.block_id) for ref in layer[: index + 1])
                    for layer in layer_blocks
                )
                for layer in links:
                    for link in layer:
                        self._retain_pin(link)
                now = time.monotonic()
                self._entries[parent] = BlockPrefixEntry(
                    parent,
                    fingerprint,
                    (index + 1) * self.block_tokens,
                    links,
                    now,
                    now,
                )
                published += 1
            self._evict()
        return published

    def clear(self) -> None:
        with self._lock:
            for entry in tuple(self._entries.values()):
                self._release_entry(entry)
            self._entries.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self.nbytes,
                "capacity_bytes": self.capacity_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "tokens_reused": self.tokens_reused,
            }

    def _retain_pin(self, link: BlockLink) -> None:
        identity = (id(link.pool), int(link.block_id))
        owners = self._pin_owners.get(identity, 0)
        if owners == 0:
            link.pool.pin(link.block_id)
        self._pin_owners[identity] = owners + 1

    def _release_pin(self, link: BlockLink) -> None:
        identity = (id(link.pool), int(link.block_id))
        owners = self._pin_owners[identity] - 1
        if owners == 0:
            del self._pin_owners[identity]
            link.pool.unpin(link.block_id)
        else:
            self._pin_owners[identity] = owners

    def _release_entry(self, entry: BlockPrefixEntry) -> None:
        for layer in entry.layer_blocks:
            for link in layer:
                self._release_pin(link)

    def _evict(self) -> None:
        while self.nbytes > self.capacity_bytes and self._entries:
            _, entry = self._entries.popitem(last=False)
            self._release_entry(entry)
            self.evictions += 1


def _block_hash(parent: str, tokens: tuple[int, ...]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(parent.encode("ascii") + b":" + encoded).hexdigest()
