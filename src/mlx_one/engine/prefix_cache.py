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
            entry.cache.release()
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
            matches = [(self._match(entry, compatibility, tokens), entry)
                       for entry in self._entries.values()]
            count, selected = max(matches, key=lambda pair: pair[0], default=(0, None))
            if count < max(minimum_tokens, 1) or selected is None:
                self.misses += 1
                return None
            restored = selected.cache.snapshot()
            try:
                if not restored.trim(len(selected.tokens) - count):
                    raise ValueError("prefix state cannot be trimmed")
            except BaseException:
                restored.release()
                raise
            self._entries.move_to_end(selected.key)
            self.hits += 1
            return tokens[:count], restored

    @staticmethod
    def _match(entry, compatibility, tokens) -> int:
        if entry.key.compatibility != compatibility:
            return 0
        count = 0
        for left, right in zip(entry.tokens, tokens[:max(len(tokens) - 1, 0)], strict=False):
            if left != right:
                break
            count += 1
        return count

    def longest_prefix_tokens(self, compatibility, tokens, *, minimum_tokens=1) -> int:
        with self._lock:
            count = max((self._match(entry, compatibility, tokens)
                         for entry in self._entries.values()), default=0)
            return count if count >= minimum_tokens else 0

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
    parent: str
    fingerprint: CacheFingerprint
    token_count: int
    tokens: tuple[int, ...]
    layer_blocks: tuple[tuple[BlockLink, ...], ...]
    last_access: float

    @property
    def metadata_bytes(self) -> int:
        # Conservative bounded accounting, independent of retained tensor bytes.
        return 512 + len(self.tokens) * 40 + len(self.layer_blocks) * 128


@dataclass(frozen=True)
class BlockPrefixMatch:
    token_count: int
    layer_references: tuple[tuple[BlockRef, ...], ...]
    block_count: int


class BlockPrefixIndex:
    """Parent-linked APC: each node pins one immutable per-layer page group."""

    def __init__(self, capacity_bytes: int, *, block_tokens: int,
                 metadata_capacity_bytes: int = 4 * 1024**2) -> None:
        if capacity_bytes < 0 or metadata_capacity_bytes < 0:
            raise ValueError("prefix capacity cannot be negative")
        self.capacity_bytes = capacity_bytes
        self.metadata_capacity_bytes = metadata_capacity_bytes
        self.block_tokens = block_tokens
        self._entries: OrderedDict[str, BlockPrefixEntry] = OrderedDict()
        self._pin_owners: dict[tuple[int, int], int] = {}
        self._lock = threading.RLock()
        self.hits = self.misses = self.evictions = self.tokens_reused = 0

    @property
    def nbytes(self) -> int:
        with self._lock:
            seen = {}
            for entry in self._entries.values():
                for layer in entry.layer_blocks:
                    for link in layer:
                        identity = (id(link.pool), int(link.block_id))
                        seen[identity] = link.pool.get(link.block_id).nbytes
            return sum(seen.values())

    def _path(self, fingerprint, tokens):
        maximum = ((len(tokens) - 1) // self.block_tokens) * self.block_tokens
        parent = fingerprint.digest
        path = []
        for start in range(0, maximum, self.block_tokens):
            chunk = tokens[start:start + self.block_tokens]
            key = _block_hash(parent, chunk)
            candidate = self._entries.get(key)
            if (candidate is None or candidate.fingerprint != fingerprint
                    or candidate.tokens != chunk):
                break
            path.append(candidate)
            parent = key
        return path

    def lookup(self, fingerprint, tokens, *, minimum_tokens=1):
        with self._lock:
            path = self._path(fingerprint, tokens)
            if not path or path[-1].token_count < minimum_tokens:
                self.misses += 1
                return None
            acquired = []
            layers = [[] for _ in path[0].layer_blocks]
            try:
                for node in path:
                    for index, links in enumerate(node.layer_blocks):
                        for link in links:
                            ref = BlockRef(link.pool, link.block_id, acquired=True)
                            acquired.append(ref)
                            layers[index].append(ref)
            except BaseException:
                for ref in reversed(acquired):
                    ref.close()
                raise
            for node in path:
                node.last_access = time.monotonic()
                self._entries.move_to_end(node.key)
            self.hits += 1
            self.tokens_reused += path[-1].token_count
            return BlockPrefixMatch(path[-1].token_count,
                                    tuple(tuple(layer) for layer in layers), len(acquired))

    def longest_prefix_tokens(self, fingerprint, tokens, *, minimum_tokens=1):
        with self._lock:
            path = self._path(fingerprint, tokens)
            count = path[-1].token_count if path else 0
            return count if count >= minimum_tokens else 0

    def publish(self, fingerprint, tokens, layer_blocks):
        if not layer_blocks or not self.capacity_bytes:
            return 0
        complete = min(*(len(layer) for layer in layer_blocks), len(tokens) // self.block_tokens)
        # Validate every layer before taking any pins.
        for layer in layer_blocks:
            for ref in layer[:complete]:
                if not ref.block.immutable or ref.block.token_count != self.block_tokens:
                    raise ValueError("prefix publication requires sealed complete pages")
        added = []
        parent = fingerprint.digest
        with self._lock:
            try:
                for index in range(complete):
                    chunk = tokens[index * self.block_tokens:(index + 1) * self.block_tokens]
                    key = _block_hash(parent, chunk)
                    if key not in self._entries:
                        links = tuple((BlockLink(layer[index].pool, layer[index].block_id),)
                                      for layer in layer_blocks)
                        retained = []
                        try:
                            for layer in links:
                                for link in layer:
                                    self._retain_pin(link)
                                    retained.append(link)
                        except BaseException:
                            for link in reversed(retained):
                                self._release_pin(link)
                            raise
                        self._entries[key] = BlockPrefixEntry(
                            key, parent, fingerprint, (index + 1) * self.block_tokens,
                            chunk, links, time.monotonic())
                        added.append(key)
                    parent = key
            except BaseException:
                for key in reversed(added):
                    self._release_entry(self._entries.pop(key))
                raise
            self._evict()
            return sum(key in self._entries for key in added)

    def clear(self):
        with self._lock:
            for entry in self._entries.values():
                self._release_entry(entry)
            self._entries.clear()

    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "bytes": self.nbytes,
                    "metadata_bytes": sum(e.metadata_bytes for e in self._entries.values()),
                    "capacity_bytes": self.capacity_bytes, "hits": self.hits,
                    "misses": self.misses, "evictions": self.evictions,
                    "tokens_reused": self.tokens_reused}

    def _retain_pin(self, link):
        identity = (id(link.pool), int(link.block_id))
        owners = self._pin_owners.get(identity, 0)
        if not owners:
            link.pool.pin(link.block_id)
        self._pin_owners[identity] = owners + 1

    def _release_pin(self, link):
        identity = (id(link.pool), int(link.block_id))
        owners = self._pin_owners[identity] - 1
        if not owners:
            del self._pin_owners[identity]
            link.pool.unpin(link.block_id)
        else:
            self._pin_owners[identity] = owners

    def _release_entry(self, entry):
        for layer in entry.layer_blocks:
            for link in layer:
                self._release_pin(link)

    def _evict(self):
        while (self.nbytes > self.capacity_bytes or
               sum(e.metadata_bytes for e in self._entries.values())
               > self.metadata_capacity_bytes):
            parents = {entry.parent for entry in self._entries.values()}
            key = next(key for key in self._entries if key not in parents)
            self._release_entry(self._entries.pop(key))
            self.evictions += 1


def _block_hash(parent: str, tokens: tuple[int, ...]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(parent.encode("ascii") + b":" + encoded).hexdigest()
