"""Byte-bounded, process-local prefix cache with deterministic LRU eviction."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

from mlx_one.engine.cache import CacheBundle


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
