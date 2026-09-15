"""Cache ownership, capability discovery, and exact byte accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_one.engine.contracts import CacheCapabilities


def array_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(array_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_nbytes(item) for item in value)
    amount = getattr(value, "nbytes", None)
    if amount is not None:
        return int(amount)
    return 0


def cache_nbytes(cache: Any) -> int:
    amount = getattr(cache, "nbytes", None)
    if amount is not None:
        return int(amount)
    state = getattr(cache, "state", None)
    if callable(state):
        return array_nbytes(state())
    values = vars(cache) if hasattr(cache, "__dict__") else cache
    return array_nbytes(values)


def cache_capabilities(cache: Any) -> CacheCapabilities:
    name = type(cache).__name__.lower()
    attention = "kvcache" in name or "attention" in name
    recurrent = "linear" in name or "conv" in name
    return CacheCapabilities(
        batchable=attention and not recurrent,
        trimmable=callable(getattr(cache, "trim", None)),
        quantizable=attention and not recurrent,
        sliding_window=bool(getattr(cache, "max_size", None)),
        block_compatible=attention and not recurrent,
    )


@dataclass
class CacheBundle:
    entries: tuple[Any, ...]

    @property
    def nbytes(self) -> int:
        return sum(cache_nbytes(entry) for entry in self.entries)

    @property
    def offset(self) -> int:
        offsets = [int(getattr(entry, "offset", 0)) for entry in self.entries]
        return max(offsets, default=0)

    @property
    def capabilities(self) -> tuple[CacheCapabilities, ...]:
        return tuple(cache_capabilities(entry) for entry in self.entries)

    def clone(self) -> CacheBundle:
        return CacheBundle(tuple(entry.clone() for entry in self.entries))

    snapshot = clone

    def reset(self) -> None:
        for entry in self.entries:
            entry.reset()

    release = reset

    def trim(self, count: int) -> bool:
        if count < 0:
            raise ValueError("cache trim count cannot be negative")
        if not all(callable(getattr(entry, "trim", None)) for entry in self.entries):
            return False
        snapshots = self.clone()
        try:
            complete = all(bool(entry.trim(count)) for entry in self.entries)
            if not complete:
                self.entries = snapshots.entries
            return complete
        except Exception:
            self.entries = snapshots.entries
            raise

    def batch_select(self, indices: tuple[int, ...]) -> CacheBundle:
        if not indices:
            raise ValueError("cache batch selection cannot be empty")
        selected = []
        for entry in self.entries:
            operation = getattr(entry, "batch_select", None)
            if not callable(operation):
                raise ValueError(f"{type(entry).__name__} is not batch selectable")
            selected.append(operation(indices))
        return CacheBundle(tuple(selected))

    @classmethod
    def merge(cls, bundles: tuple[CacheBundle, ...]) -> CacheBundle:
        if not bundles:
            raise ValueError("cannot merge an empty cache bundle sequence")
        width = len(bundles[0].entries)
        if any(len(bundle.entries) != width for bundle in bundles):
            raise ValueError("cache bundles have incompatible layer counts")
        entries = []
        for layer in range(width):
            caches = tuple(bundle.entries[layer] for bundle in bundles)
            operation = getattr(type(caches[0]), "merge", None)
            if not callable(operation) or any(type(item) is not type(caches[0]) for item in caches):
                raise ValueError("cache bundles contain incompatible cache types")
            entries.append(operation(caches))
        return cls(tuple(entries))
