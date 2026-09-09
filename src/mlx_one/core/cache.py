"""Shared key/value cache used by native decoder architectures."""

from __future__ import annotations

from typing import Any


class KVCache:
    """Append-only cache with sequence data stored on axis two."""

    def __init__(self) -> None:
        self.keys: Any | None = None
        self.values: Any | None = None

    @property
    def offset(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def update(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        if keys.shape != values.shape:
            raise ValueError("cache keys and values must have identical shapes")
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            if keys.shape[:2] != self.keys.shape[:2] or keys.shape[3:] != self.keys.shape[3:]:
                raise ValueError("cache update shape is incompatible with existing state")
            self.keys = mx.concatenate((self.keys, keys), axis=2)
            self.values = mx.concatenate((self.values, values), axis=2)
        return self.keys, self.values

    def state(self) -> tuple[Any | None, Any | None]:
        return self.keys, self.values

    def reset(self) -> None:
        self.keys = None
        self.values = None


def make_kv_caches(layer_count: int) -> tuple[KVCache, ...]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    return tuple(KVCache() for _ in range(layer_count))
