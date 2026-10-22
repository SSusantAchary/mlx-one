"""Correctness-first bounded KV state with an absolute sequence position."""

from mlx_one.core.cache import KVCache


class SlidingKVCache(KVCache):
    supports_quantization = False
    supports_batching = False

    def __init__(self, max_size: int) -> None:
        super().__init__()
        if type(max_size) is not int or max_size < 1:
            raise ValueError("sliding cache window must be a positive integer")
        self.max_size = max_size
        self.storage_dtype = None

    def update(self, keys, values):
        import mlx.core as mx

        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
            raise ValueError("cache keys and values must have matching batch/head/token shapes")
        if self.storage_dtype is not None:
            keys, values = (x.astype(getattr(mx, self.storage_dtype)) for x in (keys, values))
        if self.keys is not None:
            if keys.shape[:2] != self.keys.shape[:2] or keys.shape[-1] != self.keys.shape[-1]:
                raise ValueError("sliding cache shape mismatch")
            keys = mx.concatenate((self.keys, keys), axis=2)
            values = mx.concatenate((self.values, values), axis=2)
        count = keys.shape[2] - (0 if self.keys is None else self.keys.shape[2])
        self._offset += count
        # Return all keys needed by queries in this chunk before retaining its tail.
        self.keys, self.values = keys[:, :, -self.max_size :], values[:, :, -self.max_size :]
        return keys, values

    def clone(self):
        import mlx.core as mx

        result = SlidingKVCache(self.max_size)
        result.storage_dtype = self.storage_dtype
        if self.keys is not None:
            result.keys, result.values = mx.array(self.keys), mx.array(self.values)
        result._offset = self.offset
        return result

    snapshot = clone

    def trim(self, count):
        if count < 0:
            raise ValueError("cache trim count cannot be negative")
        if count == 0:
            return True
        if count >= self.offset:
            self.reset()
            return True
        return False

    @classmethod
    def merge(cls, caches):
        raise ValueError("sliding cache batching requires separate qualification")
