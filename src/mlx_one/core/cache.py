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


class EncoderDecoderKVCache:
    """Decoder self-attention cache plus immutable encoder projections."""

    def __init__(self) -> None:
        self.self_attention = KVCache()
        self.cross_keys: Any | None = None
        self.cross_values: Any | None = None

    @property
    def offset(self) -> int:
        return self.self_attention.offset

    def update_self(self, keys: Any, values: Any) -> tuple[Any, Any]:
        return self.self_attention.update(keys, values)

    def update_cross(self, keys: Any, values: Any) -> tuple[Any, Any]:
        if keys.shape != values.shape:
            raise ValueError("cross-attention keys and values must have identical shapes")
        if self.cross_keys is None:
            self.cross_keys, self.cross_values = keys, values
        elif keys.shape != self.cross_keys.shape:
            raise ValueError("cross-attention cache shape changed")
        return self.cross_keys, self.cross_values

    def reset(self) -> None:
        self.self_attention.reset()
        self.cross_keys = None
        self.cross_values = None


def make_encoder_decoder_caches(layer_count: int) -> tuple[EncoderDecoderKVCache, ...]:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    return tuple(EncoderDecoderKVCache() for _ in range(layer_count))


class ConvCache:
    """Fixed-width recurrent state for causal depthwise convolutions."""

    def __init__(self, kernel_size: int) -> None:
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.values: Any | None = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, values: Any) -> Any:
        import mlx.core as mx

        if values.ndim != 3:
            raise ValueError("convolution cache values must have shape [batch, length, hidden]")
        width = self.kernel_size - 1
        if self.values is None:
            prefix = mx.zeros((values.shape[0], width, values.shape[2]), dtype=values.dtype)
        else:
            if values.shape[0] != self.values.shape[0] or values.shape[2] != self.values.shape[2]:
                raise ValueError("cache update shape is incompatible with existing state")
            prefix = self.values
        combined = mx.concatenate((prefix, values), axis=1)
        self.values = combined[:, -width:, :] if width else combined[:, :0, :]
        self._offset += int(values.shape[1])
        return combined

    def reset(self) -> None:
        self.values = None
        self._offset = 0


def make_hybrid_caches(
    layer_types: tuple[str, ...] | list[str], conv_kernel_size: int
) -> tuple[KVCache | ConvCache, ...]:
    """Create the cache required by each LFM hybrid operator."""

    if not layer_types:
        raise ValueError("layer_types cannot be empty")
    caches = []
    for layer_type in layer_types:
        if layer_type in {"full_attention", "sliding_attention"}:
            caches.append(KVCache())
        elif layer_type == "conv":
            caches.append(ConvCache(conv_kernel_size))
        else:
            raise ValueError(f"unsupported hybrid layer type: {layer_type}")
    return tuple(caches)
