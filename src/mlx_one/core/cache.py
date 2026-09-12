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

    def clone(self) -> KVCache:
        result = KVCache()
        result.keys, result.values = self.keys, self.values
        return result


class QuantizedKVCache:
    """Append-only affine-quantized attention cache."""

    def __init__(
        self,
        key_bits: int | None,
        value_bits: int | None,
        group_size: int = 64,
        *,
        key_dtype: str = "f16",
        value_dtype: str = "f16",
    ) -> None:
        if key_bits not in {None, 4, 8} or value_bits not in {None, 4, 8}:
            raise ValueError("quantized KV cache bits must be native, 4, or 8")
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self._keys: Any | None = None
        self._values: Any | None = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, keys: Any, values: Any) -> tuple[Any, Any]:
        if keys.shape != values.shape:
            raise ValueError("cache keys and values must have identical shapes")
        if (self.key_bits is not None or self.value_bits is not None) and (
            keys.shape[-1] % self.group_size
        ):
            raise ValueError(
                f"KV head dimension must be divisible by quantization group {self.group_size}"
            )
        self._keys = self._append_component(
            self._keys, keys, self.key_bits, self.key_dtype
        )
        self._values = self._append_component(
            self._values, values, self.value_bits, self.value_dtype
        )
        self._offset += int(keys.shape[2])
        return self._materialize(self._keys, self.key_bits), self._materialize(
            self._values, self.value_bits
        )

    def _append_component(
        self, current: Any | None, values: Any, bits: int | None, dtype: str
    ) -> Any:
        import mlx.core as mx

        if bits is None:
            values = values.astype(mx.bfloat16 if dtype == "bf16" else mx.float16)
            return values if current is None else mx.concatenate((current, values), axis=2)
        new = mx.quantize(values, group_size=self.group_size, bits=bits)
        normalized = (new[0], new[1], new[2] if len(new) > 2 else None)
        if current is None:
            return normalized
        return tuple(
            None if right is None else mx.concatenate((left, right), axis=2)
            for left, right in zip(current, normalized, strict=True)
        )

    def _materialize(self, state: Any, bits: int | None) -> Any:
        import mlx.core as mx

        if bits is None:
            return state
        return mx.dequantize(
            state[0],
            state[1],
            state[2],
            group_size=self.group_size,
            bits=bits,
        )

    def reset(self) -> None:
        self._keys = None
        self._values = None
        self._offset = 0

    def clone(self) -> QuantizedKVCache:
        result = QuantizedKVCache(
            self.key_bits,
            self.value_bits,
            self.group_size,
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
        )
        result._keys, result._values, result._offset = self._keys, self._values, self._offset
        return result


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

    def clone(self) -> EncoderDecoderKVCache:
        result = EncoderDecoderKVCache()
        result.self_attention = self.self_attention.clone()
        result.cross_keys, result.cross_values = self.cross_keys, self.cross_values
        return result


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

    def clone(self) -> ConvCache:
        result = ConvCache(self.kernel_size)
        result.values, result._offset = self.values, self._offset
        return result


def clone_caches(caches: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(cache.clone() for cache in caches)


def configure_kv_caches(
    caches: tuple[Any, ...], key_type: str, value_type: str
) -> tuple[Any, ...]:
    """Replace attention cache entries with requested native or quantized storage."""

    key_bits = {"q4_0": 4, "q8_0": 8}.get(key_type)
    value_bits = {"q4_0": 4, "q8_0": 8}.get(value_type)
    return tuple(
        QuantizedKVCache(
            key_bits,
            value_bits,
            key_dtype=key_type,
            value_dtype=value_type,
        )
        if isinstance(cache, KVCache)
        else cache
        for cache in caches
    )


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
