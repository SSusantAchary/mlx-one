"""Shared key/value cache used by native decoder architectures."""

from __future__ import annotations

from typing import Any


def _tree_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(item) for item in value)
    amount = getattr(value, "nbytes", None)
    return 0 if amount is None else int(amount)


class KVCache:
    """Append-only cache with sequence data stored on axis two."""

    step = 256

    def __init__(self) -> None:
        self.keys: Any | None = None
        self.values: Any | None = None
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        if keys.shape != values.shape:
            raise ValueError("cache keys and values must have identical shapes")
        previous = self._offset
        if self.keys is not None:
            if keys.shape[:2] != self.keys.shape[:2] or keys.shape[3:] != self.keys.shape[3:]:
                raise ValueError("cache update shape is incompatible with existing state")
        required = previous + int(keys.shape[2])
        if self.keys is None or required > int(self.keys.shape[2]):
            chunks = (int(keys.shape[2]) + self.step - 1) // self.step
            shape = (*keys.shape[:2], chunks * self.step, *keys.shape[3:])
            new_keys = mx.zeros(shape, dtype=keys.dtype)
            new_values = mx.zeros(shape, dtype=values.dtype)
            if self.keys is None:
                self.keys, self.values = new_keys, new_values
            else:
                if previous % self.step:
                    self.keys = self.keys[:, :, :previous, :]
                    self.values = self.values[:, :, :previous, :]
                self.keys = mx.concatenate((self.keys, new_keys), axis=2)
                self.values = mx.concatenate((self.values, new_values), axis=2)
        self._offset = required
        self.keys[:, :, previous:required, :] = keys
        self.values[:, :, previous:required, :] = values
        return self.keys[:, :, :required, :], self.values[:, :, :required, :]

    def state(self) -> tuple[Any | None, Any | None]:
        if self.keys is None:
            return None, None
        return (
            self.keys[:, :, : self._offset, :],
            self.values[:, :, : self._offset, :],
        )

    @property
    def nbytes(self) -> int:
        return _tree_nbytes(self.state())

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self._offset = 0

    def clone(self) -> KVCache:
        result = KVCache()
        result.keys, result.values = self.state()
        result._offset = self._offset
        return result

    snapshot = clone

    def batch_select(self, indices: tuple[int, ...]) -> KVCache:
        result = self.clone()
        if result.keys is not None:
            result.keys = result.keys[list(indices)]
            result.values = result.values[list(indices)]
        return result

    @classmethod
    def merge(cls, caches: tuple[KVCache, ...]) -> KVCache:
        if not caches or len({cache.offset for cache in caches}) != 1:
            raise ValueError("KV caches must be non-empty and have identical offsets")
        import mlx.core as mx

        result = cls()
        states = tuple(cache.state() for cache in caches)
        if states[0][0] is not None:
            result.keys = mx.concatenate(tuple(state[0] for state in states), axis=0)
            result.values = mx.concatenate(tuple(state[1] for state in states), axis=0)
            result._offset = caches[0].offset
        return result

    def trim(self, count: int) -> bool:
        if count < 0:
            raise ValueError("cache trim count cannot be negative")
        if count == 0:
            return True
        if count >= self.offset:
            self.reset()
            return True
        self._offset -= count
        return True

    release = reset


class QuantizedKVCache:
    """Append-only affine-quantized attention cache."""

    step = 256

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

    @property
    def nbytes(self) -> int:
        return _tree_nbytes((self._keys, self._values))

    def update(self, keys: Any, values: Any) -> tuple[Any, Any]:
        if keys.shape != values.shape:
            raise ValueError("cache keys and values must have identical shapes")
        if (self.key_bits is not None or self.value_bits is not None) and (
            keys.shape[-1] % self.group_size
        ):
            raise ValueError(
                f"KV head dimension must be divisible by quantization group {self.group_size}"
            )
        previous = self._offset
        self._keys = self._append_component(
            self._keys, keys, self.key_bits, self.key_dtype, previous
        )
        self._values = self._append_component(
            self._values, values, self.value_bits, self.value_dtype, previous
        )
        self._offset += int(keys.shape[2])
        return self._materialize(self._keys, self.key_bits, self._offset), self._materialize(
            self._values, self.value_bits, self._offset
        )

    def _append_component(
        self,
        current: Any | None,
        values: Any,
        bits: int | None,
        dtype: str,
        previous: int,
    ) -> Any:
        import mlx.core as mx

        if bits is None:
            values = values.astype(mx.bfloat16 if dtype == "bf16" else mx.float16)
            return self._append_array(current, values, previous)
        encoded = mx.quantize(values, group_size=self.group_size, bits=bits)
        normalized = (encoded[0], encoded[1], encoded[2] if len(encoded) > 2 else None)
        existing = current or (None, None, None)
        return tuple(
            None
            if right is None
            else self._append_array(left, right, previous)
            for left, right in zip(existing, normalized, strict=True)
        )

    def _append_array(self, current: Any | None, values: Any, previous: int) -> Any:
        import mlx.core as mx

        required = previous + int(values.shape[2])
        if current is None or required > int(current.shape[2]):
            chunks = (int(values.shape[2]) + self.step - 1) // self.step
            shape = (*values.shape[:2], chunks * self.step, *values.shape[3:])
            extension = mx.zeros(shape, dtype=values.dtype)
            if current is None:
                current = extension
            else:
                if previous % self.step:
                    current = current[:, :, :previous, :]
                current = mx.concatenate((current, extension), axis=2)
        current[:, :, previous:required, :] = values
        return current

    def _materialize(self, state: Any, bits: int | None, length: int | None = None) -> Any:
        import mlx.core as mx

        length = self._offset if length is None else length
        if bits is None:
            return state[:, :, :length, :]
        return mx.dequantize(
            state[0][:, :, :length, :],
            state[1][:, :, :length, :],
            None if state[2] is None else state[2][:, :, :length, :],
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
        result._keys = self._active_component(self._keys)
        result._values = self._active_component(self._values)
        result._offset = self._offset
        return result

    snapshot = clone

    def batch_select(self, indices: tuple[int, ...]) -> QuantizedKVCache:
        result = self.clone()
        result._keys = self._select_component(result._keys, indices)
        result._values = self._select_component(result._values, indices)
        return result

    @classmethod
    def merge(cls, caches: tuple[QuantizedKVCache, ...]) -> QuantizedKVCache:
        if not caches or len({cache.offset for cache in caches}) != 1:
            raise ValueError("quantized KV caches must have identical offsets")
        signature = {
            (cache.key_bits, cache.value_bits, cache.group_size, cache.key_dtype, cache.value_dtype)
            for cache in caches
        }
        if len(signature) != 1:
            raise ValueError("quantized KV cache formats do not match")
        import mlx.core as mx

        first = caches[0]
        result = cls(
            first.key_bits,
            first.value_bits,
            first.group_size,
            key_dtype=first.key_dtype,
            value_dtype=first.value_dtype,
        )
        result._keys = cls._merge_components(
            tuple(cache._active_component(cache._keys) for cache in caches), mx
        )
        result._values = cls._merge_components(
            tuple(cache._active_component(cache._values) for cache in caches), mx
        )
        result._offset = first.offset
        return result

    def trim(self, count: int) -> bool:
        if count < 0:
            raise ValueError("cache trim count cannot be negative")
        if count == 0:
            return True
        if count >= self.offset:
            self.reset()
            return True
        self._offset -= count
        return True

    def _active_component(self, state: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(
                None if value is None else value[:, :, : self._offset, :]
                for value in state
            )
        return state[:, :, : self._offset, :]

    @staticmethod
    def _select_component(state: Any, indices: tuple[int, ...]) -> Any:
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(None if value is None else value[list(indices)] for value in state)
        return state[list(indices)]

    @staticmethod
    def _merge_components(states: tuple[Any, ...], mx: Any) -> Any:
        if states[0] is None:
            return None
        if isinstance(states[0], tuple):
            return tuple(
                None
                if states[0][index] is None
                else mx.concatenate(tuple(state[index] for state in states), axis=0)
                for index in range(len(states[0]))
            )
        return mx.concatenate(states, axis=0)

    release = reset


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

    @property
    def nbytes(self) -> int:
        return self.self_attention.nbytes + _tree_nbytes(
            (self.cross_keys, self.cross_values)
        )

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

    snapshot = clone

    def batch_select(self, indices: tuple[int, ...]) -> EncoderDecoderKVCache:
        result = EncoderDecoderKVCache()
        result.self_attention = self.self_attention.batch_select(indices)
        if self.cross_keys is not None:
            result.cross_keys = self.cross_keys[list(indices)]
            result.cross_values = self.cross_values[list(indices)]
        return result

    @classmethod
    def merge(
        cls, caches: tuple[EncoderDecoderKVCache, ...]
    ) -> EncoderDecoderKVCache:
        if not caches:
            raise ValueError("cannot merge an empty encoder-decoder cache sequence")
        import mlx.core as mx

        result = cls()
        result.self_attention = KVCache.merge(
            tuple(cache.self_attention for cache in caches)
        )
        if caches[0].cross_keys is not None:
            if any(cache.cross_keys is None for cache in caches):
                raise ValueError("encoder-decoder cross-attention states do not match")
            result.cross_keys = mx.concatenate(
                tuple(cache.cross_keys for cache in caches), axis=0
            )
            result.cross_values = mx.concatenate(
                tuple(cache.cross_values for cache in caches), axis=0
            )
        return result

    def trim(self, count: int) -> bool:
        return self.self_attention.trim(count)

    release = reset


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

    @property
    def nbytes(self) -> int:
        return _tree_nbytes(self.values)

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

    snapshot = clone

    def batch_select(self, indices: tuple[int, ...]) -> ConvCache:
        result = self.clone()
        if result.values is not None:
            result.values = result.values[list(indices)]
        return result

    @classmethod
    def merge(cls, caches: tuple[ConvCache, ...]) -> ConvCache:
        if not caches or len({(cache.kernel_size, cache.offset) for cache in caches}) != 1:
            raise ValueError("convolution caches are incompatible")
        import mlx.core as mx

        result = cls(caches[0].kernel_size)
        if caches[0].values is not None:
            result.values = mx.concatenate(tuple(cache.values for cache in caches), axis=0)
        result._offset = caches[0].offset
        return result
    release = reset


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
