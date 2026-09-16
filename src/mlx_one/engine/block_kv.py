"""Block-backed attention KV compatible with existing model attention layers."""

from __future__ import annotations

from typing import Any

from mlx_one.engine.block_cache import BlockPool, BlockRef
from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.cache_errors import CacheCorruptionError


class BlockKVCache:
    """B=1, unquantized attention cache stored in fixed-width blocks."""

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.block_tokens = pool.block_tokens
        self._blocks: list[BlockRef] = []
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def nbytes(self) -> int:
        total = 0
        for ref in self._blocks:
            block = ref.block
            keys, values = block.state
            total += array_nbytes(keys[:, :, : block.token_count, :])
            total += array_nbytes(values[:, :, : block.token_count, :])
        return total

    @property
    def allocated_nbytes(self) -> int:
        return sum(ref.block.nbytes for ref in self._blocks)

    @property
    def full_block_refs(self) -> tuple[BlockRef, ...]:
        return tuple(
            ref for ref in self._blocks if ref.block.token_count == self.block_tokens
        )

    def update(self, keys: Any, values: Any) -> tuple[Any, Any]:
        import mlx.core as mx

        if keys.shape != values.shape:
            raise ValueError("cache keys and values must have identical shapes")
        if int(keys.shape[0]) != 1:
            raise ValueError("block KV currently supports batch size one")
        if self._blocks:
            current_keys, _ = self._blocks[0].block.state
            if keys.shape[:2] != current_keys.shape[:2] or keys.shape[3:] != current_keys.shape[3:]:
                raise ValueError("cache update shape is incompatible with existing state")
        cursor = 0
        length = int(keys.shape[2])
        while cursor < length:
            tail = self._blocks[-1].block if self._blocks else None
            if tail is None or tail.token_count == self.block_tokens:
                shape = (*keys.shape[:2], self.block_tokens, *keys.shape[3:])
                storage = (mx.zeros(shape, dtype=keys.dtype), mx.zeros(shape, dtype=values.dtype))
                ref = self.pool.allocate_ref(storage, 1)
                self._blocks.append(ref)
                tail = ref.block
                start = 0
            else:
                if tail.immutable or tail.references != 1 or tail.pins:
                    raise CacheCorruptionError("mutable KV tail is unexpectedly shared")
                start = tail.token_count
            width = min(self.block_tokens - start, length - cursor)
            block_keys, block_values = tail.state
            block_keys[:, :, start : start + width, :] = keys[:, :, cursor : cursor + width, :]
            block_values[:, :, start : start + width, :] = values[:, :, cursor : cursor + width, :]
            tail.token_count = start + width
            cursor += width
            self._offset += width
            if tail.token_count == self.block_tokens:
                self.pool.seal(tail.block_id)
        return self._materialize(mx)

    def state(self) -> tuple[Any | None, Any | None]:
        if not self._blocks:
            return None, None
        import mlx.core as mx

        return self._materialize(mx)

    def _materialize(self, mx: Any) -> tuple[Any, Any]:
        states = tuple(
            (
                ref.block.state[0][:, :, : ref.block.token_count, :],
                ref.block.state[1][:, :, : ref.block.token_count, :],
            )
            for ref in self._blocks
        )
        if len(states) == 1:
            return states[0]
        return (
            mx.concatenate(tuple(item[0] for item in states), axis=2),
            mx.concatenate(tuple(item[1] for item in states), axis=2),
        )

    def adopt(self, references: tuple[BlockRef, ...]) -> None:
        if self._blocks or self._offset:
            raise CacheCorruptionError("prefix adoption requires an empty block cache")
        for reference in references:
            block = reference.block
            if not block.immutable or block.token_count != self.block_tokens:
                raise CacheCorruptionError("only sealed complete blocks may be adopted")
        self._blocks.extend(references)
        self._offset = len(references) * self.block_tokens

    def clone(self) -> BlockKVCache:
        import mlx.core as mx

        result = BlockKVCache(self.pool)
        for ref in self._blocks:
            block = ref.block
            if block.immutable:
                result._blocks.append(ref.fork())
                result._offset += block.token_count
                continue
            keys, values = block.state
            shape = keys.shape
            copied_keys = mx.zeros(shape, dtype=keys.dtype)
            copied_values = mx.zeros(shape, dtype=values.dtype)
            copied_keys[:, :, : block.token_count, :] = keys[:, :, : block.token_count, :]
            copied_values[:, :, : block.token_count, :] = values[:, :, : block.token_count, :]
            result._blocks.append(
                self.pool.allocate_ref((copied_keys, copied_values), block.token_count)
            )
            result._offset += block.token_count
        return result

    snapshot = clone

    def trim(self, count: int) -> bool:
        if count < 0:
            raise ValueError("cache trim count cannot be negative")
        if count == 0:
            return True
        target = max(self._offset - count, 0)
        if target == 0:
            self.reset()
            return True
        keys, values = self.state()
        assert keys is not None and values is not None
        keys, values = keys[:, :, :target, :], values[:, :, :target, :]
        self.reset()
        self.update(keys, values)
        return True

    def reset(self) -> None:
        for ref in self._blocks:
            ref.close()
        self._blocks.clear()
        self._offset = 0

    release = reset
