"""Transactional segmented page storage and direct attention dispatch.

Each segment contains 16 pages. Launches bind a bounded number of segments;
neither appending nor attention constructs a tensor containing the KV history.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import replace

from mlx_one.engine.arena import PersistentPageTable, SegmentedPageArena
from mlx_one.engine.batching import BatchedDenseRunner, request_cancelled
from mlx_one.engine.block_cache import BlockLayout
from mlx_one.engine.block_kv import BlockKVCache
from mlx_one.engine.cache import array_nbytes
from mlx_one.engine.contracts import BatchPlan, RunnerOutput, ScheduledSequence
from mlx_one.engine.modes import ModelCapabilityUnsupported

KERNEL_VERSION = "segmented-arena-online-softmax-v2"


def quantize_page(value, bits):
    """Affine unsigned groups of 32, padded head dimension, FP16 metadata."""
    import mlx.core as mx

    if bits not in {4, 8}:
        raise ValueError("paged quantization supports 4 or 8 bits")
    dim = value.shape[-1]
    padded = (dim + 31) // 32 * 32
    data = mx.pad(value.astype(mx.float32), ((0, 0), (0, 0), (0, 0), (0, padded - dim)))
    grouped = data.reshape(*data.shape[:-1], padded // 32, 32)
    bias = mx.min(grouped, axis=-1, keepdims=True).astype(mx.float16)
    scale = mx.maximum((mx.max(grouped, axis=-1, keepdims=True) - bias) / ((1 << bits) - 1),
                       1e-6).astype(mx.float16)
    encoded = mx.clip(mx.round((grouped - bias) / scale), 0, (1 << bits) - 1)
    encoded = encoded.reshape(*data.shape).astype(mx.uint8)
    if bits == 4:
        encoded = encoded[..., ::2] | (encoded[..., 1::2] << 4)
    return encoded, scale[..., 0], bias[..., 0]


def dequantize_page(state, bits, dim):
    """Diagnostic oracle only; production attention decodes individual scalars."""
    import mlx.core as mx

    data, scale, bias = state
    if bits == 4:
        data = mx.stack((data & 15, data >> 4), axis=-1).reshape(
            *data.shape[:-1], data.shape[-1] * 2)
    grouped = data.astype(mx.float32).reshape(*data.shape[:-1], -1, 32)
    return (grouped * scale[..., None] + bias[..., None]).reshape(
        *data.shape)[..., :dim]


class PagedKVCache(BlockKVCache):
    """Versioned page arena adapter; only a private mutable tail is copied."""

    def __init__(self, pool, *, layer_id=0, dtype="float16", bits=None):
        super().__init__(pool, layer_id=layer_id)
        if dtype not in {"float16", "bfloat16"} or bits not in {None, 4, 8}:
            raise ModelCapabilityUnsupported("unsupported paged storage precision")
        self.dtype, self.bits = dtype, bits
        if not isinstance(pool, SegmentedPageArena):
            raise ValueError("paged storage requires a segmented arena")
        self.page_table = PersistentPageTable(pool)

    @property
    def nbytes(self):
        return sum(ref.block.nbytes * ref.block.token_count // self.block_tokens
                   for ref in self._blocks)

    def empty(self):
        return PagedKVCache(self.pool, layer_id=self.layer_id, dtype=self.dtype, bits=self.bits)

    def clone(self):
        import mlx.core as mx

        result = self.empty()
        result.page_table = self.page_table.fork()
        try:
            for ref in self._blocks:
                block = ref.block
                if block.immutable:
                    copied = ref.fork()
                else:
                    storage = tuple(mx.zeros(x.shape, dtype=x.dtype) for x in block.state)
                    for dst, src in zip(storage, block.state, strict=True):
                        dst[...] = src
                    copied = self.pool.allocate_ref(storage, block.token_count, layout=block.layout)
                result._blocks.append(copied)
                result.page_table.update(len(result._blocks) - 1, copied)
                result._offset += block.token_count
            return result
        except BaseException:
            result.reset()
            self.pool.discard_free()
            raise

    snapshot = clone

    def append(self, keys, values):
        import mlx.core as mx

        if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError("paged append requires matching B=1 K/V tensors")
        keys, values = keys.astype(getattr(mx, self.dtype)), values.astype(getattr(mx, self.dtype))
        if self._blocks:
            layout = self._blocks[0].block.layout
            if (layout.num_kv_heads, layout.head_dim) != (keys.shape[1], keys.shape[3]):
                raise ValueError("paged append layout mismatch")
        cursor = 0
        while cursor < keys.shape[2]:
            tail = self._blocks[-1].block if self._blocks else None
            if tail is None or tail.immutable:
                shape = (1, keys.shape[1], self.block_tokens, keys.shape[3])
                storage = tuple(mx.zeros(shape, dtype=keys.dtype) for _ in range(2))
                layout = BlockLayout(version=3, layer_id=self.layer_id,
                                     page_tokens=self.block_tokens,
                                     num_kv_heads=int(keys.shape[1]), head_dim=int(keys.shape[3]),
                                     key_dtype=self.dtype, value_dtype=self.dtype,
                                     quantization_bits=None, kernel_version=KERNEL_VERSION)
                self._blocks.append(self.pool.allocate_ref(storage, 1, layout=layout))
                tail, start = self._blocks[-1].block, 0
            else:
                if tail.references != 1 or tail.pins:
                    raise ValueError("mutable paged tail is shared")
                start = tail.token_count
            width = min(self.block_tokens - start, keys.shape[2] - cursor)
            storage = tuple(mx.zeros(x.shape, dtype=x.dtype) for x in tail.state)
            for dst, src in zip(storage, tail.state, strict=True):
                dst[...] = src
            tail.state = storage
            for dst, src in zip(tail.state, (keys, values), strict=True):
                dst[:, :, start:start + width, :] = src[:, :, cursor:cursor + width, :]
            tail.token_count = start + width
            self._offset += width
            cursor += width
            mx.eval(*tail.state)
            self.pool.store(tail.block_id)
            if tail.token_count == self.block_tokens:
                if self.bits is not None:
                    storage = (*quantize_page(tail.state[0], self.bits),
                               *quantize_page(tail.state[1], self.bits))
                    mx.eval(*storage)
                    sealed = self.pool.allocate_ref(storage, self.block_tokens, layout=replace(
                        tail.layout, quantization_bits=self.bits))
                    old = self._blocks[-1]
                    self._blocks[-1] = sealed
                    old.close()
                    tail = sealed.block
                self.pool.seal(tail.block_id)
            self.page_table.update(len(self._blocks) - 1, self._blocks[-1])

    def replace_from(self, draft):
        table = draft.page_table
        super().replace_from(draft)
        self.page_table = table
        draft.page_table = PersistentPageTable(self.pool)

    def adopt(self, references):
        for reference in references:
            layout = reference.block.layout
            if (layout.version != 3 or layout.kernel_version != KERNEL_VERSION
                    or layout.key_dtype != self.dtype or layout.value_dtype != self.dtype
                    or layout.quantization_bits != self.bits):
                raise ModelCapabilityUnsupported("prefix page precision or layout mismatch")
        super().adopt(references)
        for index, reference in enumerate(references):
            self.page_table.update(index, reference)

    def reset(self):
        super().reset()
        self.page_table = PersistentPageTable(self.pool)

    def update(self, *_):
        raise ModelCapabilityUnsupported("paged cache requires direct attention dispatch")

    def state(self):
        raise ModelCapabilityUnsupported("paged histories cannot be materialized")

    def trim(self, count):
        if count < 0:
            raise ValueError("negative trim")
        target = max(self.offset - count, 0)
        if target % self.block_tokens:
            raise ModelCapabilityUnsupported("paged restoration requires complete pages")
        while self._offset > target:
            ref = self._blocks.pop()
            self._offset -= ref.block.token_count
            ref.close()
        self.page_table.truncate(len(self._blocks))
        return True


class PagedKVComponent(PagedKVCache):
    """Hybrid attention owner; carries stay outside the arena."""
    kind = "full_kv"

    @property
    def committed_position(self):
        return self.offset

    def capabilities(self):
        from mlx_one.engine.components import ComponentCapabilities

        return ComponentCapabilities(can_batch=True, can_page=True, can_snapshot=True,
                                     can_share_prefix=True, can_trim=True, transition_math=True)

    def empty(self):
        return PagedKVComponent(self.pool, layer_id=self.layer_id, dtype=self.dtype)

    def dispose(self):
        self.reset()

    def validate(self, spec):
        if spec.kind != self.kind or spec.layer_id != self.layer_id or self.offset < 0:
            raise ValueError("hybrid page topology mismatch")
        for ref in self._blocks:
            layout = ref.block.layout
            if (layout.num_kv_heads != spec.num_kv_heads
                    or layout.head_dim != spec.key_head_dim or layout.key_dtype != self.dtype
                    or spec.dtype and self.dtype != spec.dtype or self.bits is not None):
                raise ValueError("hybrid page layout or precision mismatch")


class PagedBatchCache:
    def __init__(self, rows, widths, metrics):
        self.rows, self.widths, self.metrics = rows, widths, metrics
        self.offset = max(row.offset for row in rows)

    def attend(self, queries, keys, values, *, scale, position_ids):
        import mlx.core as mx

        from mlx_one.engine.paged_attention import paged_attention

        positions = []
        for i, (cache, width) in enumerate(zip(self.rows, self.widths, strict=True)):
            positions.append(tuple(range(cache.offset, cache.offset + queries.shape[2])))
            cache.append(keys[i:i + 1, :, :width], values[i:i + 1, :, :width])
        mx.eval(queries)
        started = time.perf_counter()
        result = paged_attention(queries, self.rows, positions, self.widths, scale=scale)
        mx.eval(result)
        self.metrics["synchronized_kernel_dispatch_ms"] += (time.perf_counter() - started) * 1000
        self.metrics["direct_attention_calls"] += 1
        return result


class PagedRunner(BatchedDenseRunner):
    """Atomic page-native ragged prefill and decode through one model forward."""

    def __init__(self, bundle=None):
        super().__init__(bundle)
        self.contract_pool = SegmentedPageArena(2**63 - 1)

    def capabilities(self):
        return replace(super().capabilities(), supports_paged_attention=True,
                       supports_exact_prefix_restore=True, supported_cache_backends=("paged",))

    def make_cache(self):
        from mlx_one.engine.cache import CacheBundle
        from mlx_one.engine.reporting import tensor_dtypes

        dtype = next(d for d in tensor_dtypes(self.bundle.model.parameters())
                     if d in {"float16", "bfloat16"})
        return CacheBundle(tuple(PagedKVCache(self.contract_pool, layer_id=i, dtype=dtype)
                                 for i in range(self.bundle.model.config.num_hidden_layers)))

    def _resolve(self, plan, phase):
        if plan.execution_mode not in {"dense", "paged"}:
            raise ValueError("incompatible paged execution plan")
        return super()._resolve(replace(plan, execution_mode="dense"), phase)

    def release(self):
        super().release()
        self.contract_pool.clear()

    def _validate_bundle(self, bundle):
        from mlx_one.engine.cache_planner import build_cache_plan
        from mlx_one.engine.modes import validate_mode

        validate_mode(bundle, build_cache_plan(bundle), "paged")
        from mlx_one.engine.adapters import native_adapter

        self.adapter = native_adapter(bundle, allow_bfloat16=True)

    def execute(self, requests):
        import mlx.core as mx

        if not requests:
            raise ValueError("empty paged batch")
        first = requests[0]
        self._validate_bundle(first.bundle)
        if any(not r.cache for r in requests):
            raise ValueError("empty paged layer mapping")
        slots = tuple(ScheduledSequence(r.request_id, r.phase, r.cache[0].offset, len(r.token_ids))
                      for r in requests)
        plan = BatchPlan(slots, sum(len(r.token_ids) for r in requests), execution_mode="paged")
        for r in requests:
            if (r.bundle is not first.bundle or r.phase != first.phase
                    or r.model_epoch != first.model_epoch or len(r.cache) != len(first.cache)):
                raise ValueError("incompatible paged batch")
            if (not r.cache or any(not isinstance(c, PagedKVCache) for c in r.cache)
                    or len({c.offset for c in r.cache}) != 1
                    or any(c.layer_id != i for i, c in enumerate(r.cache))
                    or len(r.cache) != first.bundle.model.config.num_hidden_layers):
                raise ValueError("invalid paged layer mappings")
            if r.handle is not None:
                r.runtime._validate_handle(r.handle)
                if r.handle.bundle.entries is not r.cache:
                    raise ValueError("paged request does not own its cache mapping")
        if len({(id(c.pool), c.dtype, c.bits) for r in requests for c in r.cache}) != 1:
            raise ValueError("incompatible paged pools or precision")
        widths = tuple(len(r.token_ids) for r in requests)
        width = max(widths)
        started = time.perf_counter()
        metrics = {"direct_attention_calls": 0, "history_materializations": 0,
                   "conversion_bytes": 0, "kernel_version": KERNEL_VERSION,
                   "synchronized_kernel_dispatch_ms": 0.0, "gpu_timestamp_latency_ms": None}
        drafts = []
        runtime = first.runtime
        heads = int(first.bundle.model.config.num_attention_heads)
        dim = int(getattr(first.bundle.model.config, "head_dim", 0)
                  or first.bundle.model.config.hidden_size // heads)
        transient = len(requests) * heads * width * (dim + 2) * 4 * 3
        transient += (len(requests) * len(first.cache) * dim
                      * first.bundle.model.config.num_key_value_heads
                      * first.cache[0].block_tokens * 16 * 4)
        transient += 16 * sum(array_nbytes(ref.block.state) for r in requests for c in r.cache
                         for ref in c._blocks if not ref.block.immutable)
        from mlx_one.engine.pressure import execution_forecast

        transient = execution_forecast(first.runtime, first.bundle.model.config,
                                        batch=len(requests),
                                        chunk=width, history=max(r.cache[0].offset
                                                                 for r in requests),
                                        explicit_bytes=transient,
                                        history_lengths=[r.cache[0].offset for r in requests],
                                        widths=widths, phase=first.phase)["bytes"]
        if any(r.runtime is not runtime for r in requests):
            raise ValueError("paged batch spans cache managers")
        with ExitStack() as stack:
            if runtime is not None:
                stack.enter_context(runtime.transient_reservation(transient))
            try:
                for r in requests:
                    layers = []
                    drafts.append(layers)
                    for c in r.cache:
                        layers.append(c.clone())
                packed = tuple(PagedBatchCache(tuple(d[layer] for d in drafts), widths, metrics)
                               for layer in range(len(first.cache)))
                ids = mx.array([list(r.token_ids) + [0] * (width - len(r.token_ids))
                                for r in requests])
                positions = mx.array([list(range(r.cache[0].offset, r.cache[0].offset + width))
                                      for r in requests])
                output = self.adapter.forward(ids, cache=packed, position_ids=positions)
                logits = []
                for i, n in enumerate(widths):
                    begin = 0 if requests[i].return_all_logits else n - 1
                    result = mx.zeros((1, n - begin, output.logits.shape[-1]),
                                    dtype=output.logits.dtype)
                    result[...] = output.logits[i:i + 1, begin:n]
                    logits.append(result)
                mx.eval(*logits)
                for r, layers in zip(requests, drafts, strict=True):
                    if request_cancelled(r):
                        continue
                    for original, draft in zip(r.cache, layers, strict=True):
                        original.replace_from(draft)
                    if r.handle is not None:
                        r.handle.computed_tokens = r.cache[0].offset
                        runtime.reconcile(r.handle)
            finally:
                for layers in drafts:
                    for c in layers:
                        c.reset()
        metrics["arena"] = runtime.block_pool.stats() if runtime is not None else {}
        self.last_trace = {**metrics, "phase": first.phase,
                           "request_ids": [s.request_id for s in plan.sequences],
                           "batch_size": len(requests),
                           "history_lengths": [s.start_pos for s in slots],
                           "real_tokens": sum(widths), "padded_tokens": len(requests) * width,
                           "padding_efficiency": sum(widths) / (len(requests) * width),
                           "transient_reserved_bytes": transient,
                           "duration_ms": (time.perf_counter() - started) * 1000}
        return RunnerOutput(tuple(r.request_id for r in requests), tuple(logits), widths)
