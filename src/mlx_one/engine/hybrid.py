"""Component-aware ragged LFM2 execution with isolated convolution carries."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import replace

from mlx_one.engine.batching import BatchedDenseRunner, request_cancelled
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.components import ConvCarryComponent
from mlx_one.engine.contracts import BatchPlan, RunnerOutput, ScheduledSequence


class _AttentionRows:
    def __init__(self, rows, widths):
        self.rows, self.widths = rows, widths
        self.offset = max(c.offset for c in rows)

    def attend(self, queries, keys, values, *, scale, position_ids):
        import mlx.core as mx

        batch, heads, width, dim = queries.shape
        lengths = [0 if c.keys is None else min(c.offset, c.keys.shape[2]) for c in self.rows]
        history = max(lengths)
        kv_heads = keys.shape[1]
        shape = (batch, kv_heads, history + width, dim)
        dtype = getattr(mx, self.rows[0].storage_dtype)
        packed_k, packed_v = mx.zeros(shape, dtype=dtype), mx.zeros(shape, dtype=dtype)
        key_positions, query_positions = [], []
        for i, (cache, n, old_len) in enumerate(zip(self.rows, self.widths, lengths, strict=True)):
            offset = cache.offset
            start = history - old_len
            if old_len:
                packed_k[i:i + 1, :, start:history] = cache.keys[:, :, :old_len]
                packed_v[i:i + 1, :, start:history] = cache.values[:, :, :old_len]
            packed_k[i:i + 1, :, history:] = keys[i:i + 1]
            packed_v[i:i + 1, :, history:] = values[i:i + 1]
            key_positions.append([-1] * start + list(range(offset - old_len, offset))
                                 + list(range(offset, offset + n)) + [-1] * (width - n))
            query_positions.append(list(range(offset, offset + width)))
            cache.update(keys[i:i + 1, :, :n], values[i:i + 1, :, :n])
        qp = mx.array(query_positions)[:, :, None]
        kp = mx.array(key_positions)[:, None, :]
        allowed = (kp >= 0) & (kp <= qp)
        windows = {getattr(c, "max_size", None) for c in self.rows}
        if len(windows) != 1:
            raise ValueError("incompatible sliding windows")
        window = next(iter(windows))
        if window is not None:
            allowed = allowed & (kp > qp - window)
        repeats = heads // kv_heads
        if repeats > 1:
            packed_k, packed_v = mx.repeat(packed_k, repeats, axis=1), mx.repeat(
                packed_v, repeats, axis=1)
        scores = (queries @ packed_k.transpose(0, 1, 3, 2)) * scale
        scores = scores + mx.where(allowed[:, None], 0.0, -float("inf"))
        return mx.softmax(scores, axis=-1, precise=True) @ packed_v


class _ConvRows:
    def __init__(self, rows, widths):
        self.rows, self.widths = rows, widths
        self.offset = max(c.offset for c in rows)

    def update(self, values):
        import mlx.core as mx

        carry_width = self.rows[0].kernel_size - 1
        if any(c.kernel_size != carry_width + 1 for c in self.rows):
            raise ValueError("incompatible convolution carry layouts")
        prefix = mx.zeros((len(self.rows), carry_width, values.shape[-1]), dtype=values.dtype)
        for i, cache in enumerate(self.rows):
            if cache.values is not None:
                prefix[i:i + 1] = cache.values
        combined = mx.concatenate((prefix, values), axis=1)
        for i, (cache, n) in enumerate(zip(self.rows, self.widths, strict=True)):
            carry = mx.zeros((1, carry_width, values.shape[-1]), dtype=values.dtype)
            carry[...] = combined[i:i + 1, n:n + carry_width]
            cache.values, cache._offset = carry, cache.offset + n
        return combined


class HybridRunner(BatchedDenseRunner):
    def capabilities(self):
        return replace(super().capabilities(), family="lfm2",
                       supported_cache_backends=("hybrid",),
                       supported_state_kinds=("full_kv", "sliding_kv", "conv"),
                       supports_exact_prefix_restore=True)

    def make_cache(self):
        from mlx_one.engine.components import hybrid_components

        return CacheBundle(hybrid_components(self.bundle))

    def _resolve(self, plan, phase):
        if plan.execution_mode not in {"dense", "hybrid"}:
            raise ValueError("incompatible hybrid execution plan")
        return super()._resolve(replace(plan, execution_mode="dense"), phase)

    def _validate_bundle(self, bundle):
        from mlx_one.engine.cache_planner import build_cache_plan
        from mlx_one.engine.modes import validate_mode

        validate_mode(bundle, build_cache_plan(bundle), "experimental_hybrid_batch")

    def execute(self, requests):
        import mlx.core as mx

        if not requests:
            raise ValueError("empty hybrid batch")
        first = requests[0]
        self._validate_bundle(first.bundle)
        if any(not r.cache for r in requests):
            raise ValueError("empty hybrid layer mapping")
        widths = tuple(len(r.token_ids) for r in requests)
        slots = tuple(ScheduledSequence(r.request_id, r.phase, r.cache[0].offset, len(r.token_ids))
                      for r in requests)
        plan = BatchPlan(slots, sum(widths), execution_mode="hybrid")
        for r in requests:
            if (r.bundle is not first.bundle or r.phase != first.phase
                    or r.model_epoch != first.model_epoch or len(r.cache) != len(first.cache)
                    or len({c.offset for c in r.cache}) != 1):
                raise ValueError("incompatible hybrid mappings")
            kinds = {"full_attention": "full_kv", "sliding_attention": "sliding_kv", "conv": "conv"}
            expected = tuple(kinds[k] for k in first.bundle.model.config.layer_types)
            if tuple(getattr(c, "kind", None) for c in r.cache) != expected:
                raise ValueError("hybrid component mapping differs from model topology")
            from mlx_one.engine.cache_planner import build_cache_plan

            for component, spec in zip(
                    r.cache, build_cache_plan(r.bundle).state_specs, strict=True):
                component.validate(spec)
            if r.handle is not None:
                r.runtime._validate_handle(r.handle)
                if r.handle.bundle.entries is not r.cache:
                    raise ValueError("hybrid request does not own its cache mapping")
        config = first.bundle.model.config
        heads, dim = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
        width = max(widths)
        history = max(c.offset for r in requests for c in r.cache)
        transient = len(requests) * len(first.cache) * heads * dim * (history + width) * 16
        transient += len(requests) * config.hidden_size * config.conv_L_cache * 8
        from mlx_one.engine.pressure import execution_forecast

        transient = execution_forecast(first.runtime, config, batch=len(requests), chunk=width,
                                         history=history, explicit_bytes=transient,
                                         history_lengths=[r.cache[0].offset for r in requests],
                                         widths=widths, phase=first.phase)["bytes"]
        started, drafts = time.perf_counter(), []
        runtime = first.runtime
        if any(r.runtime is not runtime for r in requests):
            raise ValueError("hybrid batch spans cache managers")
        with ExitStack() as stack:
            if runtime is not None:
                stack.enter_context(runtime.transient_reservation(transient))
            try:
                for r in requests:
                    drafts.append(CacheBundle(r.cache).clone())
                packed = []
                for layer in range(len(first.cache)):
                    rows = tuple(d.entries[layer] for d in drafts)
                    wrapper = (_ConvRows if isinstance(rows[0], ConvCarryComponent)
                               else _AttentionRows)
                    packed.append(wrapper(rows, widths))
                ids = mx.array([list(r.token_ids) + [0] * (width - len(r.token_ids))
                                for r in requests])
                mask = mx.array([[1] * n + [0] * (width - n) for n in widths])
                positions = mx.array([list(range(r.cache[0].offset, r.cache[0].offset + width))
                                      for r in requests])
                output = first.bundle.model(ids, cache=tuple(packed), attention_mask=mask,
                                            position_ids=positions)
                logits = []
                for i, n in enumerate(widths):
                    begin = 0 if requests[i].return_all_logits else n - 1
                    compact = mx.zeros((1, n - begin, output.logits.shape[-1]),
                                    dtype=output.logits.dtype)
                    compact[...] = output.logits[i:i + 1, begin:n]
                    logits.append(compact)
                states = [v for d in drafts for c in d.entries
                          for v in (getattr(c, "keys", None), getattr(c, "values", None))
                          if v is not None]
                mx.eval(*logits, *states)
                for r, draft in zip(requests, drafts, strict=True):
                    if request_cancelled(r):
                        continue
                    for original, replacement in zip(r.cache, draft.entries, strict=True):
                        original.__dict__.update(replacement.__dict__)
                    if r.handle is not None:
                        r.handle.computed_tokens = r.cache[0].offset
                        runtime.reconcile(r.handle)
                # Prefix retention is optional and occurs after the entire batch is committed.
                if first.phase == "prefill" and runtime is not None:
                    for r in requests:
                        cancelled = request_cancelled(r)
                        if r.handle is not None and not cancelled:
                            try:
                                runtime.publish_hybrid_boundary(r.handle, r.prompt_ids)
                            except MemoryError:
                                pass
            finally:
                # Committed arrays are owned by originals; reset only draft attributes.
                for draft in drafts:
                    draft.release()
        self.last_trace = {"phase": first.phase,
                           "request_ids": [s.request_id for s in plan.sequences],
                           "batch_size": len(requests),
                           "history_lengths": [s.start_pos for s in slots],
                           "real_tokens": sum(widths), "padded_tokens": len(requests) * width,
                           "padding_efficiency": sum(widths) / (len(requests) * width),
                           "transient_reserved_bytes": transient,
                           "component_kinds": [getattr(c, "kind", "unknown") for c in first.cache],
                           "duration_ms": (time.perf_counter() - started) * 1000}
        return RunnerOutput(tuple(r.request_id for r in requests), tuple(logits), widths)
