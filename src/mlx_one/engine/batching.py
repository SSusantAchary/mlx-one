"""Dense batching contracts and packing, independent of HTTP and scheduling threads."""

from __future__ import annotations

import copy
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

from mlx_one.core.cache import KVCache, QuantizedKVCache
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.contracts import (
    BatchPlan,
    Phase,
    RunnerCapabilities,
    RunnerOutput,
    ScheduledSequence,
)
from mlx_one.engine.runners import CausalLMRunner
from mlx_one.engine.transactions import commit_evaluated, release_all


@dataclass(frozen=True)
class BatchingConfig:
    max_sequences: int = 1
    queue_size: int = 8
    timeout: float = 600.0
    prefill_chunk_size: int = 512
    max_scheduled_tokens: int = 2048
    max_prefill_tokens: int = 1024
    output_buffer_events: int = 32
    output_stall_timeout_s: float = 30.0
    admission_age_s: float = 2.0
    admission_policy: str = "deadline"

    def __post_init__(self):
        for name in ("max_sequences", "queue_size", "prefill_chunk_size", "max_scheduled_tokens",
                     "max_prefill_tokens", "output_buffer_events"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.timeout < 0 or self.output_stall_timeout_s <= 0 or self.admission_age_s < 0:
            raise ValueError("invalid batching timeout or admission age")
        if self.admission_policy not in {"deadline", "cache_aware"}:
            raise ValueError("unsupported admission policy")


@dataclass
class ForwardRequest:
    bundle: Any
    token_ids: tuple[int, ...]
    cache: tuple[Any, ...]
    phase: str
    prompt_ids: tuple[int, ...]
    runtime: Any = None
    handle: Any = None
    request_id: str = ""
    model_epoch: int = 0
    cancel: Any = None
    options: Any = None
    deadline: float | None = None
    return_all_logits: bool = False
    staged_verification: bool = False
    generated_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GenerationProgress:
    generated_ids: tuple[int, ...]
    computed_len: int


def request_cancelled(request):
    expired = request.deadline is not None and time.monotonic() >= request.deadline
    if expired and request.cancel is not None:
        request.cancel.set()
    return expired or (request.cancel is not None and request.cancel.is_set())


class BatchScheduler:
    """Combined token budget, decode-first service, rotating prefill/decode fairness."""

    def __init__(self, config: BatchingConfig):
        self.config = config
        self._decode_cursor = self._prefill_cursor = 0
        self._single_budget_turn = 0

    def select(self, requests: list[ForwardRequest]) -> tuple[tuple[ForwardRequest, ...], ...]:
        cfg = self.config
        decodes = [r for r in requests if r.phase == "decode"]
        prefills = [r for r in requests if r.phase == "prefill"]
        budget = cfg.max_scheduled_tokens
        quantum = min(cfg.prefill_chunk_size, cfg.max_prefill_tokens, max(budget // 2, 1))
        if any(len(request.token_ids) > quantum for request in prefills):
            raise ValueError("prefill request exceeds the configured service quantum")
        reserved = min(quantum, min((len(r.token_ids) for r in prefills), default=0))
        if budget == 1 and decodes and prefills:
            self._single_budget_turn += 1
            reserved = self._single_budget_turn % 2
        if decodes:
            index = self._decode_cursor % len(decodes)
            decodes = decodes[index:] + decodes[:index]
        selected_decode = decodes[:max(budget - reserved, 0)]
        self._decode_cursor += len(selected_decode)
        budget -= len(selected_decode)
        if prefills:
            index = self._prefill_cursor % len(prefills)
            prefills = prefills[index:] + prefills[:index]
        selected_prefill = []
        prefill_budget = min(budget, cfg.max_prefill_tokens)
        for request in prefills:
            if len(request.token_ids) <= prefill_budget:
                selected_prefill.append(request)
                prefill_budget -= len(request.token_ids)
        self._prefill_cursor += max(len(selected_prefill), 1)
        return (*execution_groups(selected_decode),*execution_groups(selected_prefill))


def execution_groups(requests):
    """Partition state transition buckets without changing request ownership."""
    buckets = {}
    for request in requests:
        key = tuple(component.batch_key(len(request.token_ids))
                    if callable(getattr(component,"batch_key",None)) else None
                    for component in getattr(request,"cache",()))
        buckets.setdefault(key,[]).append(request)
    return tuple(tuple(rows) for rows in buckets.values())


def execute_compatible(runner, requests):
    """Qualification helpers use the same state buckets as actor scheduling."""
    if not requests or len({r.request_id for r in requests}) != len(requests):
        raise ValueError("execution buckets require unique nonempty request mappings")
    outputs = {}
    for group in execution_groups(requests):
        result=runner.execute(group)
        if result.request_ids != tuple(r.request_id for r in group):
            raise ValueError("execution bucket result mapping mismatch")
        for request_id,logits,count in zip(result.request_ids,result.logits,
                result.committed_token_counts,strict=True):
            if request_id in outputs:
                raise ValueError("duplicate execution bucket request")
            outputs[request_id]=(logits,count)
    return RunnerOutput(tuple(r.request_id for r in requests),
        tuple(outputs[r.request_id][0] for r in requests),
        tuple(outputs[r.request_id][1] for r in requests))


class _PackedCache:
    """Exact temporary buffers; offset is physical padded history, not model position."""

    def __init__(self, keys, values, offset, *, lengths=(), widths=(), compact_attention=False):
        self.keys, self.values, self.offset = keys, values, offset
        self.lengths,self.widths=lengths,widths
        if compact_attention:
            self.attend=self._attend_compact

    def _attend_compact(self,queries,keys,values,*,scale,position_ids):
        """Keep each INT8 row's reduction width identical to independent execution.

        QKV/MLP still execute in one batched model call. Only attention reductions
        exclude physical padding, whose zeros otherwise change FP32 reduction trees.
        """
        import mlx.core as mx

        from mlx_one.models.shared.execution import attention_matmul

        history=self.offset
        self.update(keys,values)
        output=mx.zeros(queries.shape,dtype=queries.dtype)
        repeats=queries.shape[1]//keys.shape[1]
        for row,(old,width) in enumerate(zip(self.lengths,self.widths,strict=True)):
            begin,end=history-old,history+width
            k,v=(mx.repeat(t[row:row+1,:,begin:end],repeats,axis=1)
                 for t in (self.keys,self.values))
            q=queries[row:row+1,:,:width]
            scores=attention_matmul(q,k.transpose(0,1,3,2))*scale
            allowed=mx.arange(old+width)[None,:] <= old+mx.arange(width)[:,None]
            scores=scores+mx.where(allowed,0.0,-float("inf"))
            probabilities=mx.softmax(scores,axis=-1,precise=True)
            output[row:row+1,:,:width]=attention_matmul(probabilities,v).astype(queries.dtype)
        return output

    def update(self, keys, values):
        import mlx.core as mx

        stop = self.offset + int(keys.shape[2])
        self.keys[:, :, self.offset:stop, :] = keys.astype(mx.float16)
        self.values[:, :, self.offset:stop, :] = values.astype(mx.float16)
        self.offset = stop
        return self.keys[:, :, :stop, :], self.values[:, :, :stop, :]


def _state(cache):
    if isinstance(cache, QuantizedKVCache):
        if (cache.key_bits is not None or cache.value_bits is not None
                or cache.key_dtype != "f16" or cache.value_dtype != "f16"):
            raise ValueError("batched dense execution requires unquantized KV")
        return cache._keys, cache._values
    if isinstance(cache, KVCache):
        return cache.keys, cache.values
    raise ValueError("unsupported dense cache layout")


class BatchedDenseRunner(CausalLMRunner):
    """One evaluated forward for ragged dense requests, with atomic cache scatter."""

    def __init__(self, bundle=None):
        self.bundle = bundle
        self._sequences = {}
        self._validated_bundle = None
        if bundle is not None:
            self._validate_bundle(bundle)

    def _validate_bundle(self, bundle):
        if self._validated_bundle is bundle:
            return
        from mlx_one.engine.cache_planner import build_cache_plan
        from mlx_one.engine.modes import validate_mode

        validate_mode(bundle, build_cache_plan(bundle), "experimental_dense_batch")
        from mlx_one.engine.adapters import native_adapter

        self.adapter = native_adapter(bundle)
        self._validated_bundle = bundle

    def make_cache(self):
        from mlx_one.core.cache import configure_kv_caches
        from mlx_one.engine.cache import CacheBundle

        if self.bundle is None:
            raise ValueError("contract preparation requires a loaded bundle")
        return CacheBundle(configure_kv_caches(self.bundle.model.make_cache(), "f16", "f16"))

    def _execute(self, plan, phase):
        states = self._resolve(plan, phase)
        requests = tuple(ForwardRequest(
            self.bundle,
            (state.input_ids[slot.start_pos:slot.start_pos + slot.input_token_count]
             if phase == "prefill" else (state.generated_ids[-1],)),
            state.cache.entries, phase, state.input_ids, request_id=state.request_id,
            model_epoch=state.model_epoch, cancel=state.cancel)
            for state, slot in zip(states, plan.sequences, strict=True))
        try:
            result = self.execute(requests)
            for state, amount in zip(states, result.committed_token_counts, strict=True):
                if not state.cancel.is_set():
                    state.commit(amount)
            return result
        except BaseException as exc:
            for state in states:
                state.error = exc
                state.transition(Phase.FAILED)
                self.finalize(state)
            raise

    def capabilities(self):
        from mlx_one.engine.contracts import RequestKind

        return RunnerCapabilities(frozenset({RequestKind.TEXT}), batchable=True,
                                  chunked_prefill=True, family=(self.adapter.spec.architecture
                                      if hasattr(self, "adapter") else "native"),
                                  supported_state_kinds=("full_kv",),
                                  supports_batched_decode=True, supports_batched_prefill=True)

    def execute(self, requests: tuple[ForwardRequest, ...]) -> RunnerOutput:
        import mlx.core as mx

        if not requests:
            raise ValueError("cannot execute an empty batch")
        first = requests[0]
        self._validate_bundle(first.bundle)
        history = max(int(r.cache[0].offset) for r in requests)
        width = max(len(r.token_ids) for r in requests)
        count = len(requests)
        slots = []
        for request in requests:
            if request.bundle is not first.bundle or request.phase != first.phase:
                raise ValueError("batch contains incompatible models or phases")
            if request.model_epoch != first.model_epoch or len(request.cache) != len(first.cache):
                raise ValueError("batch contains incompatible epochs or layers")
            offsets = {int(c.offset) for c in request.cache}
            if len(offsets) != 1:
                raise ValueError("layer cache offsets differ")
            for cache in request.cache:
                keys, values = _state(cache)
                if keys is not None and (int(keys.shape[0]) != 1 or keys.dtype != mx.float16
                                         or values.dtype != mx.float16):
                    raise ValueError("batched dense cache must be B=1 FP16")
            slots.append(ScheduledSequence(request.request_id, request.phase,
                                            request.cache[0].offset, len(request.token_ids)))
        plan = BatchPlan(tuple(slots), sum(len(r.token_ids) for r in requests))
        config = self.adapter.activation_config
        specs = self.adapter.plan.layer_specs
        dim = self.adapter.spec.head_dim
        per_token = self.adapter.plan.bytes_per_token()
        replacement_tokens = sum(((r.cache[0].offset + len(r.token_ids) + 255) // 256) * 256
                                 for r in requests)
        transient = per_token * (count * (history + width) + replacement_tokens)
        arithmetic_scratch = 0
        if (getattr(first.bundle, "quantization", {}).get("bits") == 8
                and self.adapter.activation_dtype == "float16"):
            from mlx_one.models.shared.execution import attention_scratch_bound

            arithmetic_scratch = attention_scratch_bound(batch=count,
                heads=self.adapter.spec.query_heads, head_dim=dim, chunk=width, history=history)
            transient += arithmetic_scratch
        adapter_scratch = self.adapter.additional_transient_bytes(batch=count, chunk=width)
        transient += adapter_scratch
        packed = []
        staged = []
        started = time.perf_counter()
        with ExitStack() as stack:
            runtimes = {id(r.runtime): r.runtime for r in requests if r.runtime is not None}
            if len(runtimes) > 1:
                raise ValueError("batch spans cache managers")
            for runtime in runtimes.values():
                from mlx_one.engine.pressure import execution_forecast

                transient = execution_forecast(runtime, config, batch=count, chunk=width,
                    history=history, explicit_bytes=transient,
                    history_lengths=[r.cache[0].offset for r in requests],
                    widths=[len(r.token_ids) for r in requests], phase=first.phase)["bytes"]
                stack.enter_context(runtime.transient_reservation(transient))
            ids = [[*r.token_ids, *([0] * (width - len(r.token_ids)))] for r in requests]
            positions = [[r.cache[0].offset + i if i < len(r.token_ids) else 0
                          for i in range(width)] for r in requests]
            mask = [[0] * (history - r.cache[0].offset) + [1] * r.cache[0].offset
                    + [1] * len(r.token_ids) + [0] * (width - len(r.token_ids)) for r in requests]
            for layer in range(len(first.cache)):
                spec = specs[layer]
                key_shape = (count, spec.num_kv_heads, history + width, spec.key_head_dim)
                value_shape = (count, spec.num_kv_heads, history + width,
                               spec.value_head_dim or spec.key_head_dim)
                keys = mx.zeros(key_shape, dtype=mx.float16)
                values = mx.zeros(value_shape, dtype=mx.float16)
                for row, request in enumerate(requests):
                    old_keys, old_values = _state(request.cache[layer])
                    length = request.cache[layer].offset
                    if length:
                        keys[row:row + 1, :, history - length:history, :] = (
                            old_keys[:, :, :length, :])
                        values[row:row + 1, :, history - length:history, :] = (
                            old_values[:, :, :length, :])
                packed.append(_PackedCache(keys, values, history,
                    lengths=tuple(r.cache[0].offset for r in requests),
                    widths=tuple(len(r.token_ids) for r in requests),
                    compact_attention=bool(arithmetic_scratch)
                        and self.adapter.spec.architecture in {"qwen2","qwen3","llama"}))
            output = self.adapter.forward(mx.array(ids), cache=tuple(packed),
                                        attention_mask=mx.array(mask),
                                        position_ids=mx.array(positions))
            mx.eval(output.logits)
            # Every scatter is evaluated before changing any original layer.
            try:
                for row, request in enumerate(requests):
                    layers = []
                    staged.append(layers)
                    length = request.cache[0].offset
                    real = length + len(request.token_ids)
                    for original, cache in zip(request.cache, packed, strict=True):
                        replacement = copy.copy(original)
                        replacement.reset()
                        layers.append(replacement)
                        start, stop = history - length, history + len(request.token_ids)
                        selected_keys = cache.keys[row:row + 1, :, start:stop, :]
                        selected_values = cache.values[row:row + 1, :, start:stop, :]
                        new_keys, new_values = replacement.update(selected_keys, selected_values)
                        if replacement.offset != real:
                            raise ValueError("scatter boundary mismatch")
                        mx.eval(new_keys, new_values)
                compact_logits = []
                for row, request in enumerate(requests):
                    last = len(request.token_ids)
                    begin=0 if request.return_all_logits else last-1
                    logits = mx.zeros((1,last-begin,output.logits.shape[-1]),
                                      dtype=output.logits.dtype)
                    logits[...] = output.logits[row:row+1,begin:last]
                    mx.eval(logits)
                    compact_logits.append(logits)
            except BaseException:
                release_all(tuple(CacheBundle(tuple(layers)) for layers in staged))
                raise

            drafts = tuple(CacheBundle(tuple(layers)) for layers in staged)
            try:
                commit_evaluated(requests, drafts, specs=self.adapter.plan.state_specs)
            finally:
                release_all(drafts)
        self.last_trace = {"phase": first.phase, "request_ids": [r.request_id for r in requests],
                           "batch_size": count,
                           "history_lengths": [s.start_pos for s in plan.sequences],
                           "real_tokens": plan.total_scheduled_tokens,
                           "padded_tokens": count * width,
                           "padding_efficiency": plan.total_scheduled_tokens / (count * width),
                           "transient_reserved_bytes": transient,
                           "arithmetic_scratch_bound_bytes": arithmetic_scratch,
                           "attention_reduction": "compact-real-row" if any(
                               callable(getattr(c,"attend",None)) for c in packed)
                               else "padded-batch",
                           "adapter_transient_bound_bytes": adapter_scratch,
                           "packed_history_bytes": per_token * count * history,
                           "replacement_cache_bytes": per_token * replacement_tokens,
                           "adapter_version": self.adapter.spec.adapter_version,
                           "duration_ms": (time.perf_counter() - started) * 1000}
        return RunnerOutput(tuple(r.request_id for r in requests),
                            tuple(compact_logits),
                            tuple(len(r.token_ids) for r in requests))
