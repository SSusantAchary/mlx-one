"""OpenAI request adaptation onto mlx-one native generation."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from mlx_one.engine.cache_errors import PrefixReuseUnsupported
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig, CacheHandle
from mlx_one.engine.modes import (
    InferenceMode,
    configure_execution_dtype,
    resolve_backend,
)
from mlx_one.engine.reporting import native_report
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import current_queue_ms
from mlx_one.text import TextGenerationOptions, stream_chat
from mlx_one.utils.memory import get_memory_snapshot


@dataclass(frozen=True)
class GenerationEvent:
    text: str = ""
    reasoning_content: str = ""
    finish_reason: str | None = None
    metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeStats:
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None
    generation_latency_ms: float | None = None
    context_used: int | None = None
    memory_used_bytes: int | None = None
    memory_peak_bytes: int | None = None
    reasoning_tokens: int | None = None
    prompt_cache_hit: bool | None = None
    reused_prompt_tokens: int | None = None
    drafted_tokens: int | None = None
    accepted_draft_tokens: int | None = None
    rejected_draft_tokens: int | None = None
    draft_acceptance_ratio: float | None = None
    prompt_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    inter_token_latency_ms: float | None = None
    cache_bytes: int | None = None
    cache_backend: str | None = None
    cache_logical_bytes: int | None = None
    cache_allocated_bytes: int | None = None
    cache_reserved_bytes: int | None = None
    cache_shared_blocks: int | None = None
    cache_lookup_time_ms: float | None = None
    cache_allocation_time_ms: float | None = None
    avoided_prefill_tokens: int | None = None
    server_text_ttft_ms: float | None = None
    model_token_count: int | None = None
    model_token_gap_p50_ms: float | None = None
    model_token_gap_p95_ms: float | None = None
    timing_samples_truncated: bool = False
    actual_prefill_tokens: int | None = None
    eligible_prefix_tokens: int | None = None
    matched_prefix_tokens: int | None = None
    prefix_publications: int | None = None
    publication_rejections: int | None = None
    queue_ms: float | None = None
    tokenization_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    model_token_times_ms: tuple[float, ...] | None = None
    model_generation_started_at: float | None = None
    model_forward_batch_sizes: tuple[int, ...] | None = None
    execution_observer: str | None = None
    mlx_cached_bytes: int | None = None
    actual_logits_dtypes: tuple[str, ...] | None = None
    cache_layers: tuple[dict[str, Any], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationEngine:
    def __init__(
        self,
        manager: ModelManager,
        *,
        prefix_cache_bytes: int = 512 * 1024**2,
        cache_backend: str | None = None,
        inference_mode: InferenceMode | None = None,
        cache_block_size: int = 32,
        cache_memory_budget_bytes: int | None = None,
        collect_execution_metrics: bool = False,
        execution_dtype: str | None = None,
        prefill_chunk_size: int = 512,
        memory_pressure_config: Any = None,
        qualification_registry: Any = None,
        qualification_profile: dict[str, Any] | None = None,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
    ) -> None:
        self.manager = manager
        self._inference_mode = inference_mode
        self._requested_mode = inference_mode
        self._qualification_registry = qualification_registry
        self._qualification_profile = qualification_profile
        self._selection = {"requested_mode": inference_mode, "resolved_mode": inference_mode,
                           "reason": "explicit configuration", "evidence_scope": None}
        self._pressure = None
        if memory_pressure_config is not None or inference_mode in {
            "paged", "experimental_hybrid_batch", "auto"
        }:
            from mlx_one.engine.pressure import MemoryPressureController

            self._pressure = MemoryPressureController(memory_pressure_config)
        self._execution_dtype = execution_dtype
        self._prefill_chunk_size = prefill_chunk_size
        self._converted_bundle = None
        self._actor = None
        self._collect_execution_metrics = collect_execution_metrics
        self._stats = RuntimeStats()
        self._lock = threading.Lock()
        self._cache_config = CacheConfig(
            backend=resolve_backend(inference_mode, cache_backend),  # type: ignore[arg-type]
            memory_budget_bytes=cache_memory_budget_bytes,
            prefix_cache_budget_bytes=prefix_cache_bytes,
            block_size_tokens=cache_block_size,
            key_type=cache_type_k, value_type=cache_type_v,
        )
        self._cache_runtime: CacheManager | None = None
        self._cache_bundle: Any | None = None

    @property
    def stats(self) -> RuntimeStats:
        with self._lock:
            return self._stats

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: TextGenerationOptions,
        cancel: threading.Event,
        context_length: int | None = None,
        context_shift: bool = False,
        reasoning: str = "auto",
        reasoning_budget: int = -1,
        reasoning_format: str = "deepseek",
        reasoning_preserve: bool = False,
        cache_prompt: bool = False,
        cache_reuse: int = 256,
        cache_entries: int = 1,
        cache_idle_slots: bool = False,
        cache_namespace: str | None = None,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        spec_type: str = "none",
        spec_draft_n_max: int = 3,
    ) -> Iterator[GenerationEvent]:
        lifecycle: dict[str, Any] = {}
        acquire = getattr(self.manager, "model_lease", None)
        lease = acquire() if callable(acquire) else None
        try:
            yield from self._stream_request(
                model=model,
                messages=messages,
                options=options,
                cancel=cancel,
                context_length=context_length,
                context_shift=context_shift,
                reasoning=reasoning,
                reasoning_budget=reasoning_budget,
                reasoning_format=reasoning_format,
                reasoning_preserve=reasoning_preserve,
                cache_prompt=cache_prompt,
                cache_reuse=cache_reuse,
                cache_entries=cache_entries,
                cache_idle_slots=cache_idle_slots,
                cache_namespace=cache_namespace,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                spec_type=spec_type,
                spec_draft_n_max=spec_draft_n_max,
                lifecycle=lifecycle,
            )
        finally:
            runtime = lifecycle.get("runtime")
            handle = lifecycle.get("handle")
            try:
                if (isinstance(runtime, CacheManager) and isinstance(handle, CacheHandle)
                        and not handle.closed):
                    runtime.release(handle)
            finally:
                if lease is not None:
                    lease.release()

    def _stream_request(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: TextGenerationOptions,
        cancel: threading.Event,
        context_length: int | None = None,
        context_shift: bool = False,
        reasoning: str = "auto",
        reasoning_budget: int = -1,
        reasoning_format: str = "deepseek",
        reasoning_preserve: bool = False,
        cache_prompt: bool = False,
        cache_reuse: int = 256,
        cache_entries: int = 1,
        cache_idle_slots: bool = False,
        cache_namespace: str | None = None,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        spec_type: str = "none",
        spec_draft_n_max: int = 3,
        lifecycle: dict[str, Any],
    ) -> Iterator[GenerationEvent]:
        queue_ms = current_queue_ms()
        bundle = self.manager.current_model()
        if model != self.manager.model_info().id:
            raise ValueError(f"model {model!r} is not loaded")
        if bundle.chat_template is None:
            raise RuntimeError("loaded model has no chat template")
        if reasoning == "on" and "enable_thinking" not in (bundle.chat_template.template or ""):
            raise ValueError("loaded chat template does not support explicit reasoning control")
        from mlx_one.engine.prompts import prepare_chat

        tokenization_started = time.perf_counter()
        prepared = prepare_chat(
            bundle, messages, options.max_tokens, context_length or bundle.context_length,
            context_shift=context_shift, rendering_options=_reasoning_template_options(
                reasoning, reasoning_budget, reasoning_preserve))
        rendered_messages, prompt = list(prepared.messages), prepared.text
        prompt_ids = prepared.token_ids
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        prompt_tokens = len(prompt_ids)
        runtime = self._cache_manager_for(bundle)
        if runtime.config.backend in {"paged", "hybrid"} or self._requested_mode == "auto":
            from mlx_one.engine.selection import validate_context_admission

            validate_context_admission(runtime, prompt_tokens, options.max_tokens)
        handle = runtime.create_handle(
            f"server-{time.time_ns()}",
            key_type=cache_type_k,
            value_type=cache_type_v,
        )
        handle.metadata["is_cancelled"] = cancel.is_set
        handle.metadata["prepared_prompt"] = prepared
        handle.metadata["semantic_boundary"] = (
            prepared.semantic_boundary if cache_prompt and runtime.config.backend == "dense"
            and runtime.plan.prefix_shareable else 0)
        handle.metadata["maximum_prefix_entries"] = cache_entries
        lifecycle.update({"runtime": runtime, "handle": handle})
        cached_ids: tuple[int, ...] = ()
        fingerprint = runtime.fingerprint(
            key_type=cache_type_k,
            value_type=cache_type_v,
            context_tokens=context_length or bundle.context_length,
            namespace=cache_namespace,
            rendering_options=_reasoning_template_options(
                reasoning, reasoning_budget, reasoning_preserve),
        )
        if cache_prompt:
            try:
                reused = runtime.adopt_prefix(
                    handle, fingerprint, prompt_ids, minimum_tokens=cache_reuse
                )
            except PrefixReuseUnsupported:
                reused = 0
            cached_ids = prompt_ids[:reused]
        requested = runtime.estimate_growth(
            handle,
            prompt_tokens=len(prompt_ids),
            max_new_tokens=options.max_tokens,
        )
        runtime.reserve(
            handle,
            tokens=max(len(prompt_ids) - len(cached_ids), 0) + options.max_tokens,
            requested_bytes=requested,
        )
        started = time.perf_counter()
        first_at: float | None = None
        generated = 0
        finish_reason = "length"
        parser = _ReasoningParser(
            reasoning_format,
            reasoning_preserve,
            reasoning_budget,
            start_in_reasoning=prompt.rstrip().endswith("<think>"),
        )
        speculative_stats = {"drafted_tokens": 0, "accepted_draft_tokens": 0}
        from mlx_one.engine.timing import RequestTiming

        timings = RequestTiming(started, trace=self._collect_execution_metrics)
        iterator = stream_chat(
            bundle,
            rendered_messages,
            options=options,
            is_cancelled=cancel.is_set,
            template_options=_reasoning_template_options(
                reasoning, reasoning_budget, reasoning_preserve
            ),
            context_length=context_length,
            cache=handle.bundle.entries,
            cached_prompt_ids=cached_ids,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            on_cache_update=(
                lambda tokens, caches: runtime.publish_prefix(
                    handle,
                    fingerprint,
                    tuple(tokens),
                    maximum_entries=cache_entries,
                )
                if cache_prompt
                else None
            ),
            spec_type=spec_type,
            spec_draft_n_max=spec_draft_n_max,
            speculative_stats=speculative_stats,
            reasoning_budget=reasoning_budget,
            yield_steps=True,
            on_execution=timings.observe,
            _cooperative=self._actor is not None,
            _prefill_chunk_size=self._prefill_chunk_size,
            _cache_runtime=runtime, _cache_handle=handle,
        )
        from mlx_one.engine.batching import ForwardRequest

        try:
            chunk = next(iterator)
            while True:
                if isinstance(chunk, ForwardRequest):
                    response = yield chunk
                    chunk = iterator.send(response)
                    continue
                if chunk.token_id is not None:
                    generated = chunk.generated_tokens
                    if first_at is None:
                        first_at = time.perf_counter()
                    text, thought = parser.feed(chunk.text) if chunk.text else ("", "")
                    yield GenerationEvent(text=text, reasoning_content=thought)
                elif chunk.finish_reason is None:
                    yield GenerationEvent()
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                chunk = next(iterator)
        except StopIteration:
            pass
        finally:
            iterator.close()
        trailing_text, trailing_reasoning = parser.flush()
        if trailing_text or trailing_reasoning:
            yield GenerationEvent(
                text=trailing_text, reasoning_content=trailing_reasoning
            )
        ended = time.perf_counter()
        elapsed = max(ended - started, 0.0)
        memory_used: int | None = None
        memory_peak: int | None = None
        try:
            memory = get_memory_snapshot()
            memory_used, memory_peak = memory.active_bytes, memory.peak_bytes
        except Exception:
            pass
        cache_snapshot = runtime.stats()
        token_times = tuple(timings.tokens) if timings.tokens is not None else None
        prefill_ms, decode_ms = timings.forward_ms["prefill"], timings.forward_ms["decode"]
        observed_tokenization = timings.tokenization_ms
        mx = sys.modules.get("mlx.core")
        cached_memory = getattr(mx, "get_cache_memory", None)
        from mlx_one.engine.reporting import describe_cache

        stats = RuntimeStats(
            queue_ms=queue_ms,
            tokenization_ms=tokenization_ms + observed_tokenization,
            prefill_ms=prefill_ms if timings.forwards else None,
            decode_ms=decode_ms if timings.forwards else None,
            model_token_times_ms=token_times,
            model_generation_started_at=started if token_times is not None else None,
            model_forward_batch_sizes=(tuple(timings.batches)
                                       if timings.batches is not None else None),
            execution_observer="synchronized-forward-v2" if timings.forwards else None,
            actual_logits_dtypes=tuple(sorted(timings.dtypes)) if timings.forwards else None,
            actual_prefill_tokens=timings.prefill_tokens if timings.forwards else None,
            eligible_prefix_tokens=prepared.semantic_boundary,
            matched_prefix_tokens=len(cached_ids),
            prefix_publications=int(handle.metadata.get("prefix_publications", 0)),
            publication_rejections=int(handle.metadata.get("publication_rejections", 0)),
            **timings.summary(),
            server_text_ttft_ms=(first_at - started) * 1000 if first_at is not None else None,
            mlx_cached_bytes=int(cached_memory()) if callable(cached_memory) else None,
            cache_layers=tuple(describe_cache(handle.bundle.entries, runtime.plan)),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            tokens_per_second=generated / elapsed if elapsed > 0 else None,
            generation_latency_ms=elapsed * 1000,
            context_used=prompt_tokens + generated,
            memory_used_bytes=memory_used,
            memory_peak_bytes=memory_peak,
            reasoning_tokens=parser.reasoning_tokens,
            prompt_cache_hit=bool(cached_ids),
            reused_prompt_tokens=len(cached_ids),
            drafted_tokens=speculative_stats["drafted_tokens"],
            accepted_draft_tokens=speculative_stats["accepted_draft_tokens"],
            rejected_draft_tokens=max(
                speculative_stats["drafted_tokens"]
                - speculative_stats["accepted_draft_tokens"],
                0,
            ),
            draft_acceptance_ratio=(
                speculative_stats["accepted_draft_tokens"]
                / speculative_stats["drafted_tokens"]
                if speculative_stats["drafted_tokens"]
                else None
            ),
            prompt_tokens_per_second=(
                (prompt_tokens - len(cached_ids)) * 1000 / prefill_ms if prefill_ms else None
            ),
            cache_bytes=cache_snapshot.physical_bytes,
            cache_backend=handle.backend,
            cache_logical_bytes=handle.bundle.nbytes,
            cache_allocated_bytes=handle.bundle.allocated_nbytes,
            cache_reserved_bytes=handle.reserved_bytes,
            cache_shared_blocks=int(handle.metadata.get("shared_blocks", 0)),
            cache_lookup_time_ms=float(
                handle.metadata.get("prefix_lookup_time_ms", 0.0)
            ),
            cache_allocation_time_ms=float(
                handle.metadata.get("allocation_time_ms", 0.0)
            ),
            avoided_prefill_tokens=len(cached_ids),
        )
        if handle.reservation is not None:
            runtime.commit(handle.reservation, min(handle.bundle.allocated_nbytes, requested))
        with self._lock:
            self._stats = stats
        sink = getattr(self, "_qualification_token_sink", None)
        if sink is not None and timings.token_ids is not None:
            sink(handle.request_id, tuple(timings.token_ids))
        yield GenerationEvent(finish_reason=finish_reason, metrics=stats.to_dict())

    def clear_caches(self) -> None:
        if self._cache_runtime is not None:
            self._cache_runtime.clear_prefixes()

    def close(self) -> None:
        if self._pressure is not None:
            self._pressure.close()
        self._converted_bundle = None
        if self._cache_runtime is not None:
            self._cache_runtime.close()
            self._cache_runtime = None
            self._cache_bundle = None

    def runtime_stats(self) -> dict[str, Any]:
        try:
            bundle = self.manager.current_model()
        except RuntimeError:
            return {"engine_mode": None, "qualification": {}, "model_revision": None}
        runtime = self._cache_runtime if self._cache_bundle is bundle else None
        report = native_report(bundle, runtime, backend=self._cache_config.backend,
                               mode=self._inference_mode, execution_dtype=self._execution_dtype)
        from mlx_one.engine.timing import METRIC_DEFINITIONS

        report["selection"] = self._selection
        report["metric_definitions"] = METRIC_DEFINITIONS
        report["latest_completed_request"] = self.stats.to_dict()
        report["aggregate_metric_scope"] = "latest_completed_request"
        report["memory_pressure"] = self._pressure.stats() if self._pressure else None
        return report

    def cache_stats(self) -> dict[str, Any]:
        if self._cache_runtime is None:
            return {}
        runtime = self._cache_runtime
        result: dict[str, Any] = runtime.stats().to_dict()
        block_supported = (
            runtime.plan.architecture == "qwen2" and runtime.plan.block_compatible
            and not getattr(runtime.bundle, "quantization", {})
        )
        result.update(
            {
                "backend": runtime.config.backend,
                "block_size_tokens": runtime.config.block_size_tokens,
                "capabilities": {
                    "dense_kv": True,
                    "quantized_kv": True,
                    "block_kv": block_supported,
                    "prefix_block_cache": block_supported,
                    "sliding_block_kv": False,
                    "hybrid_prefix_restore": runtime.hybrid_state,
                    "continuous_batching": self._inference_mode in {
                        "experimental_dense_batch", "paged", "experimental_hybrid_batch"},
                },
            }
        )
        return result

    def preview_prefix_tokens(
        self,
        prompt_ids: tuple[int, ...],
        *,
        cache_type_k: str,
        cache_type_v: str,
        context_length: int,
        minimum_tokens: int,
        cache_idle_slots: bool,
        slot: int = 0,
        cache_namespace: str | None = None,
    ) -> int:
        bundle = self.manager.current_model()
        runtime = self._cache_manager_for(bundle)
        fingerprint = runtime.fingerprint(
            key_type=cache_type_k,
            value_type=cache_type_v,
            context_tokens=context_length,
            namespace=cache_namespace,
        )
        return runtime.preview_prefix(
            fingerprint, prompt_ids, minimum_tokens=minimum_tokens
        )

    def bind_actor(self, actor) -> None:
        self._actor = actor
        actor.pressure = self._pressure
        actor.execution_mode = self._inference_mode
        config = actor.config
        self._prefill_chunk_size = min(self._prefill_chunk_size, config.prefill_chunk_size,
                                       config.max_prefill_tokens,
                                       max(config.max_scheduled_tokens // 2, 1))

    def _cache_manager_for(self, bundle: Any) -> CacheManager:
        if (self._actor is not None and threading.current_thread() is not self._actor._thread
                and not self._actor.device_executor.is_owner
                and (self._cache_runtime is None or self._cache_bundle is not bundle)):
            return self._actor.execute(lambda: self._cache_manager_for(bundle))
        if self._cache_runtime is None or self._cache_bundle is not bundle:
            if self._cache_runtime is not None:
                self._cache_runtime.close()
            self._cache_runtime = None
            self._cache_bundle = None
            if self._converted_bundle is not bundle:
                configure_execution_dtype(bundle, self._execution_dtype)
                self._converted_bundle = bundle
            from mlx_one.engine.construction import construct_runtime

            runtime, selection = construct_runtime(
                bundle, self._cache_config, self._requested_mode,
                batching=self._actor.config if self._actor is not None else None,
                pressure=self._pressure, registry=self._qualification_registry,
                profile=self._qualification_profile)
            try:
                if self._pressure is not None:
                    self._pressure.start()
                self._selection = selection
                self._inference_mode = selection["resolved_mode"]
                if self._actor is not None:
                    self._actor.execution_mode = self._inference_mode
            except BaseException:
                runtime.close()
                raise
            self._cache_runtime = runtime
            self._cache_bundle = bundle
        return self._cache_runtime

    def preview_request(self, messages, options, *, namespace, context_length,
                        context_shift=False, reasoning="auto", reasoning_budget=-1,
                        reasoning_preserve=False, minimum_tokens=1):
        bundle = self.manager.current_model()
        from mlx_one.engine.prompts import prepare_chat

        prepared = prepare_chat(bundle, messages, options.max_tokens, context_length,
                                context_shift=context_shift,
                                rendering_options=_reasoning_template_options(
                                    reasoning, reasoning_budget, reasoning_preserve))
        runtime = self._cache_manager_for(bundle)
        fingerprint = runtime.fingerprint(
            namespace=namespace, context_tokens=prepared.context_tokens,
            key_type=runtime.config.key_type, value_type=runtime.config.value_type,
            rendering_options=prepared.rendering_options)
        return runtime.preview_prefix(fingerprint, prepared.token_ids,
                                      minimum_tokens=minimum_tokens)


def _reasoning_template_options(
    reasoning: str, budget: int = -1, preserve: bool = False
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "thinking_budget": budget,
        "reasoning_budget": budget,
        "preserve_reasoning": preserve,
        "preserve_thinking": preserve,
    }
    if reasoning != "auto":
        options["enable_thinking"] = reasoning == "on"
    return options


def _fit_messages(
    bundle: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    context_length: int,
    context_shift: bool,
    reasoning: str,
    reasoning_budget: int = -1,
    reasoning_preserve: bool = False,
) -> tuple[list[dict[str, str]], str]:
    from mlx_one.engine.prompts import prepare_chat

    prepared = prepare_chat(bundle, messages, max_tokens, context_length,
                            context_shift=context_shift,
                            rendering_options=_reasoning_template_options(
                                reasoning, reasoning_budget, reasoning_preserve))
    return list(prepared.messages), prepared.text


class _ReasoningParser:
    def __init__(
        self,
        format_name: str,
        preserve: bool,
        budget: int,
        *,
        start_in_reasoning: bool = False,
    ) -> None:
        self.format_name = format_name
        self.preserve = preserve
        self.budget = budget
        self.in_reasoning = start_in_reasoning
        self.buffer = ""
        self.reasoning_tokens = 0
        self._emit_open = start_in_reasoning and self._keeps_tags

    @property
    def _keeps_tags(self) -> bool:
        return self.format_name == "deepseek-legacy"

    def feed(self, piece: str) -> tuple[str, str]:
        if self.format_name == "none":
            return piece, ""
        self.buffer += piece
        content = "<think>" if self._emit_open else ""
        self._emit_open = False
        reasoning = ""
        while self.buffer:
            marker = "</think>" if self.in_reasoning else "<think>"
            index = self.buffer.find(marker)
            if index < 0:
                keep = min(len(self.buffer), len(marker) - 1)
                ready = self.buffer[:-keep] if keep else self.buffer
                self.buffer = self.buffer[-keep:] if keep else ""
                if self.in_reasoning:
                    if ready:
                        self.reasoning_tokens += 1
                    reasoning += ready
                    if self._keeps_tags:
                        content += ready
                else:
                    content += ready
                break
            ready = self.buffer[:index]
            self.buffer = self.buffer[index + len(marker) :]
            if self.in_reasoning:
                reasoning += ready
                if self._keeps_tags:
                    content += ready + marker
            else:
                content += ready
                if self._keeps_tags:
                    content += marker
            self.in_reasoning = not self.in_reasoning
        return content, reasoning

    def flush(self) -> tuple[str, str]:
        ready, self.buffer = self.buffer, ""
        if self.in_reasoning:
            if ready:
                self.reasoning_tokens += 1
            return (ready if self._keeps_tags else ""), ready
        return ready, ""
