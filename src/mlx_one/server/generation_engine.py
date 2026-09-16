"""OpenAI request adaptation onto mlx-one native generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from mlx_one.engine.cache_errors import PrefixReuseUnsupported
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig, CacheHandle
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import current_slot
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationEngine:
    def __init__(
        self,
        manager: ModelManager,
        *,
        prefix_cache_bytes: int = 512 * 1024**2,
        cache_backend: str = "dense",
        cache_block_size: int = 32,
        cache_memory_budget_bytes: int | None = None,
    ) -> None:
        self.manager = manager
        self._stats = RuntimeStats()
        self._lock = threading.Lock()
        self._cache_config = CacheConfig(
            backend=cache_backend,  # type: ignore[arg-type]
            memory_budget_bytes=cache_memory_budget_bytes,
            prefix_cache_budget_bytes=prefix_cache_bytes,
            block_size_tokens=cache_block_size,
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
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        spec_type: str = "none",
        spec_draft_n_max: int = 3,
    ) -> Iterator[GenerationEvent]:
        lifecycle: dict[str, Any] = {}
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
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
                spec_type=spec_type,
                spec_draft_n_max=spec_draft_n_max,
                lifecycle=lifecycle,
            )
        finally:
            runtime = lifecycle.get("runtime")
            handle = lifecycle.get("handle")
            if (
                isinstance(runtime, CacheManager)
                and isinstance(handle, CacheHandle)
                and not handle.closed
            ):
                runtime.release(handle)

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
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
        spec_type: str = "none",
        spec_draft_n_max: int = 3,
        lifecycle: dict[str, Any],
    ) -> Iterator[GenerationEvent]:
        bundle = self.manager.current_model()
        if model != self.manager.model_info().id:
            raise ValueError(f"model {model!r} is not loaded")
        if bundle.chat_template is None:
            raise RuntimeError("loaded model has no chat template")
        if reasoning == "on" and "enable_thinking" not in (bundle.chat_template.template or ""):
            raise ValueError("loaded chat template does not support explicit reasoning control")
        rendered_messages, prompt = _fit_messages(
            bundle,
            messages,
            options.max_tokens,
            context_length or bundle.context_length,
            context_shift,
            reasoning,
            reasoning_budget,
            reasoning_preserve,
        )
        prompt_tokens = len(bundle.tokenizer.encode(prompt))
        prompt_ids = tuple(bundle.tokenizer.encode(prompt))
        runtime = self._cache_manager_for(bundle)
        handle = runtime.create_handle(
            f"server-{time.time_ns()}",
            key_type=cache_type_k,
            value_type=cache_type_v,
        )
        lifecycle.update({"runtime": runtime, "handle": handle})
        cached_ids: tuple[int, ...] = ()
        slot = current_slot()
        fingerprint = runtime.fingerprint(
            key_type=cache_type_k,
            value_type=cache_type_v,
            context_tokens=context_length or bundle.context_length,
            namespace="global" if cache_idle_slots else f"slot:{slot}",
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
        for chunk in stream_chat(
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
        ):
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
        stats = RuntimeStats(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            tokens_per_second=generated / elapsed if elapsed > 0 else None,
            time_to_first_token_ms=(first_at - started) * 1000 if first_at else None,
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
            decode_tokens_per_second=generated / elapsed if elapsed > 0 else None,
            inter_token_latency_ms=(elapsed * 1000 / generated if generated else None),
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
        yield GenerationEvent(finish_reason=finish_reason, metrics=stats.to_dict())

    def clear_caches(self) -> None:
        if self._cache_runtime is not None:
            self._cache_runtime.clear_prefixes()

    def close(self) -> None:
        if self._cache_runtime is not None:
            self._cache_runtime.close()
            self._cache_runtime = None
            self._cache_bundle = None

    def cache_stats(self) -> dict[str, Any]:
        if self._cache_runtime is None:
            return {}
        runtime = self._cache_runtime
        result: dict[str, Any] = runtime.stats().to_dict()
        qualified_block = (
            runtime.plan.architecture == "qwen2" and runtime.plan.block_compatible
        )
        result.update(
            {
                "backend": runtime.config.backend,
                "block_size_tokens": runtime.config.block_size_tokens,
                "capabilities": {
                    "dense_kv": True,
                    "quantized_kv": True,
                    "block_kv": qualified_block,
                    "prefix_block_cache": qualified_block,
                    "sliding_block_kv": False,
                    "hybrid_prefix_restore": False,
                    "continuous_batching": False,
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
    ) -> int:
        bundle = self.manager.current_model()
        runtime = self._cache_manager_for(bundle)
        fingerprint = runtime.fingerprint(
            key_type=cache_type_k,
            value_type=cache_type_v,
            context_tokens=context_length,
            namespace="global" if cache_idle_slots else f"slot:{slot}",
        )
        return runtime.preview_prefix(
            fingerprint, prompt_ids, minimum_tokens=minimum_tokens
        )

    def _cache_manager_for(self, bundle: Any) -> CacheManager:
        if self._cache_runtime is None or self._cache_bundle is not bundle:
            if self._cache_runtime is not None:
                self._cache_runtime.close()
            self._cache_runtime = CacheManager(bundle, self._cache_config)
            self._cache_bundle = bundle
        return self._cache_runtime


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
    if max_tokens >= context_length:
        raise ValueError("max_tokens leaves no room inside the configured context")
    selected = list(messages)
    template_options = _reasoning_template_options(
        reasoning, reasoning_budget, reasoning_preserve
    )
    while True:
        prompt = bundle.chat_template.render(selected, **template_options)
        if len(bundle.tokenizer.encode(prompt)) + max_tokens <= context_length:
            return selected, prompt
        if not context_shift:
            raise ValueError("messages and requested output exceed the configured context")
        start = 1 if selected and selected[0]["role"] == "system" else 0
        if len(selected) - start <= 1:
            raise ValueError("the system message or newest turn cannot fit in the context")
        # Remove the oldest complete user/assistant turn while retaining the system message.
        end = start + 1
        while end < len(selected) and selected[end]["role"] == "assistant":
            end += 1
        del selected[start:end]


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
