"""OpenAI request adaptation onto mlx-one native generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from mlx_one.core.cache import clone_caches
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationEngine:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self._stats = RuntimeStats()
        self._lock = threading.Lock()
        self._prompt_caches: list[
            tuple[int, tuple[object, ...], tuple[int, ...], tuple[object, ...]]
        ] = []

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
        cached_ids: tuple[int, ...] = ()
        cached_state: tuple[object, ...] | None = None
        slot = current_slot()
        cache_key = (
            bundle.model_id,
            bundle.revision,
            id(bundle.tokenizer),
            bundle.chat_template.template,
            reasoning,
            reasoning_budget,
            reasoning_preserve,
            context_length or bundle.context_length,
            cache_type_k,
            cache_type_v,
        )
        if cache_prompt:
            candidates = [
                entry
                for entry in self._prompt_caches
                if (cache_idle_slots or entry[0] == slot)
                and entry[1] == cache_key
                and len(entry[2]) >= cache_reuse
                and len(entry[2]) < len(prompt_ids)
                and prompt_ids[: len(entry[2])] == entry[2]
            ]
            if candidates:
                _, _, cached_ids, cached_state = max(
                    candidates, key=lambda item: len(item[2])
                )
                cached_state = clone_caches(cached_state)
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
            cache=cached_state,
            cached_prompt_ids=cached_ids,
            cache_type_k=cache_type_k,
            cache_type_v=cache_type_v,
            on_cache_update=(
                lambda tokens, caches: self._store_prompt_cache(
                    slot, cache_key, tokens, caches, cache_entries
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
            prompt_cache_hit=bool(cached_state),
            reused_prompt_tokens=len(cached_ids),
            drafted_tokens=speculative_stats["drafted_tokens"],
            accepted_draft_tokens=speculative_stats["accepted_draft_tokens"],
        )
        with self._lock:
            self._stats = stats
        yield GenerationEvent(finish_reason=finish_reason, metrics=stats.to_dict())

    def _store_prompt_cache(
        self,
        slot: int,
        key: tuple[object, ...],
        tokens: tuple[int, ...],
        caches: tuple[object, ...],
        capacity: int,
    ) -> None:
        if not tokens or capacity < 1:
            return
        self._prompt_caches = [entry for entry in self._prompt_caches if entry[0] != slot]
        self._prompt_caches.append((slot, key, tokens, clone_caches(caches)))
        del self._prompt_caches[: -capacity]

    def clear_caches(self) -> None:
        self._prompt_caches.clear()


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
