"""OpenAI request adaptation onto mlx-one native generation."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from mlx_one.server.model_manager import ModelManager
from mlx_one.text import TextGenerationOptions, stream_chat
from mlx_one.utils.memory import get_memory_snapshot


@dataclass(frozen=True)
class GenerationEvent:
    text: str = ""
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationEngine:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self._stats = RuntimeStats()
        self._lock = threading.Lock()

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
    ) -> Iterator[GenerationEvent]:
        bundle = self.manager.current_model()
        if model != bundle.model_id:
            raise ValueError(f"model {model!r} is not loaded")
        if bundle.chat_template is None:
            raise RuntimeError("loaded model has no chat template")
        prompt = bundle.chat_template.render(messages)
        prompt_tokens = len(bundle.tokenizer.encode(prompt))
        started = time.perf_counter()
        first_at: float | None = None
        generated = 0
        finish_reason = "length"
        for chunk in stream_chat(
            bundle, messages, options=options, is_cancelled=cancel.is_set
        ):
            if chunk.token_id is not None:
                generated = chunk.generated_tokens
                if first_at is None:
                    first_at = time.perf_counter()
                if chunk.text:
                    yield GenerationEvent(text=chunk.text)
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
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
        )
        with self._lock:
            self._stats = stats
        yield GenerationEvent(finish_reason=finish_reason, metrics=stats.to_dict())
