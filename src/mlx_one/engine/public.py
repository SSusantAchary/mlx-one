"""Public lifecycle wrapper for native mlx-one text inference."""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.text import (
    GenerationChunk,
    GenerationResult,
    LoadedTextModel,
    TextGenerationOptions,
    generate,
    load_text_model,
    stream_generate,
)


class EngineRequestHandle:
    """A cooperatively cancellable generation submitted to an ``Engine``."""

    def __init__(self, future: Future[GenerationResult], cancel_event: threading.Event) -> None:
        self._future = future
        self._cancel_event = cancel_event

    def cancel(self) -> bool:
        self._cancel_event.set()
        return self._future.cancel() or not self._future.done()

    def result(self, timeout: float | None = None) -> GenerationResult:
        return self._future.result(timeout)

    def done(self) -> bool:
        return self._future.done()


class Engine:
    """Own a native text model and expose synchronous and asynchronous generation."""

    def __init__(
        self,
        model: str | Path | LoadedTextModel,
        *,
        revision: str | None = None,
        offline: bool = False,
        cache_dir: str | Path | None = None,
        tokenizer_source: str | Path | None = None,
        cache_config: CacheConfig | None = None,
    ) -> None:
        self._bundle = (
            model
            if isinstance(model, LoadedTextModel)
            else load_text_model(
                model,
                revision=revision,
                offline=offline,
                cache_dir=cache_dir,
                tokenizer_source=tokenizer_source,
            )
        )
        self._cache_manager = (
            CacheManager(self._bundle, cache_config)
            if callable(getattr(self._bundle.model, "make_cache", None))
            else None
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-one-engine")
        self._closed = False
        self._latest_result: GenerationResult | None = None
        self._lock = threading.RLock()

    @property
    def model(self) -> LoadedTextModel:
        self._ensure_open()
        return self._bundle

    @property
    def capabilities(self) -> dict[str, Any]:
        bundle = self.model
        qualified_block = (
            self._cache_manager is not None
            and self._cache_manager.plan.architecture == "qwen2"
            and self._cache_manager.plan.block_compatible
        )
        return {
            "architecture": bundle.architecture,
            "modalities": ("text",),
            "tasks": ("text-generation",),
            "context_length": bundle.context_length,
            "quantization": dict(bundle.quantization),
            "streaming": True,
            "cancellation": True,
            "cache": {
                "dense_kv": True,
                "quantized_kv": True,
                "block_kv": qualified_block,
                "prefix_block_cache": qualified_block,
                "sliding_block_kv": False,
                "hybrid_prefix_restore": False,
                "continuous_batching": False,
            },
        }

    @property
    def latest_result(self) -> GenerationResult | None:
        with self._lock:
            return self._latest_result

    def generate(
        self,
        prompt: str,
        *,
        options: TextGenerationOptions | None = None,
        **settings: Any,
    ) -> GenerationResult:
        resolved = _options(options, settings)
        if self._cache_manager is None:
            result = generate(self.model, prompt, options=resolved)
            if not isinstance(result, GenerationResult):
                raise RuntimeError("single-prompt generation returned a batch")
            with self._lock:
                self._latest_result = result
            return result
        return self._consume(prompt, resolved, threading.Event())

    def stream(
        self,
        prompt: str,
        *,
        options: TextGenerationOptions | None = None,
        cancel: threading.Event | None = None,
        **settings: Any,
    ) -> Iterator[GenerationChunk]:
        resolved = _options(options, settings)
        event = cancel or threading.Event()
        yield from stream_generate(
            self.model,
            prompt,
            options=resolved,
            is_cancelled=event.is_set,
            _cache_manager=self._cache_manager,
        )

    def submit(
        self,
        prompt: str,
        *,
        options: TextGenerationOptions | None = None,
        **settings: Any,
    ) -> EngineRequestHandle:
        self._ensure_open()
        cancel = threading.Event()
        resolved = _options(options, settings)
        future = self._executor.submit(self._consume, prompt, resolved, cancel)
        return EngineRequestHandle(future, cancel)

    async def async_generate(
        self,
        prompt: str,
        *,
        options: TextGenerationOptions | None = None,
        **settings: Any,
    ) -> GenerationResult:
        return await asyncio.to_thread(
            self.generate, prompt, options=options, **settings
        )

    def _consume(
        self,
        prompt: str,
        options: TextGenerationOptions,
        cancel: threading.Event,
    ) -> GenerationResult:
        chunks = tuple(self.stream(prompt, options=options, cancel=cancel))
        terminal = chunks[-1] if chunks else None
        result = GenerationResult(
            prompt=prompt,
            text="".join(chunk.text for chunk in chunks),
            prompt_tokens=len(self.model.tokenizer.encode(prompt)) or 1,
            generation_tokens=sum(chunk.token_id is not None for chunk in chunks),
            finish_reason=(terminal.finish_reason if terminal else None) or "length",
            model=self.model.model_id,
            revision=self.model.revision,
        )
        with self._lock:
            self._latest_result = result
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._cache_manager is not None:
            self._cache_manager.close()
        gc.collect()
        try:
            mx = sys.modules.get("mlx.core")
            if mx is not None:
                mx.clear_cache()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")

    def __enter__(self) -> Engine:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _options(
    options: TextGenerationOptions | None, settings: dict[str, Any]
) -> TextGenerationOptions:
    if options is not None and settings:
        raise ValueError("pass either options or generation keyword settings, not both")
    return options or TextGenerationOptions(**settings)
