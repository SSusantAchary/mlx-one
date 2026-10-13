"""Public lifecycle wrapper for native mlx-one text inference."""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.modes import (
    ACTOR_MODES,
    InferenceMode,
    configure_execution_dtype,
    resolve_backend,
    resolved_mode,
    validate_mode,
)
from mlx_one.engine.pressure import MemoryPressureConfig, MemoryPressureController
from mlx_one.engine.reporting import native_report
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
        inference_mode: InferenceMode | None = None,
        execution_dtype: str | None = None,
        batching_config: BatchingConfig | None = None,
        cache_namespace: str | None = None,
        memory_pressure_config: MemoryPressureConfig | None = None,
        qualification_registry: Any = None,
        qualification_profile: dict[str, Any] | None = None,
    ) -> None:
        backend = resolve_backend(inference_mode, cache_config.backend if cache_config else None)
        config = (replace(cache_config, backend=backend)
                  if cache_config else CacheConfig(backend=backend))
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
        configure_execution_dtype(self._bundle, execution_dtype)
        self._requested_mode = inference_mode
        self._inference_mode = inference_mode
        self._execution_dtype = execution_dtype
        self._cache_namespace = cache_namespace
        self._batching_config = batching_config or BatchingConfig()
        self._actor = None
        self._cache_manager = None
        self._pressure = None
        if memory_pressure_config is not None or self._requested_mode in {
            "paged", "experimental_hybrid_batch", "auto"
        }:
            self._pressure = MemoryPressureController(memory_pressure_config)
        self._selection = {"requested_mode": inference_mode,
                           "resolved_mode": inference_mode or resolved_mode(backend),
                           "reason": "explicit configuration", "evidence_scope": None}
        if callable(getattr(self._bundle.model, "make_cache", None)):
            from mlx_one.engine.construction import construct_runtime

            self._cache_manager, self._selection = construct_runtime(
                self._bundle, config, inference_mode, batching=self._batching_config,
                pressure=self._pressure, registry=qualification_registry,
                profile=qualification_profile)
            inference_mode = self._selection["resolved_mode"]
            self._inference_mode = inference_mode
            backend = self._cache_manager.config.backend
        if (self._batching_config.admission_policy == "cache_aware"
                and inference_mode not in ACTOR_MODES):
            raise ValueError("cache-aware admission requires an actor execution mode")
        try:
            validate_mode(self._bundle, self._cache_manager.plan if self._cache_manager else None,
                          inference_mode)
            if self._pressure is not None:
                self._pressure.start()
        except Exception:
            if self._cache_manager is not None:
                self._cache_manager.close()
            if self._pressure is not None:
                self._pressure.close()
            raise
        self._backend = backend
        if inference_mode in ACTOR_MODES or self._requested_mode == "auto":
            from mlx_one.engine.actor import InferenceActor
            from mlx_one.engine.dispatch import DispatchRunner

            self._actor = InferenceActor(self._batching_config, runner=DispatchRunner(),
                                         pressure=self._pressure)
            self._actor.execution_mode = inference_mode
        workers = (self._batching_config.max_sequences + self._batching_config.queue_size
                   if self._actor is not None else 1)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mlx-one-engine")
        self._closed = False
        self._latest_result: GenerationResult | None = None
        self._lock = threading.RLock()

    @property
    def model(self) -> LoadedTextModel:
        self._ensure_open()
        return self._bundle

    def capabilities(self) -> dict[str, Any]:
        bundle = self.model
        block_supported = (
            self._cache_manager is not None
            and self._cache_manager.plan.architecture == "qwen2"
            and self._cache_manager.plan.block_compatible
            and not bundle.quantization
        )
        return {
            "architecture": bundle.architecture,
            "modalities": ("text",),
            "tasks": ("text-generation",),
            "context_length": bundle.context_length,
            "quantization": dict(bundle.quantization),
            "streaming": True,
            "cancellation": True,
            **self.runtime_stats(),
            "cache": {
                "dense_kv": True,
                "quantized_kv": True,
                "block_kv": block_supported,
                "prefix_block_cache": block_supported,
                "sliding_block_kv": False,
                "hybrid_prefix_restore": self._inference_mode == "experimental_hybrid_batch",
                "continuous_batching": self._inference_mode in ACTOR_MODES,
            },
        }

    def runtime_stats(self) -> dict[str, Any]:
        self._ensure_open()
        report = native_report(self._bundle, self._cache_manager, backend=self._backend,
                               mode=self._inference_mode, execution_dtype=self._execution_dtype)
        if self._actor is not None:
            report["scheduler"] = self._actor.stats()
        report["selection"] = self._selection
        report["memory_pressure"] = self._pressure.stats() if self._pressure else None
        report["cache_stats"] = (
            self._cache_manager.stats().to_dict() if self._cache_manager else None
        )
        return report

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
        if self._actor is not None:
            latest = None
            for chunk in self._actor.schedule_sync(
                self._batch_factory(prompt, resolved), cancel=event):
                latest = chunk
                yield chunk
            if event.is_set() and (latest is None or latest.finish_reason is None):
                yield GenerationChunk(None, "", latest.generated_tokens if latest else 0, "stop")
        else:
            yield from stream_generate(
                self.model, prompt, options=resolved, is_cancelled=event.is_set,
                _cache_manager=self._cache_manager, _cache_namespace=self._cache_namespace)

    def _batch_factory(self, prompt, options):
        cfg = self._batching_config
        chunk = min(cfg.prefill_chunk_size, cfg.max_prefill_tokens,
                    max(cfg.max_scheduled_tokens // 2, 1))
        bundle = self.model
        def factory(cancel):
            return stream_generate(
                bundle, prompt, options=options, is_cancelled=cancel.is_set,
                _cache_manager=self._cache_manager, _cache_namespace=self._cache_namespace,
                _cooperative=True, _prefill_chunk_size=chunk)
        if self._cache_namespace is not None:
            factory.prefix_preview = lambda: self._cache_manager.preview_prefix(
                self._cache_manager.fingerprint(namespace=self._cache_namespace,
                                                key_type=self._cache_manager.config.key_type,
                                                value_type=self._cache_manager.config.value_type),
                tuple(bundle.tokenizer.encode(prompt)))
        return factory

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
        if self._actor is not None:
            job = self._actor._enqueue(self._batch_factory(prompt, resolved), cancel=cancel)
            future = self._executor.submit(
                self._consume_chunks, prompt, self._actor.read_sync(job), cancel)
        else:
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
        return self._consume_chunks(
            prompt, self.stream(prompt, options=options, cancel=cancel), cancel)

    def _consume_chunks(self, prompt, stream, cancel=None):
        chunks = tuple(stream)
        terminal = chunks[-1] if chunks else None
        result = GenerationResult(
            prompt=prompt,
            text="".join(chunk.text for chunk in chunks),
            prompt_tokens=len(self.model.tokenizer.encode(prompt)) or 1,
            generation_tokens=sum(chunk.token_id is not None for chunk in chunks),
            finish_reason=("stop" if cancel is not None and cancel.is_set() else
                           (terminal.finish_reason if terminal else None) or "length"),
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
        if self._actor is not None:
            self._actor.close()
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._cache_manager is not None:
            self._cache_manager.close()
        if self._pressure is not None:
            self._pressure.close()
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
