"""Task-neutral service bridge; text uses existing Engine and result/event types."""
from __future__ import annotations

import threading

from mlx_one.engine.model_pool import ModelPool


class InferenceService:
    def __init__(self, *, pool=None):
        self.pool = pool or ModelPool()
        self._models = {}
        self._lock = threading.RLock()
        self._closed = False

    def register_text(self, name, model, *, estimated_bytes=0, **settings):
        """Register local loading policy, without loading weights or executing plugins."""
        with self._lock:
            if self._closed or not name or name in self._models:
                raise ValueError("service is closed or model alias is already registered")
            self._models[name] = (model, dict(settings), estimated_bytes)

    def _lease(self, name):
        from mlx_one.engine.cache import array_nbytes
        from mlx_one.engine.public import Engine

        with self._lock:
            if self._closed:
                raise RuntimeError("inference service is closed")
            model, settings, estimate = self._models[name]
        return self.pool.acquire(name, lambda: Engine(model, **settings), estimated_bytes=estimate,
            measure=lambda engine: array_nbytes(engine.model.model.parameters()),
            drain=lambda engine: engine.close())

    def generate(self, model, prompt, *, options=None, **settings):
        with self._lease(model) as runtime:
            return runtime.value.generate(prompt, options=options, **settings)

    def stream(self, model, prompt, *, options=None, cancel=None, **settings):
        event = cancel or threading.Event()
        with self._lease(model) as runtime:
            stream = runtime.value.stream(prompt, options=options, cancel=event, **settings)
            try:
                yield from stream
            finally:
                event.set()
                stream.close()

    def unload(self, model, *, timeout=None):
        self.pool.unload(model, timeout=timeout)

    def close(self):
        with self._lock:
            self._closed = True
        self.pool.close()
