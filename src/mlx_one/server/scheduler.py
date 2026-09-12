"""Bounded single-worker FIFO scheduler for MLX generation."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


class QueueFullError(RuntimeError):
    pass


@dataclass
class _Job:
    factory: Callable[[threading.Event], Iterator[Any]]
    cancel: threading.Event = field(default_factory=threading.Event)
    output: queue.Queue[tuple[str, Any]] = field(default_factory=queue.Queue)


class GenerationScheduler:
    def __init__(self, max_pending: int = 8) -> None:
        self._jobs: queue.Queue[_Job | None] = queue.Queue(maxsize=max_pending)
        self._closed = threading.Event()
        self._active: _Job | None = None
        self._thread = threading.Thread(target=self._run, name="mlx-one-generation", daemon=True)
        self._thread.start()

    def schedule(self, factory: Callable[[threading.Event], Iterator[Any]]) -> AsyncIterator[Any]:
        if self._closed.is_set():
            raise RuntimeError("generation scheduler is closed")
        job = _Job(factory)
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            raise QueueFullError("generation queue is full") from exc
        return self._read(job)

    async def submit(
        self, factory: Callable[[threading.Event], Iterator[Any]]
    ) -> AsyncIterator[Any]:
        async for item in self.schedule(factory):
            yield item

    async def _read(self, job: _Job) -> AsyncIterator[Any]:
        try:
            while True:
                kind, value = await asyncio.to_thread(job.output.get)
                if kind == "item":
                    yield value
                elif kind == "error":
                    raise value
                else:
                    break
        finally:
            job.cancel.set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._active is not None:
            self._active.cancel.set()
        while True:
            try:
                pending = self._jobs.get_nowait()
            except queue.Empty:
                break
            if pending is not None:
                pending.cancel.set()
                pending.output.put(("error", RuntimeError("generation scheduler stopped")))
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._closed.is_set():
            job = self._jobs.get()
            if job is None:
                return
            self._active = job
            try:
                if not job.cancel.is_set():
                    for item in job.factory(job.cancel):
                        if job.cancel.is_set():
                            break
                        job.output.put(("item", item))
            except BaseException as exc:
                job.output.put(("error", exc))
            finally:
                job.output.put(("done", None))
                self._active = None
