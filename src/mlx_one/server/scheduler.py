"""Bounded single-worker FIFO scheduler for MLX generation."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


class QueueFullError(RuntimeError):
    pass


class RequestTimeoutError(TimeoutError):
    pass


_slot_local = threading.local()


def current_slot() -> int:
    return int(getattr(_slot_local, "value", 0))


@dataclass
class _Job:
    factory: Callable[[threading.Event], Iterator[Any]]
    cancel: threading.Event = field(default_factory=threading.Event)
    output: queue.Queue[tuple[str, Any]] = field(default_factory=queue.Queue)
    deadline: float | None = None
    iterator: Iterator[Any] | None = None
    slot_id: int | None = None


@dataclass
class _Call:
    factory: Callable[[], Any]
    result: Future[Any] = field(default_factory=Future)


class GenerationScheduler:
    def __init__(self, max_pending: int = 8, *, parallel: int = 1, timeout: float = 0) -> None:
        if max_pending < 1 or parallel < 1 or timeout < 0:
            raise ValueError("invalid scheduler capacity or timeout")
        self._jobs: queue.Queue[_Job | _Call | None] = queue.Queue(maxsize=max_pending)
        self._closed = threading.Event()
        self._active: _Job | None = None
        self._active_jobs: deque[_Job] = deque()
        self._parallel = parallel
        self._timeout = timeout
        self._deferred_call: _Call | None = None
        self._finalizer: Callable[[], Any] | None = None
        self._thread = threading.Thread(target=self._run, name="mlx-one-generation", daemon=True)
        self._thread.start()

    def stats(self) -> dict[str, int]:
        active = len(self._active_jobs) + (1 if self._active is not None else 0)
        return {
            "active_slots": active,
            "parallel_slots": self._parallel,
            "queue_depth": self._jobs.qsize(),
            "queue_capacity": self._jobs.maxsize,
        }

    def execute(self, factory: Callable[[], Any]) -> Any:
        """Run lifecycle work on the same thread used for model generation."""

        if self._closed.is_set():
            raise RuntimeError("generation scheduler is closed")
        if threading.current_thread() is self._thread:
            return factory()
        call = _Call(factory)
        try:
            self._jobs.put_nowait(call)
        except queue.Full as exc:
            raise QueueFullError("generation queue is full") from exc
        return call.result.result()

    def schedule(self, factory: Callable[[threading.Event], Iterator[Any]]) -> AsyncIterator[Any]:
        if self._closed.is_set():
            raise RuntimeError("generation scheduler is closed")
        deadline = time.monotonic() + self._timeout if self._timeout else None
        job = _Job(factory, deadline=deadline)
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

    def close(self, finalizer: Callable[[], Any] | None = None) -> None:
        if self._closed.is_set():
            return
        self._finalizer = finalizer
        self._closed.set()
        if self._active is not None:
            self._active.cancel.set()
        for active in tuple(self._active_jobs):
            active.cancel.set()
        if self._deferred_call is not None:
            self._deferred_call.result.set_exception(
                RuntimeError("generation scheduler stopped")
            )
            self._deferred_call = None
        while True:
            try:
                pending = self._jobs.get_nowait()
            except queue.Empty:
                break
            if isinstance(pending, _Job):
                pending.cancel.set()
                pending.output.put(("error", RuntimeError("generation scheduler stopped")))
            elif isinstance(pending, _Call):
                pending.result.set_exception(RuntimeError("generation scheduler stopped"))
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)

    def _run(self) -> None:
        try:
            while True:
                if self._active_jobs:
                    self._fill_active()
                    work = self._active_jobs.popleft()
                    self._active = work
                    if self._step(work):
                        self._active_jobs.append(work)
                    self._active = None
                    continue
                if self._closed.is_set():
                    return
                if self._deferred_call is not None:
                    work = self._deferred_call
                    self._deferred_call = None
                    try:
                        work.result.set_result(work.factory())
                    except BaseException as exc:
                        work.result.set_exception(exc)
                    continue
                work = self._jobs.get()
                if work is None:
                    return
                if isinstance(work, _Call):
                    try:
                        work.result.set_result(work.factory())
                    except BaseException as exc:
                        work.result.set_exception(exc)
                    continue
                self._assign_slot(work)
                self._active_jobs.append(work)
        finally:
            if self._finalizer is not None:
                self._finalizer()

    def _fill_active(self) -> None:
        if self._deferred_call is not None:
            return
        while len(self._active_jobs) < self._parallel:
            try:
                work = self._jobs.get_nowait()
            except queue.Empty:
                return
            if work is None:
                self._closed.set()
                return
            if isinstance(work, _Call):
                # Lifecycle calls are barriers and run after active inference.
                self._deferred_call = work
                return
            self._assign_slot(work)
            self._active_jobs.append(work)

    def _assign_slot(self, work: _Job) -> None:
        used = {job.slot_id for job in self._active_jobs}
        work.slot_id = next(index for index in range(self._parallel) if index not in used)

    def _step(self, work: _Job) -> bool:
        try:
            _slot_local.value = work.slot_id or 0
            if work.cancel.is_set():
                work.output.put(("done", None))
                return False
            if work.deadline is not None and time.monotonic() >= work.deadline:
                work.cancel.set()
                raise RequestTimeoutError("generation request timed out")
            if work.iterator is None:
                work.iterator = iter(work.factory(work.cancel))
            work.output.put(("item", next(work.iterator)))
            return True
        except StopIteration:
            work.output.put(("done", None))
            return False
        except BaseException as exc:
            work.output.put(("error", exc))
            work.output.put(("done", None))
            return False
