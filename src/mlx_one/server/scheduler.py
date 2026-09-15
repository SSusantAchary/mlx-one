"""Bounded single-worker FIFO scheduler for MLX generation."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
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
    created_at: float = field(default_factory=time.monotonic)
    first_step_at: float | None = None
    recorded: bool = False


@dataclass
class _Call:
    factory: Callable[[], Any]
    result: Future[Any] = field(default_factory=Future)
    deadline: float | None = None


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
        self._metrics_lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._cancelled = 0
        self._timed_out = 0
        self._failed = 0
        self._rejected = 0
        self._latencies_ms: deque[float] = deque(maxlen=1024)
        self._ttft_ms: deque[float] = deque(maxlen=1024)
        self._thread = threading.Thread(target=self._run, name="mlx-one-generation", daemon=True)
        self._thread.start()

    def stats(self) -> dict[str, Any]:
        active = len(self._active_jobs) + (1 if self._active is not None else 0)
        with self._metrics_lock:
            result = {
                "submitted_requests": self._submitted,
                "completed_requests": self._completed,
                "cancelled_requests": self._cancelled,
                "timed_out_requests": self._timed_out,
                "failed_requests": self._failed,
                "rejected_requests": self._rejected,
                "latency_p50_ms": _percentile(self._latencies_ms, 0.50),
                "latency_p95_ms": _percentile(self._latencies_ms, 0.95),
                "latency_p99_ms": _percentile(self._latencies_ms, 0.99),
                "ttft_p50_ms": _percentile(self._ttft_ms, 0.50),
            }
        return {
            "active_slots": active,
            "parallel_slots": self._parallel,
            "queue_depth": self._jobs.qsize(),
            "queue_capacity": self._jobs.maxsize,
            **result,
        }

    def execute(self, factory: Callable[[], Any], *, apply_timeout: bool = False) -> Any:
        """Run lifecycle work on the same thread used for model generation."""

        if self._closed.is_set():
            raise RuntimeError("generation scheduler is closed")
        if threading.current_thread() is self._thread:
            return factory()
        deadline = (
            time.monotonic() + self._timeout
            if apply_timeout and self._timeout
            else None
        )
        call = _Call(factory, deadline=deadline)
        try:
            self._jobs.put_nowait(call)
        except queue.Full as exc:
            raise QueueFullError("generation queue is full") from exc
        try:
            return call.result.result(self._timeout if deadline is not None else None)
        except FutureTimeoutError as exc:
            raise RequestTimeoutError("request timed out before task completion") from exc

    def schedule(self, factory: Callable[[threading.Event], Iterator[Any]]) -> AsyncIterator[Any]:
        if self._closed.is_set():
            raise RuntimeError("generation scheduler is closed")
        deadline = time.monotonic() + self._timeout if self._timeout else None
        job = _Job(factory, deadline=deadline)
        try:
            self._jobs.put_nowait(job)
        except queue.Full as exc:
            with self._metrics_lock:
                self._rejected += 1
            raise QueueFullError("generation queue is full") from exc
        with self._metrics_lock:
            self._submitted += 1
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
                self._record(pending, "cancelled")
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
                        if work.deadline is not None and time.monotonic() >= work.deadline:
                            raise RequestTimeoutError("request timed out in the task queue")
                        work.result.set_result(work.factory())
                    except BaseException as exc:
                        work.result.set_exception(exc)
                    continue
                work = self._jobs.get()
                if work is None:
                    return
                if isinstance(work, _Call):
                    try:
                        if work.deadline is not None and time.monotonic() >= work.deadline:
                            raise RequestTimeoutError("request timed out in the task queue")
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
                self._record(work, "cancelled")
                return False
            if work.deadline is not None and time.monotonic() >= work.deadline:
                work.cancel.set()
                raise RequestTimeoutError("generation request timed out")
            if work.iterator is None:
                work.iterator = iter(work.factory(work.cancel))
            item = next(work.iterator)
            if work.first_step_at is None:
                work.first_step_at = time.monotonic()
            work.output.put(("item", item))
            return True
        except StopIteration:
            work.output.put(("done", None))
            self._record(work, "completed")
            return False
        except RequestTimeoutError as exc:
            work.output.put(("error", exc))
            work.output.put(("done", None))
            self._record(work, "timed_out")
            return False
        except BaseException as exc:
            work.output.put(("error", exc))
            work.output.put(("done", None))
            self._record(work, "failed")
            return False

    def _record(self, work: _Job, outcome: str) -> None:
        if work.recorded:
            return
        work.recorded = True
        ended = time.monotonic()
        with self._metrics_lock:
            setattr(self, f"_{outcome}", getattr(self, f"_{outcome}") + 1)
            self._latencies_ms.append((ended - work.created_at) * 1000)
            if work.first_step_at is not None:
                self._ttft_ms.append((work.first_step_at - work.created_at) * 1000)


def _percentile(values: deque[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]
