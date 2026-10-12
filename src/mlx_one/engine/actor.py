"""Single-owner inference actor with bounded channels and lifecycle barriers."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from mlx_one.engine.batching import (
    BatchedDenseRunner,
    BatchingConfig,
    BatchScheduler,
    ForwardRequest,
)
from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import EngineRequest, Phase, RequestKind, SequenceState
from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError, _slot_local


@dataclass(eq=False)
class _Job:
    factory: Any
    cancel: threading.Event
    admission: Any = None
    on_cancel: Any = None
    created_at: float = field(default_factory=time.monotonic)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    output: Any = None
    terminal: Future = field(default_factory=Future)
    iterator: Any = None
    request: ForwardRequest | None = None
    reply: Any = None
    state: SequenceState | None = None
    stalled_at: float | None = None
    deadline: float | None = None
    admitted: bool = False
    retry_at: float = 0
    prefix_preview: Any = None
    preview_tokens: int = 0
    preview_lookup_ms: float = 0.0


@dataclass
class _Call:
    factory: Any
    result: Future = field(default_factory=Future)


class InferenceActor:
    def __init__(self, config: BatchingConfig | None = None, *, runner=None, pressure=None):
        self.config = config or BatchingConfig()
        self.policy = BatchScheduler(self.config)
        self.runner = runner or BatchedDenseRunner()
        self.pressure = pressure
        self.execution_mode = "experimental_dense_batch"
        self._condition = threading.Condition()
        self._pending = deque()
        self._active = []
        self._closed = False
        self._epoch = 0
        self._finalizer = None
        self._traces = deque(maxlen=4096)
        self._trace_sink = None
        self.capture_memory = False
        self._skipped_admissions = 0
        self._counts = dict(submitted=0, completed=0, cancelled=0, failed=0,
                            timed_out=0, rejected=0)
        self._thread = threading.Thread(target=self._run, name="mlx-one-inference", daemon=True)
        self._thread.start()

    def set_trace_sink(self, sink):
        self._trace_sink = sink

    def stats(self):
        with self._condition:
            return {"parallel_slots": self.config.max_sequences, "active_slots": len(self._active),
                    "queue_depth": len(self._pending), "queue_capacity": self.config.queue_size,
                    "model_epoch": self._epoch, "execution_mode": self.execution_mode,
                    "forward_traces": list(self._traces),
                    "skipped_admissions": self._skipped_admissions,
                    **{f"{key}_requests": value for key, value in self._counts.items()}}

    def _enqueue(self, factory, *, cancel=None, admission=None, on_cancel=None,
                 prefix_preview=None):
        job = _Job(factory, cancel if cancel is not None else threading.Event(),
                   admission, on_cancel)
        job.output = queue.Queue(maxsize=self.config.output_buffer_events)
        job.prefix_preview = prefix_preview or getattr(factory, "prefix_preview", None)
        if self.config.timeout:
            job.deadline = job.created_at + self.config.timeout
        with self._condition:
            if self._closed:
                raise RuntimeError("inference actor is closed")
            if len(self._pending) >= self.config.queue_size:
                self._counts["rejected"] += 1
                raise QueueFullError("generation queue is full")
            self._pending.append(job)
            self._counts["submitted"] += 1
            self._condition.notify()
        return job

    def schedule_sync(self, factory, *, cancel=None):
        job = self._enqueue(factory, cancel=cancel)
        yield from self.read_sync(job)

    def read_sync(self, job):
        try:
            while True:
                try:
                    yield job.output.get(timeout=0.05)
                except queue.Empty:
                    if job.terminal.done():
                        job.terminal.result()
                        break
        finally:
            job.cancel.set()
            with self._condition:
                self._condition.notify()

    def schedule(self, factory, *, admission=None, on_cancel=None, prefix_preview=None):
        job = self._enqueue(factory, admission=admission, on_cancel=on_cancel,
                            prefix_preview=prefix_preview)
        return self._read(job)

    async def _read(self, job):
        def take():
            try:
                return True, job.output.get(timeout=0.05)
            except queue.Empty:
                return False, None
        try:
            while True:
                available, item = await asyncio.to_thread(take)
                if available:
                    yield item
                elif job.terminal.done():
                    job.terminal.result()
                    break
        finally:
            job.cancel.set()

    async def submit(self, factory, **kwargs):
        async for item in self.schedule(factory, **kwargs):
            yield item

    def execute(self, factory, *, apply_timeout=False):
        if threading.current_thread() is self._thread:
            return factory()
        call = _Call(factory)
        with self._condition:
            if self._closed:
                raise RuntimeError("inference actor is closed")
            self._pending.append(call)
            self._condition.notify()
        timeout = self.config.timeout if apply_timeout and self.config.timeout else None
        return call.result.result(timeout)

    def close(self, finalizer=None):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._finalizer = finalizer
            for job in self._active:
                job.cancel.set()
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _finish(self, job, error=None, phase=Phase.FINISHED):
        if job.terminal.done():
            return
        try:
            close = getattr(job.iterator, "close", None)
            if callable(close):
                close()
            if job.on_cancel is not None:
                job.on_cancel()
        except BaseException as cleanup_error:
            error = error or cleanup_error
            phase = Phase.FAILED
        if job.state is not None and not job.state.phase.terminal:
            job.state.error = error
            job.state.transition(phase)
        if job in self._active:
            self._active.remove(job)
        key = {Phase.FINISHED: "completed", Phase.CANCELLED: "cancelled",
               Phase.TIMED_OUT: "timed_out", Phase.FAILED: "failed"}[phase]
        self._counts[key] += 1
        if error is not None:
            job.terminal.set_exception(error)
        else:
            job.terminal.set_result(None)

    def _advance_one(self, job):
        if job.request is not None or job.output.full():
            return
        try:
            if job.iterator is None:
                if job.admission is not None and not job.admitted:
                    decision = job.admission()
                    status = getattr(decision, "status", decision)
                    status = str(getattr(status, "value", status)).lower()
                    if status == "queue":
                        self._defer(job)
                        return
                    if status in {"reject", "false"}:
                        raise CacheCapacityError(
                            getattr(decision, "reason", "admission rejected"), retryable=False)
                    job.admitted = True
                _slot_local.queue_ms = (time.monotonic() - job.created_at) * 1000
                job.iterator = iter(job.factory(job.cancel))
            # One item at a time keeps detokenization and output queues bounded.
            if job.reply is not None:
                reply, job.reply = job.reply, None
                item = job.iterator.send(reply)
            else:
                item = next(job.iterator)
            if isinstance(item, ForwardRequest):
                item.request_id = job.request_id
                item.model_epoch, item.cancel = self._epoch, job.cancel
                item.deadline = job.deadline
                job.request = item
                if job.state is None:
                    normalized = EngineRequest(
                        RequestKind.TEXT, item.bundle.model_id,
                        {"options": item.options, "input_ids": item.prompt_ids},
                        request_id=job.request_id, created_at=job.created_at,
                        deadline=job.deadline, max_tokens=getattr(item.options, "max_tokens", 0))
                    job.state = SequenceState(job.request_id, item.prompt_ids, len(item.prompt_ids),
                                              computed_len=item.cache[0].offset,
                                              phase=Phase.PREFILL, cache=item.cache,
                                              request=normalized,
                                              rng_seed=getattr(item.options, "seed", None),
                                              cancel=job.cancel, model_epoch=self._epoch,
                                              deadline_ns=(int(job.deadline * 1e9)
                                                           if job.deadline else None))
                if item.phase == "decode":
                    job.state.transition(Phase.DECODE)
                    job.state.generated_ids.append(item.token_ids[-1])
            else:
                job.output.put_nowait(item)
        except StopIteration:
            self._finish(job)
        except CacheCapacityError as exc:
            if exc.retryable and job.state is None:
                if job.iterator is not None:
                    job.iterator.close()
                job.iterator = None
                # Release the slot so other feasible queued requests can be considered.
                self._defer(job)
            else:
                self._finish(job, exc, Phase.FAILED)
        except BaseException as exc:
            self._finish(job, exc, Phase.FAILED)

    def _defer(self, job):
        self._skipped_admissions += 1
        job.retry_at = time.monotonic() + .01
        self._active.remove(job)
        with self._condition:
            boundary = next((i for i, work in enumerate(self._pending)
                             if isinstance(work, _Call)), len(self._pending))
            self._pending.insert(boundary, job)

    def _advance(self, job):
        for _ in range(self.config.output_buffer_events + 1):
            if job.terminal.done() or job.request is not None or job.output.full():
                return
            self._advance_one(job)
            if job.iterator is None:
                return

    def _admit(self):
        with self._condition:
            boundary = next((i for i, work in enumerate(self._pending) if isinstance(work, _Call)),
                            len(self._pending))
            candidates = list(self._pending)[:boundary]
            now = time.monotonic()
            def admission_order(job):
                if self.config.admission_policy == "cache_aware":
                    if job.deadline is not None and job.deadline - now <= .25:
                        return (0, job.deadline, job.created_at)
                    if now - job.created_at >= self.config.admission_age_s:
                        return (1, job.created_at, job.created_at)
                    if job.prefix_preview is not None:
                        started = time.perf_counter()
                        try:
                            job.preview_tokens = max(int(job.prefix_preview()), 0)
                        except (ValueError, RuntimeError):
                            job.preview_tokens = 0
                        finally:
                            job.preview_lookup_ms += (time.perf_counter() - started) * 1000
                    return (2, -job.preview_tokens, job.deadline or float("inf"), job.created_at)
                if now - job.created_at >= self.config.admission_age_s:
                    return (0, job.created_at)
                return (1, job.deadline or float("inf"), job.created_at)
            candidates.sort(key=admission_order)
            for job in candidates:
                if job.retry_at > now:
                    continue
                if len(self._active) >= self.config.max_sequences:
                    break
                self._pending.remove(job)
                self._active.append(job)
            if not self._active and self._pending and isinstance(self._pending[0], _Call):
                call = self._pending.popleft()
            else:
                call = None
        if call is not None:
            try:
                result = call.factory()
                self._epoch += 1
                call.result.set_result(result)
            except BaseException as exc:
                call.result.set_exception(exc)

    def _run(self):
        try:
            while not self._closed:
                if self.pressure is not None:
                    self.pressure.poll()
                    runtimes = {id(j.request.runtime): j.request.runtime for j in self._active
                                if j.request is not None and j.request.runtime is not None}
                    for runtime in runtimes.values():
                        self.pressure.shed(runtime)
                self._admit()
                now = time.monotonic()
                with self._condition:
                    pending = [j for j in self._pending if isinstance(j, _Job)]
                for job in [*self._active, *pending]:
                    if job.cancel.is_set() or (job.deadline is not None and now >= job.deadline):
                        if job in pending:
                            with self._condition:
                                self._pending.remove(job)
                        timed_out = not job.cancel.is_set()
                        job.cancel.set()
                        error = (RequestTimeoutError("generation request timed out")
                                 if timed_out else None)
                        self._finish(job, error,
                                     Phase.TIMED_OUT if timed_out else Phase.CANCELLED)
                for job in tuple(self._active):
                    if job.output.full():
                        job.stalled_at = job.stalled_at or now
                        if now - job.stalled_at >= self.config.output_stall_timeout_s:
                            job.cancel.set()
                            self._finish(job, TimeoutError("generation output consumer stalled"),
                                         Phase.TIMED_OUT)
                    else:
                        job.stalled_at = None
                        self._advance(job)
                ready = [job.request for job in self._active if job.request is not None
                         and not job.output.full()]
                # Different model bundles are kept in distinct microbatches.
                if ready:
                    ready = [request for request in ready if request.bundle is ready[0].bundle]
                try:
                    groups = self.policy.select(ready)
                except Exception as exc:
                    for job in tuple(self._active):
                        if any(job.request is request for request in ready):
                            self._finish(job, exc, Phase.FAILED)
                    groups = ()
                for group in groups:
                    if (self.pressure is not None and self.pressure.state in {"warning", "critical"}
                            and group[0].phase == "prefill"):
                        group = group[:1]
                    jobs = [next(j for j in self._active if j.request is request)
                            for request in group]
                    try:
                        output = self.runner.execute(group)
                        if output.request_ids != tuple(j.request_id for j in jobs):
                            raise ValueError("runner output mapping mismatch")
                        trace = dict(self.runner.last_trace)
                        trace["queue_age_ms"] = [(now - j.created_at) * 1000 for j in jobs]
                        trace["prefix_preview_tokens"] = [j.preview_tokens for j in jobs]
                        trace["prefix_preview_lookup_ms"] = [j.preview_lookup_ms for j in jobs]
                        trace["pressure_state"] = (self.pressure.state
                                                    if self.pressure is not None else None)
                        if self.capture_memory:
                            import mlx.core as mx

                            trace["observed_mlx_memory"] = {
                                "active_bytes": int(mx.get_active_memory()),
                                "cached_bytes": int(mx.get_cache_memory()),
                                "process_peak_bytes": int(mx.get_peak_memory())}
                            trace["host_telemetry"] = (self.pressure.stats()
                                                       if self.pressure is not None else None)
                        self._traces.append(trace)
                        if self._trace_sink is not None:
                            self._trace_sink(trace)
                        for job, logits, committed in zip(jobs, output.logits,
                                                          output.committed_token_counts,
                                                          strict=True):
                            if job.deadline is not None and time.monotonic() >= job.deadline:
                                job.cancel.set()
                                error = RequestTimeoutError("generation request timed out")
                                self._finish(job, error,
                                             Phase.TIMED_OUT)
                                continue
                            if job.cancel.is_set() or job.state.model_epoch != self._epoch:
                                self._finish(job, phase=Phase.CANCELLED)
                                continue
                            job.state.commit(committed)
                            job.request = None
                            job.reply = SimpleNamespace(
                                logits=logits, batch_size=trace["batch_size"],
                                forward_duration_ms=trace["duration_ms"])
                    except BaseException as exc:
                        for job in jobs:
                            self._finish(job, exc, Phase.FAILED)
                with self._condition:
                    self._condition.wait(timeout=0.001 if self._active or self._pending else 0.05)
        finally:
            for job in tuple(self._active):
                self._finish(job, phase=Phase.CANCELLED)
            with self._condition:
                pending = list(self._pending)
                self._pending.clear()
            for work in pending:
                if isinstance(work, _Call):
                    work.result.set_exception(RuntimeError("inference actor stopped"))
                else:
                    self._finish(work, RuntimeError("inference actor stopped"), Phase.CANCELLED)
            if self._finalizer is not None:
                self._finalizer()
