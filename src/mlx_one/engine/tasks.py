"""Bounded task microbatches attached to the existing inference actor/device owner."""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import RequestPhase, TaskKind
from mlx_one.engine.execution_context import QueueFullError, RequestTimeoutError


@dataclass(frozen=True)
class TaskResult:
    request_id: str
    task: TaskKind
    output: object
    metadata: dict


@dataclass(frozen=True)
class PreparedTask:
    task: TaskKind
    payload: object
    lengths: tuple[int, ...]
    semantic_key: tuple = ()
    transient_bytes: int = 0

    def __post_init__(self):
        if (not self.lengths or any(n < 1 for n in self.lengths) or self.transient_bytes < 0):
            raise ValueError('task requires nonempty inputs and valid memory estimates')


@dataclass(eq=False)
class TaskHandle:
    request_id: str
    cancel_event: threading.Event
    future: Future = field(default_factory=Future)
    phase: RequestPhase = RequestPhase.QUEUED

    def transition(self, phase):
        self.phase.validate_transition(phase)
        self.phase=phase

    def cancel(self):
        self.cancel_event.set()
        return not self.future.done()

    def result(self, timeout=None):
        return self.future.result(timeout)


@dataclass(eq=False)
class _TaskJob:
    adapter: object
    prepared: PreparedTask
    lease: object
    handle: TaskHandle
    deadline: float | None
    created: float = field(default_factory=time.monotonic)

    @property
    def key(self):
        return (self.lease.runtime.key,self.lease.runtime.epoch,id(self.adapter.bundle),
                self.prepared.task,self.prepared.semantic_key)


class TaskMicrobatches:
    def __init__(self, actor, *, transient_budget_bytes=256*1024**2):
        self.actor=actor
        self.budget=transient_budget_bytes
        self.condition=threading.Condition()
        self.pending=deque()
        self.active=[]
        self.closed=False
        self.traces=deque(maxlen=256)
        self.thread=threading.Thread(target=self._run,name='mlx-one-task-work',daemon=True)
        self.thread.start()

    def submit(self, adapter, prepared, lease, *, cancel=None, deadline=None):
        cfg=self.actor.config
        handle=TaskHandle('task-'+uuid.uuid4().hex,
                          cancel if cancel is not None else threading.Event())
        try:
            if deadline is not None and not math.isfinite(deadline):
                raise ValueError('task deadline must be finite')
            if (max(prepared.lengths)*len(prepared.lengths)>cfg.max_scheduled_tokens
                    or len(prepared.lengths)>cfg.max_sequences
                    or prepared.transient_bytes>self.budget):
                raise CacheCapacityError('task exceeds configured execution bounds',retryable=False)
            with self.condition:
                if self.closed:
                    raise RuntimeError('task actor is closed')
                if len(self.pending)+len(self.active)>=cfg.queue_size+cfg.max_sequences:
                    raise QueueFullError('task execution queue is full')
                handle.transition(RequestPhase.PREPARING)
                handle.transition(RequestPhase.ADMITTED)
                if cfg.timeout:
                    limit=time.monotonic()+cfg.timeout
                    deadline=min(deadline,limit) if deadline is not None else limit
                self.pending.append(_TaskJob(adapter,prepared,lease,handle,deadline))
                self.condition.notify()
            return handle
        except BaseException:
            lease.release()
            raise

    def _finish(self,job,output=None,error=None):
        try:
            if not job.handle.phase.terminal:
                phase=(RequestPhase.CANCELLED if isinstance(error,CancelledError) else
                       RequestPhase.TIMED_OUT if isinstance(error,RequestTimeoutError) else
                       RequestPhase.FAILED if error else RequestPhase.FINISHED)
                job.handle.transition(phase)
                if job.handle.future.done():
                    return
                if error:
                    job.handle.future.set_exception(error)
                else:
                    job.handle.future.set_result(TaskResult(job.handle.request_id,job.prepared.task,
                        output,{'model_epoch':job.lease.runtime.epoch,'qualification':'unqualified'}))
        finally:
            job.lease.release()

    def _error(self,job):
        if self.closed or job.handle.cancel_event.is_set():
            return CancelledError('task cancelled')
        if job.deadline is not None and time.monotonic()>=job.deadline:
            return RequestTimeoutError('task request timed out')
        return None

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.closed or self.pending)
                if self.closed and not self.pending:
                    return
                first=self.pending.popleft()
                selected=[first]
                for candidate in list(self.pending):
                    group=[*selected,candidate]
                    rows=sum(len(j.prepared.lengths) for j in group)
                    padded=rows*max(n for j in group for n in j.prepared.lengths)
                    if (candidate.key==first.key and rows<=self.actor.config.max_sequences
                            and padded<=self.actor.config.max_scheduled_tokens
                            and sum(j.prepared.transient_bytes for j in group)<=self.budget):
                        self.pending.remove(candidate)
                        selected.append(candidate)
                self.active=selected
            live=[]
            for job in selected:
                error=self._error(job)
                if error:
                    self._finish(job,error=error)
                else:
                    live.append(job)
            if live:
                started=time.monotonic()
                try:
                    for job in live:
                        job.handle.transition(RequestPhase.RUNNING)
                    result=self.actor.device_executor.call(live[0].lease.runtime.key,
                        lambda jobs=tuple(live):jobs[0].adapter.run_batch(
                            tuple(j.prepared for j in jobs)))
                    if len(result)!=len(live):
                        raise ValueError('task result mapping mismatch')
                    for job,output in zip(live,result,strict=True):
                        self._finish(job,output,error=self._error(job))
                except BaseException as exc:
                    for job in live:
                        if not job.handle.phase.terminal:
                            self._finish(job,error=exc)
                self.traces.append({'task':live[0].prepared.task.value,'requests':len(live),
                    'rows':sum(len(j.prepared.lengths) for j in live),
                    'real_inputs':sum(sum(j.prepared.lengths) for j in live),
                    'forward_ms':(time.monotonic()-started)*1000})
            with self.condition:
                self.active=[]

    def close(self):
        with self.condition:
            self.closed=True
            for job in [*self.pending,*self.active]:
                job.handle.cancel_event.set()
            self.condition.notify_all()
        if threading.current_thread() is not self.thread:
            self.thread.join()
