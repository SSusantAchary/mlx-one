"""Bounded, fair, single-thread device ownership; construction is backend-free."""
from __future__ import annotations

import threading
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future

from mlx_one.engine.execution_context import QueueFullError


class DeviceExecutor:
    _shared = None
    _shared_lock = threading.Lock()

    def __init__(self, *, queue_size=256):
        if queue_size < 1:
            raise ValueError("device queue must be positive")
        self.queue_size = queue_size
        self._condition = threading.Condition()
        self._queues = OrderedDict()
        self._pending = 0
        self._closed = False
        self._dispatches = 0
        self._thread = threading.Thread(target=self._run, name="mlx-one-device", daemon=True)
        self._thread.start()

    @classmethod
    def shared(cls):
        with cls._shared_lock:
            if cls._shared is None or cls._shared._closed:
                cls._shared = cls()
            return cls._shared

    @property
    def is_owner(self):
        return threading.current_thread() is self._thread

    def submit(self, model_key, function, *, evaluate=None, cancel=None):
        future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("device executor is closed")
            if self._pending >= self.queue_size:
                raise QueueFullError("device execution queue is full")
            self._queues.setdefault(model_key, deque()).append((function, evaluate, cancel, future))
            self._pending += 1
            self._condition.notify()
        return future

    def call(self, model_key, function, *, evaluate=None, cancel=None):
        # Nested model operations already own the device; never deadlock the worker.
        if self.is_owner:
            if cancel is not None and cancel.is_set():
                raise CancelledError()
            value = function()
            if evaluate is not None:
                evaluate(value)
            return value
        return self.submit(model_key, function, evaluate=evaluate, cancel=cancel).result()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closed)
                if not self._pending and self._closed:
                    return
                key, items = self._queues.popitem(last=False)
                function, evaluate, cancel, future = items.popleft()
                if items:
                    self._queues[key] = items
                self._pending -= 1
            if not future.set_running_or_notify_cancel():
                continue
            try:
                if cancel is not None and cancel.is_set():
                    raise CancelledError()
                value = function()
                if evaluate is not None:
                    evaluate(value)
                # Transactions own commit/rollback. Cancellation cannot interrupt Metal.
                future.set_result(value)
            except BaseException as error:
                future.set_exception(error)
            finally:
                self._dispatches += 1

    def stats(self):
        with self._condition:
            return {"queued": self._pending, "queue_capacity": self.queue_size,
                    "dispatches": self._dispatches, "worker_alive": self._thread.is_alive()}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if not self.is_owner:
            self._thread.join()
