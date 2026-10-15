"""Ordered request-owned sessions retaining a model epoch until disposal."""
from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError

from mlx_one.engine.execution_context import RequestTimeoutError


class DeviceSession:
    def __init__(self,lease,executor,factory,*,methods,deadline=None,cancel=None):
        self.lease,self.executor=lease,executor
        self.methods=frozenset(methods)
        self.deadline=deadline
        self.cancel=cancel or threading.Event()
        self.lock=threading.RLock()
        self.closed=False
        self.sequence=0
        self.on_close=None
        self.key=('session',lease.runtime.key,lease.runtime.epoch,id(self))
        try:
            self.state=executor.call(self.key,factory,cancel=self.cancel)
        except BaseException:
            lease.release()
            raise

    def call(self,method,*args,event_number=None,**kwargs):
        with self.lock:
            if self.closed or self.cancel.is_set():
                raise CancelledError('session is closed or cancelled')
            if method not in self.methods:
                raise ValueError('session operation is unavailable')
            if event_number is not None and event_number!=self.sequence:
                raise ValueError('late or reordered session event')
            if self.deadline is not None and time.monotonic()>=self.deadline:
                self.close()
                raise RequestTimeoutError('session deadline exceeded')
            try:
                output=self.executor.call(self.key,
                    lambda:getattr(self.state,method)(*args,**kwargs),cancel=self.cancel)
                if self.cancel.is_set():
                    raise CancelledError('session cancelled during execution')
                self.sequence+=1
                return output
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed=True
            self.cancel.set()
            try:
                self.executor.call(self.key,self.state.close)
            finally:
                self.lease.release()
                if self.on_close is not None:
                    self.on_close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()
