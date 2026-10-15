"""Single-flight model residency with epoch leases and draining lifecycle barriers."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.device_executor import DeviceExecutor


@dataclass(frozen=True)
class ModelRuntime:
    key: str
    epoch: int
    value: object
    resident_bytes: int


@dataclass
class _Resident:
    runtime: ModelRuntime
    dispose: object
    leases: int = 0
    pinned: bool = False
    used_at: float = 0
    drain: object = None


class ModelLease:
    def __init__(self, pool, entry):
        self._pool, self._entry = pool, entry

    @property
    def runtime(self):
        if self._entry is None:
            raise RuntimeError("model lease has been released")
        return self._entry.runtime

    def release(self):
        with self._pool._condition:
            if self._entry is not None:
                self._entry.leases -= 1
                self._entry.used_at = self._pool.clock()
                self._entry = None
                self._pool._condition.notify_all()

    def __enter__(self):
        return self.runtime

    def __exit__(self, *args):
        self.release()


class ModelPool:
    def __init__(self, *, executor=None, max_models=1, budget_bytes=None, idle_ttl=None,
                 clock=time.monotonic):
        if max_models < 1 or (budget_bytes is not None and budget_bytes < 1) or (
                idle_ttl is not None and idle_ttl < 0):
            raise ValueError("invalid model residency budget")
        self.executor = executor or DeviceExecutor.shared()
        self.max_models, self.budget_bytes, self.idle_ttl = max_models, budget_bytes, idle_ttl
        self.clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._entries, self._loading, self._epochs = {}, {}, {}
        self._retiring = {}
        self._draining = set()
        self._closed = False

    def _remove_idle(self, *, required_bytes=0, extra_slots=0, expired_only=False):
        """Caller holds lock. Remove ownership before device disposal outside lock."""
        removed = []
        now = self.clock()
        ordered = sorted(self._entries.items(), key=lambda pair: (pair[1].used_at, pair[0]))
        for key, entry in ordered:
            total = sum(e.runtime.resident_bytes for e in self._entries.values())
            total += sum(size for _, size in self._loading.values())
            other_retiring = [e for e in self._retiring.values()
                                 if id(e) not in {id(r) for r in removed}]
            total += sum(e.runtime.resident_bytes for e in other_retiring)
            fits = (len(self._entries) + len(self._loading) + len(other_retiring)
                    + extra_slots <= self.max_models
                    and (self.budget_bytes is None or total + required_bytes <= self.budget_bytes))
            expired = self.idle_ttl is not None and now - entry.used_at >= self.idle_ttl
            if entry.leases or entry.pinned or key in self._draining:
                continue
            if (expired_only and not expired) or (not expired_only and fits):
                continue
            removed.append(self._entries.pop(key))
            self._retiring[id(entry)] = entry
        return removed

    def _dispose(self, entries):
        error = None
        for entry in entries:
            try:
                if entry.drain is not None:
                    entry.drain(entry.runtime.value)
            except BaseException as exc:
                error = error or exc
            try:
                self.executor.call(entry.runtime.key, lambda e=entry: e.dispose(e.runtime.value))
            except BaseException as exc:
                error = error or exc
            finally:
                with self._condition:
                    self._retiring.pop(id(entry), None)
                    self._condition.notify_all()
        if error is not None:
            raise error

    def acquire(self, key, loader, *, estimated_bytes=0, dispose=lambda value: None,
                evaluate=None, measure=None, pinned=False, drain=None):
        if not key or not isinstance(estimated_bytes, int) or estimated_bytes < 0:
            raise ValueError("model requires identity and a nonnegative byte estimate")
        with self._condition:
            if self._closed or key in self._draining:
                raise RuntimeError("model pool is closed or model is draining")
            if key in self._entries:
                entry = self._entries[key]
                entry.leases += 1
                entry.pinned |= pinned
                return ModelLease(self, entry)
            if key in self._loading:
                future = self._loading[key][0]
                owner, removed = False, []
            else:
                removed = self._remove_idle(required_bytes=estimated_bytes, extra_slots=1)
                total = sum(e.runtime.resident_bytes for e in self._entries.values())
                total += sum(size for _, size in self._loading.values())
                other_retiring = [e for e in self._retiring.values()
                                 if id(e) not in {id(r) for r in removed}]
                total += sum(e.runtime.resident_bytes for e in other_retiring)
                if (len(self._entries) + len(self._loading) + len(other_retiring)
                        >= self.max_models) or (
                        self.budget_bytes is not None
                        and total + estimated_bytes > self.budget_bytes):
                    # Preserve removed entries until their releases have actually been attempted.
                    capacity_error = CacheCapacityError("model residency capacity exhausted")
                else:
                    capacity_error = None
                if capacity_error is None:
                    future = Future()
                    self._loading[key] = (future, estimated_bytes)
                owner = True
        if owner:
            if capacity_error is not None:
                self._dispose(removed)
                raise capacity_error
            value = None
            try:
                self._dispose(removed)
                value = self.executor.call(key, loader, evaluate=evaluate)
                observed = measure(value) if measure else estimated_bytes
                if not isinstance(observed, int) or observed < 0:
                    raise ValueError("observed model storage must be nonnegative bytes")
                with self._condition:
                    total = sum(e.runtime.resident_bytes for e in self._entries.values())
                    total += sum(size for name, (_, size) in self._loading.items() if name != key)
                    total += sum(e.runtime.resident_bytes for e in self._retiring.values())
                    if self._closed or (self.budget_bytes is not None
                                        and total + observed > self.budget_bytes):
                        raise CacheCapacityError("loaded model exceeds residency capacity")
                    epoch = self._epochs.get(key, 0) + 1
                    self._epochs[key] = epoch
                    entry = _Resident(ModelRuntime(key, epoch, value, observed), dispose,
                                      leases=1, pinned=pinned, used_at=self.clock(), drain=drain)
                    self._entries[key] = entry
                    del self._loading[key]
                    self._condition.notify_all()
                future.set_result(entry.runtime)
                return ModelLease(self, entry)
            except BaseException as exc:
                if value is not None:
                    try:
                        if drain is not None:
                            drain(value)
                    except BaseException as cleanup_error:
                        exc.__context__ = cleanup_error
                    try:
                        self.executor.call(key, lambda: dispose(value))
                    except BaseException as cleanup_error:
                        exc.__context__ = cleanup_error
                with self._condition:
                    loading = self._loading.pop(key, None)
                    self._condition.notify_all()
                if loading is not None:
                    loading[0].set_exception(exc)
                raise
        future.result()
        # Do not recurse into loading: acquire the exact published epoch or fail closed.
        with self._condition:
            if self._closed or key in self._draining or key not in self._entries:
                raise RuntimeError("loaded epoch was unloaded before lease acquisition")
            entry = self._entries[key]
            if entry.runtime is not future.result():
                raise RuntimeError("model epoch changed during loading")
            entry.leases += 1
            entry.pinned |= pinned
            return ModelLease(self, entry)

    def unload(self, key, *, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        owns_drain = False
        try:
            with self._condition:
                if key in self._draining:
                    raise RuntimeError("model is already draining")
                self._draining.add(key)
                owns_drain = True
                while key in self._loading or (key in self._entries and self._entries[key].leases):
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError("model unload is waiting for active leases")
                    self._condition.wait(remaining)
                entry = self._entries.pop(key, None)
                if entry is not None:
                    self._retiring[id(entry)] = entry
            if entry is not None:
                self._dispose([entry])
        finally:
            with self._condition:
                if owns_drain:
                    self._draining.discard(key)

    def evict_idle(self):
        with self._condition:
            removed = self._remove_idle(expired_only=True)
        self._dispose(removed)
        return len(removed)

    def pin(self, key, pinned=True):
        with self._condition:
            self._entries[key].pinned = pinned

    def stats(self):
        with self._condition:
            return {"resident_models": len(self._entries), "loading_models": len(self._loading),
                    "resident_bytes": sum(e.runtime.resident_bytes for e in self._entries.values()),
                    "reserved_bytes": sum(size for _, size in self._loading.values()),
                    "active_leases": sum(e.leases for e in self._entries.values())}

    def close(self, *, timeout=None):
        with self._condition:
            self._closed = True
            keys = set(self._entries) | set(self._loading)
        error = None
        for key in sorted(keys):
            try:
                self.unload(key, timeout=timeout)
            except BaseException as exc:
                error = error or exc
        if error is not None:
            raise error
