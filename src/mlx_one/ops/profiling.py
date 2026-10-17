"""Optional synchronized profiling; the ordinary path does not read a clock."""
from __future__ import annotations

import time
from contextlib import contextmanager

from mlx_one.engine.optimization_policy import current_math


@contextmanager
def phase(name, *, output=None, observer=None):
    observer = current_math().observer if observer is None else observer
    if observer is None or not getattr(observer, "profile_math", False):
        yield
        return
    import mlx.core as mx

    mx.synchronize()
    started = time.perf_counter()
    try:
        yield
        if output is not None:
            mx.eval(output())
        mx.synchronize()
    finally:
        observer({"kind": "math_phase", "phase": name,
                  "duration_ms": (time.perf_counter() - started) * 1000,
                  "instrumented": True, "gpu_kernel_count": None,
                  "active_bytes": mx.get_active_memory(),
                  "cached_bytes": mx.get_cache_memory(), "peak_bytes": mx.get_peak_memory()})


def dispatch(operation, requested, selected, reason, **shape):
    observer = current_math().observer
    if observer is not None and getattr(observer, "capture_math_dispatch", False):
        observer({"kind": "math_dispatch", "operation": operation, "requested": requested,
                  "selected": selected, "reason": reason, "shape": shape})
