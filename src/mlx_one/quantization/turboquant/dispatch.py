"""Deterministic dispatch; measurements and tuning rationale live with benchmarks."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar

_stats = ContextVar("tq3_dispatch", default=None)


def record(name):
    stats = _stats.get()
    if stats is not None:
        stats[name] += 1


@contextmanager
def collect_dispatch():
    stats = Counter()
    token = _stats.set(stats)
    try:
        yield stats
    finally:
        _stats.reset(token)


def dense_kernel(rows, outputs):
    if rows == 1:
        return "qmv"
    if rows == 2 and outputs <= 2880:
        return "qmv_rows"
    return "qmm"


def expert_kernel(routings, outputs):
    return "gather_qmm" if routings >= (128 if outputs >= 512 else 512) else "gather_qmv"
