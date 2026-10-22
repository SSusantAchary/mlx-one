import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
from mlx_one.ops.compiled import CompiledMathCache
from mlx_one.ops.profiling import phase


def test_throughput_path_has_no_profiling_clocks_sync_or_array_evaluation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("uninstrumented execution invoked profiler")
    monkeypatch.setattr("mlx_one.ops.profiling.time.perf_counter", forbidden)
    # No MLX import is needed at all for the nonprofiling path.
    with math_execution(NativeMathPolicy(), observer=SimpleNamespace(profile_math=False)):
        with phase("attention_core", output=forbidden):
            value = 3
    assert value == 3


def test_profile_boundaries_are_synchronized_and_not_reported_as_kernel_counts(monkeypatch):
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    actions, events = [], []
    mx.synchronize = lambda: actions.append("sync")
    mx.eval = lambda result: actions.append(("eval", result))
    mx.get_active_memory = mx.get_cache_memory = mx.get_peak_memory = lambda: 12
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    class Observer:
        profile_math = True
        def __call__(self, event):
            events.append(event)
    with math_execution(NativeMathPolicy(), observer=Observer()):
        with phase("mlp", output=lambda: 7):
            pass
    assert actions == ["sync", ("eval", 7), "sync"]
    assert events[0]["instrumented"] is True
    assert events[0]["gpu_kernel_count"] is None


def test_compiled_cache_bounded_and_weights_remain_dynamic(monkeypatch):
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    compiled = []
    def compile_op(function, *, shapeless):
        assert not shapeless
        compiled.append(function)
        return function
    mx.compile = compile_op
    mx.default_device = lambda: "gpu"
    mx.fast = SimpleNamespace(rms_norm=lambda x, w, eps:
        x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w)
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    cache = CompiledMathCache(limit=2)
    first = np.ones((1, 4), dtype=np.float32)
    weight = np.ones((4,), dtype=np.float32)
    a = cache.rms_norm(first, weight, eps=1e-5)
    b = cache.rms_norm(first, weight * 3, eps=1e-5)
    np.testing.assert_allclose(b, a * 3)
    assert len(compiled) == 1
    cache.rms_norm(np.ones((2, 4), dtype=np.float32), weight, eps=1e-5)
    cache.rms_norm(np.ones((3, 4), dtype=np.float32), weight, eps=1e-5)
    assert cache.stats()["entries"] == 2 and cache.stats()["evictions"] == 1
    assert cache.stats()["qualified"] is False
    assert cache.stats()["native_graph_memory_reconciled"] is False
    cache.clear()
    assert cache.stats()["entries"] == 0
    with pytest.raises(ValueError):
        CompiledMathCache(limit=True)
