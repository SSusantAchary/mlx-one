import json
import subprocess
import sys

import pytest

from mlx_one.mlxboost.native.profiling import benchmark, export_report, summary


def test_backend_free_import_and_status_without_probe():
    code = """
import importlib.abc, sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'torch', 'transformers'}:
            raise AssertionError('eager backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
import mlx_one
import mlx_one.mlxboost
from mlx_one.mlxboost.native import native_status
from mlx_one.mlxboost.native import dispatcher, registry, profiling, autotune
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as pool:
    list(pool.map(lambda _: __import__('mlx_one.mlxboost.native'), range(100)))
assert 'mlx.core' not in sys.modules
print(native_status(probe=False))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class FakeMLX:
    def __init__(self):
        self.events = []
        self.pending = False

    def synchronize(self):
        self.events.append("sync")
        self.pending = False

    def eval(self, value):
        self.events.append("eval")
        assert value == "lazy tensor"
        self.pending = True

    def reset_peak_memory(self):
        self.events.append("reset_peak")

    def get_peak_memory(self):
        assert not self.pending
        return 7

    get_active_memory = get_peak_memory
    get_cache_memory = get_peak_memory


def test_timer_evaluates_and_synchronizes_every_output(monkeypatch):
    mx = FakeMLX()
    clock = 0

    def now():
        nonlocal clock
        # Completion must precede the clock read at the end of each sample.
        assert not mx.pending
        assert mx.events[-1] == "sync"
        clock += 1000000
        return clock

    monkeypatch.setattr("mlx_one.mlxboost.native.profiling.time.perf_counter_ns", now)
    profiles = benchmark(
        {"ref": lambda: "lazy tensor", "native": lambda: "lazy tensor"}, mx=mx, warmup=2, trials=3
    )
    assert mx.events.count("eval") == 12
    for profile in profiles.values():
        assert profile["samples_ms"] == [1, 1, 1]
        assert len(profile["warmup_samples_ms"]) == 2
        assert profile["cold_first_use_ms"] == 1
        assert profile["memory"]["mlx_peak_bytes"] == 7
        assert profile["timing_scope"] == "synchronized_wall"


def test_json_markdown_and_summary(tmp_path):
    record = {
        "schema_version": 1,
        "profiles": benchmark({"ref": lambda: "lazy tensor"}, mx=FakeMLX(), warmup=0, trials=3),
    }
    path = export_report(tmp_path, record)
    assert json.loads(path.read_text()) == record
    assert "No model speedup claim" in path.with_suffix(".md").read_text()
    assert export_report(tmp_path, record) == path
    assert summary([1, 2, 3])["median_ms"] == 2
    with pytest.raises(ValueError):
        benchmark({"ref": lambda: None}, trials=0, mx=FakeMLX())


def test_capture_is_explicit_and_tuning_budgets_precede_gpu(monkeypatch, tmp_path):
    from mlx_one.mlxboost.native.autotune import tune
    from mlx_one.mlxboost.native.profiling import metal_capture

    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="MTL_CAPTURE_ENABLED"):
        with metal_capture(tmp_path / "capture"):
            pass
    with pytest.raises(ValueError, match="memory budget"):
        tune(tmp_path, length=10000, memory_budget_bytes=100)
