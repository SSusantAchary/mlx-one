"""Backend-free execution-contract tests for opt-in native gather attention."""
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.engine.kernel_dispatch import KernelDecision
from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
from mlx_one.ops import attention as ops
from mlx_one.ops import reference_tiled


class Observer:
    capture_math_dispatch = True
    profile_math = False

    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)


@pytest.fixture
def core(monkeypatch):
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    mx.gpu, mx.float32, mx.int32 = "gpu", np.float32, np.int32
    mx.default_device = lambda: "gpu"
    mx.metal = SimpleNamespace(is_available=lambda: True)
    calls = {"device_info": 0, "candidate": [], "reference": []}

    def device_info():
        calls["device_info"] += 1
        return {"device_name": "Apple M4"}

    mx.device_info = device_info
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setattr(ops, "mlx_version", lambda: (0, 32, 2))
    ops._native_gather_device_name.cache_clear()

    def candidate(q, k, v, **kwargs):
        calls["candidate"].append({"q": q.shape, "k": k.shape, "kwargs": kwargs})
        return "candidate-output"

    def reference(q, k, v, **kwargs):
        calls["reference"].append(kwargs)
        return "reference-output"

    monkeypatch.setattr(reference_tiled, "reference_compatible_attention", candidate)
    monkeypatch.setattr(ops, "attention_reference", reference)
    yield mx, calls
    ops._native_gather_device_name.cache_clear()


def operands(*, batch=1, heads=16, kv_heads=8, queries=7, keys=19,
             width=128, dtype=np.float16):
    return (np.ones((batch, heads, queries, width), dtype=dtype),
            np.ones((batch, kv_heads, keys, width), dtype=dtype),
            np.ones((batch, kv_heads, keys, width), dtype=dtype))


def policy():
    return NativeMathPolicy(attention="native_gather_tiled")


def test_new_policy_is_opt_in_and_never_self_qualifies():
    assert NativeMathPolicy().attention == "reference"
    assert NativeMathPolicy.optimized().attention == "reference"
    report = policy().to_dict()
    assert report["attention"] == "native_gather_tiled"
    assert report["attention_accumulation_operands"] == "float32"
    assert report["qualified"] is False


@pytest.mark.parametrize("queries,keys", [(1, 1), (1, 96), (7, 7), (7, 96), (64, 64), (64, 96)])
def test_measured_envelope_selects_before_stable_sdpa_guard(core, queries, keys):
    q, k, v = operands(queries=queries, keys=keys)
    mask = np.zeros((queries, keys), dtype=np.float32)
    with math_execution(policy(), family="qwen3"):
        decision = ops.select_attention(q, k, v, mask=mask, requires_stable_rounding=True)
    assert decision.selected == "native_gather_tiled"
    assert "checkpoint qualification separate" in decision.reason
    assert core[1]["candidate"] == core[1]["reference"] == []


def test_total_history_is_gated_before_cache_append(core):
    q, k, v = operands(queries=1, keys=1)
    with math_execution(policy(), family="qwen3"):
        accepted = ops.select_attention(q, k, v, history=95, requires_stable_rounding=True)
        rejected = ops.select_attention(q, k, v, history=96, requires_stable_rounding=True)
    assert accepted.selected == "native_gather_tiled"
    assert rejected.selected == "reference" and "Q<=K<=96" in rejected.reason
    assert k.shape[2] == v.shape[2] == 1


@pytest.mark.parametrize("change,reason", [
    ("version", "MLX 0.32.2"), ("metal", "Metal GPU"), ("cpu", "Metal GPU"),
    ("device", "Apple M4"), ("family", "stable INT8"), ("stable", "stable INT8"),
    ("dtype", "FP16 Q/K/V"), ("mixed", "FP16 Q/K/V"),
    ("batch", "B1 Hq16/Hkv8"), ("heads", "B1 Hq16/Hkv8"),
    ("width", "B1 Hq16/Hkv8"), ("queries", "Q{1,7,64}"),
    ("history", "Q<=K<=96"), ("value_width", "B1 Hq16/Hkv8"),
    ("bool_mask", "FP32 array mask"), ("half_mask", "FP32 array mask"),
    ("string_mask", "FP32 array mask"),
])
def test_unqualified_configurations_select_explicit_reference(core, monkeypatch, change, reason):
    mx, calls = core
    kwargs, family, stable, mask = {}, "qwen3", True, None
    if change == "version":
        monkeypatch.setattr(ops, "mlx_version", lambda: (0, 32, 3))
    elif change == "metal":
        mx.metal.is_available = lambda: False
    elif change == "cpu":
        mx.default_device = lambda: "cpu"
    elif change == "device":
        mx.device_info = lambda: {"device_name": "Apple M4 Pro"}
    elif change == "family":
        family = "qwen2"
    elif change == "stable":
        stable = False
    elif change == "dtype":
        kwargs["dtype"] = np.float32
    elif change == "batch":
        kwargs["batch"] = 2
    elif change == "heads":
        kwargs.update(heads=8, kv_heads=4)
    elif change == "width":
        kwargs["width"] = 64
    elif change == "queries":
        kwargs["queries"] = 8
    elif change == "history":
        kwargs["keys"] = 97
    q, k, v = operands(**kwargs)
    if change == "mixed":
        k = k.astype(np.float32)
    elif change == "value_width":
        v = v[..., :64]
    elif change == "bool_mask":
        mask = np.ones((q.shape[2], k.shape[2]), dtype=bool)
    elif change == "half_mask":
        mask = np.zeros((q.shape[2], k.shape[2]), dtype=np.float16)
    elif change == "string_mask":
        mask = "causal"
    with math_execution(policy(), family=family):
        decision = ops.select_attention(q, k, v, mask=mask, requires_stable_rounding=stable)
    assert decision.selected == "reference" and reason in decision.reason
    assert calls["candidate"] == []


def test_actual_backend_and_fp32_operands_are_reported(core):
    q, k, v = operands()
    observer = Observer()
    with math_execution(policy(), family="qwen3", observer=observer):
        result = ops.attention(q, k, v, scale=128**-0.5, stable=True, promote=True)
    assert result == "candidate-output" and not core[1]["reference"]
    assert core[1]["candidate"][0]["k"] == (1, 8, 19, 128)
    assert core[1]["candidate"][0]["kwargs"]["query_tile"] == 128
    event, = observer.events
    assert event["requested"] == event["selected"] == "native_gather_tiled"
    assert event["shape"]["native_sdpa"] is False
    assert event["shape"]["accumulation_operands"] == "float32"
    assert event["shape"]["query_tile"] == 128


@pytest.mark.parametrize("change", ["reference_policy", "cpu", "family", "batch", "stable",
                                    "scale"])
def test_supplied_candidate_decision_cannot_bypass_live_gate(core, change):
    mx, calls = core
    settings = {"stable": True, "promote": True, "scale": 128**-0.5}
    selected_policy, family = policy(), "qwen3"
    q, k, v = operands(batch=2 if change == "batch" else 1)
    if change == "reference_policy":
        selected_policy = NativeMathPolicy()
    elif change == "cpu":
        mx.default_device = lambda: "cpu"
    elif change == "family":
        family = "qwen2"
    elif change == "stable":
        settings["stable"] = False
    elif change == "scale":
        settings["scale"] = .125
    observer = Observer()
    with math_execution(selected_policy, family=family, observer=observer):
        result = ops.attention(q, k, v, decision=KernelDecision("native_gather_tiled", "supplied"),
                               **settings)
    assert result == "reference-output" and calls["candidate"] == []
    assert observer.events[0]["selected"] == "reference"
    assert "fallback" in observer.events[0]["reason"]
    assert calls["reference"][0]["stable"] is settings["stable"]


def test_executed_candidate_exception_is_never_retried(core, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected native gather failure")

    monkeypatch.setattr(reference_tiled, "reference_compatible_attention", fail)
    q, k, v = operands()
    with math_execution(policy(), family="qwen3"):
        with pytest.raises(RuntimeError, match="injected native gather"):
            ops.attention(q, k, v, scale=128**-0.5, stable=True, promote=True)
    assert core[1]["reference"] == []


def test_stock_sdpa_stable_rounding_guard_is_retained(core):
    q, k, v = operands()
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="qwen3"):
        decision = ops.select_attention(q, k, v, requires_stable_rounding=True)
        result = ops.attention(q, k, v, scale=128**-0.5, stable=True, promote=True,
                               decision=KernelDecision("mlx_sdpa", "supplied"))
    assert decision.selected == "reference"
    assert decision.reason == ops.STABLE_ROUNDING_REASON
    assert result == "reference-output" and core[1]["candidate"] == []


def test_actual_device_lookup_is_cached_but_active_device_is_checked_each_call(core):
    q, k, v = operands()
    with math_execution(policy(), family="qwen3"):
        for _ in range(2):
            decision = ops.select_attention(q, k, v, requires_stable_rounding=True)
            assert decision.selected == "native_gather_tiled"
        core[0].default_device = lambda: "cpu"
        assert ops.select_attention(q, k, v, requires_stable_rounding=True).selected == "reference"
    assert core[1]["device_info"] == 1
