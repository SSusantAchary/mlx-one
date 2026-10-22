import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.ops.numerical_trace import (
    compare_tensors,
    compare_traces,
    criterion_for_dtype,
    emit,
    numerical_trace,
    tensor_metadata,
    trace_scope,
)


def test_disabled_trace_never_reads_or_evaluates_tensor():
    class Unevaluated:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled trace accessed {name}")

    emit(0, "input", Unevaluated())


def test_trace_bounds_layers_events_and_preserves_unevaluated_tensor():
    events, operand = [], object()
    with numerical_trace(events.append, max_layers=3, max_events=2) as state:
        emit(-1, "invalid", operand)
        emit(3, "outside", operand)
        emit(0, "input", operand)
        emit(2, "output", operand)
        emit(1, "over-budget", operand)
    assert state.emitted == 2 and state.dropped == 1
    assert [event["name"] for event in events] == ["input", "output"]
    assert events[0]["tensor"] is operand
    assert [event["sequence"] for event in events] == [0, 1]


@pytest.mark.parametrize("bounds", [{"max_layers": 0}, {"max_events": -1},
                                    {"max_events": True}, {"max_layers": 1.5}])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValueError, match="positive integers"):
        with numerical_trace(lambda event: None, **bounds):
            pass


def test_context_restores_after_exception_and_scopes_metadata():
    outer, inner = [], []
    with numerical_trace(outer.append, metadata={"execution": "reference"}):
        with trace_scope("full-prefill", tokens=64):
            emit(0, "input", None)
            with pytest.raises(RuntimeError), numerical_trace(inner.append):
                emit(0, "input", None)
                raise RuntimeError("injected")
            emit(0, "output", None)
        emit(0, "outside-forward", None)
    emit(0, "disabled", None)
    assert len(outer) == 3 and len(inner) == 1
    assert outer[0]["metadata"] == {
        "execution": "reference", "forward": "full-prefill", "tokens": 64}
    assert "forward" not in outer[-1]["metadata"]


def test_trace_context_does_not_leak_to_unrelated_thread():
    events = []
    with numerical_trace(events.append), ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(emit, 0, "unrelated", None).result()
        context = copy_context()
        executor.submit(context.run, emit, 0, "inherited", None).result()
    assert [event["name"] for event in events] == ["inherited"]


def test_callback_failure_is_not_silenced():
    def fail(event):
        raise RuntimeError("incomplete trace")

    with pytest.raises(RuntimeError, match="incomplete trace"), numerical_trace(fail):
        emit(0, "input", None)


@pytest.mark.parametrize("dtype,expected", [("float32", (1e-4, 1e-4)),
                                           ("float16", (1e-2, 1e-2)),
                                           ("bfloat16", (5e-2, 1e-2))])
def test_criterion_is_the_frozen_gate(dtype, expected):
    criterion = criterion_for_dtype(dtype)
    assert (criterion["atol"], criterion["rtol"]) == expected
    assert criterion["source"] == "mlx_one.inference_baseline.TOLERANCES"


def test_numerical_gate_counts_element_failures_and_relative_error():
    reference = np.array([0, 1, 2], dtype=np.float32)
    candidate = np.array([0, 1.0001, 2.01], dtype=np.float32)
    result = compare_tensors(reference, candidate)
    assert result["status"] == "FAIL"
    assert result["failing_elements"] == 1
    assert result["changed_elements"] == 2
    assert result["elements"] == 3
    assert result["max_abs_error"] == pytest.approx(.01, abs=1e-6)
    assert result["max_relative_error"] == pytest.approx(.005, abs=1e-6)
    assert result["mean_abs_error"] == pytest.approx(.0101 / 3, abs=1e-6)


def test_nonfinite_values_always_fail_even_when_equal():
    reference = np.array([1, np.inf, np.nan], dtype=np.float32)
    result = compare_tensors(reference, reference.copy())
    assert result["status"] == "FAIL"
    assert result["failing_elements"] == 2 and result["nonfinite_elements"] == 2


def test_shape_mismatch_cannot_broadcast_into_pass():
    result = compare_tensors(np.zeros((2, 1), np.float32), np.zeros((1, 2), np.float32))
    assert result["status"] == "FAIL" and result["reason"] == "shape mismatch"


def test_metadata_exactness_and_integer_precision():
    reference = np.array([2**63 - 2], dtype=np.int64)
    candidate = np.array([2**63 - 1], dtype=np.int64)
    assert compare_tensors(reference, candidate)["failing_elements"] == 1
    assert compare_tensors("causal", "causal")["status"] == "PASS"
    assert compare_tensors(None, None)["status"] == "PASS"


def test_layout_metadata_reports_actual_strides_and_unknown_stride_support():
    operand = np.arange(12, dtype=np.float32).reshape(3, 4).T
    assert tensor_metadata(operand) == {
        "shape": [4, 3], "dtype": "float32", "strides": [1, 4],
        "strides_unit": "elements", "strides_available": True}
    assert tensor_metadata(object())["strides_available"] is False
    result = compare_tensors(operand, operand.copy())
    assert result["status"] == "PASS" and result["strides_match"] is False


def test_replay_gate_is_explicit_despite_mixed_operand_dtypes():
    reference = np.array([1], dtype=np.float32)
    candidate = np.array([1.005], dtype=np.float16)
    assert compare_tensors(reference, candidate)["status"] == "FAIL"
    result = compare_tensors(reference, candidate, dtype="float16")
    assert result["status"] == "PASS" and result["dtype_match"] is False
    assert result["criterion"]["dtype"] == "float16"
    with pytest.raises(ValueError, match="no frozen"):
        compare_tensors(np.array([1.], dtype=np.float64), np.array([1.], dtype=np.float64))


def test_host_snapshot_conversion_preserves_original_bfloat16_gate():
    reference = np.array([1], dtype=np.float32)
    candidate = np.array([1.04], dtype=np.float32)
    metadata = {**tensor_metadata(reference), "dtype": "bfloat16"}
    result = compare_tensors(reference, candidate, reference_metadata=metadata)
    assert result["criterion"]["dtype"] == "bfloat16" and result["status"] == "PASS"


def test_float16_ulp_reports_single_rounding_steps_and_equal_signed_zero():
    reference = np.array([1, -1, 0], dtype=np.float16)
    candidate = np.array([np.nextafter(np.float16(1), np.float16(2)),
                          np.nextafter(np.float16(-1), np.float16(-2)), -0.], dtype=np.float16)
    result = compare_tensors(reference, candidate)
    assert result["status"] == "PASS" and result["changed_elements"] == 2
    assert result["max_float16_ulp"] == 1 and result["float16_ulp_elements"] == 3
    assert compare_tensors(np.array([0], np.float16), np.array([-0.], np.float16))[
        "max_float16_ulp"] == 0


def test_float16_cast_pair_ulp_can_be_requested_for_float32_replay():
    reference = np.array([1], dtype=np.float32)
    candidate = np.array([1.00097], dtype=np.float32)
    assert compare_tensors(reference, candidate)["max_float16_ulp"] is None
    assert compare_tensors(reference, candidate, float16_ulp=True)["max_float16_ulp"] == 1


def _event(sequence, name, value, *, layer=0):
    return {"sequence": sequence, "layer": layer, "name": name,
            "tensor": np.array([value], dtype=np.float32),
            "metadata": {"forward": "prefill"}}


def test_trace_identifies_earliest_tensor_and_compares_all_following_events():
    reference = [_event(0, "input", 1), _event(1, "attention", 1),
                 _event(2, "key", 1, layer=2)]
    candidate = [_event(0, "input", 1), _event(1, "attention", 1.01),
                 _event(2, "key", 1.2, layer=2)]
    report = compare_traces(reference, candidate)
    assert report["comparison_count"] == 3 and report["failing_comparisons"] == 2
    assert report["earliest_violation"]["name"] == "attention"
    assert report["earliest_violation"]["criterion"]["dtype"] == "float32"


def test_missing_unexpected_and_repeated_trace_events_cannot_pass():
    reference = [_event(0, "input", 1), _event(1, "input", 2), _event(2, "key", 1)]
    candidate = [_event(0, "input", 1), _event(1, "other", 1)]
    report = compare_traces(reference, candidate)
    assert report["status"] == "FAIL" and report["failing_comparisons"] == 3
    assert report["reference_event_count"] == 3 and report["candidate_event_count"] == 2
    assert report["earliest_violation"]["reason"] == "missing candidate event"
    assert report["earliest_violation"]["occurrence"] == 1


def test_unexecuted_trace_is_not_a_pass():
    result = compare_traces([], [])
    assert result["status"] == "FAIL"
    assert result["earliest_violation"]["reason"] == "no captured reference events"


@pytest.fixture
def transformer(monkeypatch):
    """Execute the real shared Transformer against deterministic CPU primitives.

    Packed-weight arithmetic has separate independent tests. These fixtures check
    that diagnostic hooks preserve the model's existing arithmetic and state flow.
    """
    core, nn, package = ModuleType("mlx.core"), ModuleType("mlx.nn"), ModuleType("mlx")
    for name in ("arange", "zeros", "concatenate", "repeat", "power", "broadcast_to",
                 "cos", "sin", "where", "triu"):
        setattr(core, name, getattr(np, name))
    core.float32, core.float16, core.int32 = np.float32, np.float16, np.int32
    core.full = lambda shape, value: np.full(shape, value, dtype=np.float32)
    core.sigmoid = lambda value: 1 / (1 + np.exp(-value))
    core.gather_calls = []

    def gather_mm(left, right, *, lhs_indices, rhs_indices):
        core.gather_calls.append((left.shape, right.shape))
        return np.stack([left[i] @ right[j] for i, j in
                         zip(lhs_indices, rhs_indices, strict=True)])

    def gather_qmm(rows, weights, **options):
        return (rows.astype(np.float32) @ weights.astype(np.float32).T).astype(np.float16)

    def softmax(value, *, axis, precise):
        assert precise is True
        probability = np.exp(value - value.max(axis=axis, keepdims=True))
        return probability / probability.sum(axis=axis, keepdims=True)

    def no_eval(*args):
        raise AssertionError("model trace hooks must not evaluate arrays")

    core.gather_mm, core.gather_qmm = gather_mm, gather_qmm
    core.softmax, core.eval = softmax, no_eval

    class Module:
        pass

    class Linear(dict):
        bits, group_size, mode = 8, 64, "affine"

        def __init__(self, inputs, outputs, *, bias=False):
            super().__init__()
            values = np.arange(outputs * inputs, dtype=np.float32).reshape(outputs, inputs)
            self.weight = ((values % 11 - 5) / 32).astype(np.float16)
            self.scales = np.ones((outputs, 1), np.float16)
            self.biases = np.zeros((outputs, 1), np.float16)
            if bias:
                self.bias = np.zeros((outputs,), np.float16)
                self["bias"] = self.bias

        def __call__(self, value):
            result = value @ self.weight.T
            return result + self.bias if "bias" in self else result

    class RMSNorm:
        def __init__(self, dimensions, *, eps):
            self.eps = eps

        def __call__(self, value):
            promoted = value.astype(np.float32)
            normalized = promoted / np.sqrt(np.mean(promoted**2, axis=-1, keepdims=True)
                                            + self.eps)
            return normalized.astype(value.dtype)

    nn.Module, nn.Linear, nn.RMSNorm = Module, Linear, RMSNorm
    package.core, package.nn = core, nn
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    # A separate module object avoids changing cached backend bindings in other tests.
    name = "mlx_one.models.shared._numerical_trace_test_transformer"
    path = Path(__file__).parents[2] / "src/mlx_one/models/shared/transformer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    module.test_core = core
    return module


def _decoder(transformer, index=None):
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=4, rope_theta=10000, rms_norm_eps=1e-5,
                             intermediate_size=12)
    layer = transformer.DecoderLayer(config, qkv_bias=False, qk_norm=True)
    layer.self_attn._mlx_one_activation_dtype = "float16"
    if index is not None:
        layer._mlx_one_trace_layer = index
        layer.self_attn._mlx_one_trace_layer = index
    return layer


def _snapshot(events):
    def capture(event):
        tensor = event["tensor"]
        events.append({**event, "tensor": tensor.copy() if hasattr(tensor, "copy") else tensor,
                       "tensor_metadata": tensor_metadata(tensor)})
    return capture


def test_real_decoder_hooks_preserve_reference_arithmetic_and_cache_state(transformer):
    from mlx_one.core.cache import KVCache

    layer = _decoder(transformer, 0)
    inputs = np.arange(24, dtype=np.float16).reshape(1, 3, 8) / np.float16(32)
    plain_cache, traced_cache = KVCache(), KVCache()
    expected = layer(inputs, mask="causal", cache=plain_cache)
    events = []
    with numerical_trace(_snapshot(events)), trace_scope("full-prefill"):
        actual = layer(inputs, mask="causal", cache=traced_cache)
    np.testing.assert_array_equal(actual, expected)
    for plain, traced in zip(plain_cache.state(), traced_cache.state(), strict=True):
        np.testing.assert_array_equal(plain, traced)
    assert plain_cache.offset == traced_cache.offset == 3
    by_name = {event["name"]: event for event in events}
    assert {"hidden_input", "q_projected", "k_projected", "v_projected", "q_normalized",
            "k_normalized", "q_rope", "k_rope", "q_operand", "k_operand", "v_operand",
            "attention_mask", "positions", "cache_before_0", "cache_before_1",
            "k_committed", "v_committed", "attention_output", "attention_output_cast",
            "attention_projected", "attention_residual", "mlp_output", "hidden_output"} <= set(
                by_name)
    assert by_name["cache_before_0"]["tensor"] is None
    assert by_name["k_committed"]["metadata"]["offset"] == 3
    assert by_name["positions"]["metadata"]["effective_positions"] == [0, 1, 2]
    assert by_name["attention_output"]["metadata"]["selected"] == "reference"
    assert by_name["attention_output"]["tensor"].dtype == np.float32
    assert by_name["attention_output_cast"]["tensor"].dtype == np.float16
    assert transformer.test_core.gather_calls
    assert all(left[1] == 1 for left, right in transformer.test_core.gather_calls)


def test_real_decoder_continuation_trace_captures_precommit_and_committed_offsets(transformer):
    from mlx_one.core.cache import KVCache

    layer, cache = _decoder(transformer, 1), KVCache()
    layer(np.ones((1, 3, 8), np.float16), mask="causal", cache=cache)
    previous = tuple(component.copy() for component in cache.state())
    events = []
    with numerical_trace(_snapshot(events)), trace_scope("continuation-0"):
        layer(np.ones((1, 1, 8), np.float16), mask=None, cache=cache)
    by_name = {event["name"]: event for event in events}
    for component in range(2):
        before = by_name[f"cache_before_{component}"]
        np.testing.assert_array_equal(before["tensor"], previous[component])
        assert before["metadata"]["offset"] == 3
    assert by_name["k_committed"]["tensor"].shape[2] == 4
    assert by_name["k_committed"]["metadata"]["offset"] == cache.offset == 4
    assert by_name["positions"]["metadata"]["effective_positions"] == [3]
    assert {event["metadata"]["forward"] for event in events} == {"continuation-0"}


def test_real_transformer_tracing_requires_opt_in_attributes_and_bounds_layers(transformer):
    from mlx_one.core.cache import KVCache

    inputs, events = np.ones((1, 2, 8), np.float16), []
    with numerical_trace(_snapshot(events)):
        _decoder(transformer)(inputs, mask="causal", cache=KVCache())
    assert events == []
    with numerical_trace(_snapshot(events), max_layers=3):
        for index in range(4):
            inputs = _decoder(transformer, index)(inputs, mask="causal", cache=KVCache())
    assert {event["layer"] for event in events} == {0, 1, 2}
    bounded = []
    with numerical_trace(_snapshot(bounded), max_events=2) as state:
        _decoder(transformer, 0)(inputs, mask="causal", cache=KVCache())
    assert len(bounded) == 2 and state.dropped > 0


def test_real_stable_int8_decoder_selects_safe_route_before_cache_append(transformer, monkeypatch):
    from mlx_one.core.cache import KVCache
    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.ops.attention import STABLE_ROUNDING_REASON

    order = []

    class Cache(KVCache):
        def update(self, keys, values):
            order.append(("append", self.offset))
            return super().update(keys, values)

    cache = Cache()
    cache.update(np.zeros((1, 1, 2, 4), np.float16), np.zeros((1, 1, 2, 4), np.float16))
    order.clear()
    selector = transformer.select_attention

    def select(*args, **kwargs):
        assert cache.offset == 2
        assert kwargs["requires_stable_rounding"] is True
        decision = selector(*args, **kwargs)
        assert decision.selected == "reference" and decision.reason == STABLE_ROUNDING_REASON
        order.append(("select", cache.offset))
        return decision

    monkeypatch.setattr(transformer, "select_attention", select)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        _decoder(transformer, 0)(np.ones((1, 2, 8), np.float16), mask="causal", cache=cache)
    assert order == [("select", 2), ("append", 2)]
    assert cache.offset == 4
