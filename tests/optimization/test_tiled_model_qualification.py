"""Backend-free gate tests; the fake model does not qualify actual hardware."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_reference_tiled_attention import candidate as candidate_fixture

from mlx_one.core.cache import KVCache
from mlx_one.engine.kernel_dispatch import KernelDecision
from mlx_one.ops.attention import attention_reference


@pytest.fixture
def qualification():
    path = Path(__file__).parents[2] / "scripts/qualification/tiled_model_qualification.py"
    spec = importlib.util.spec_from_file_location("tiled_model_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_requires_both_frozen_fp32_and_strict_fp16_gates(qualification):
    # Equal FP16 bits alone can conceal a violation of the tighter FP32 gate.
    left, right = np.array([1000], np.float32), np.array([1000.24], np.float32)
    result = qualification.boundary_metrics(left, right)
    assert result["fp16_bit_unequal"] == 0
    assert result["fp32_within_tolerance"] is False and result["status"] == "FAIL"
    assert result["max_abs_error"] == pytest.approx(.24, abs=1e-4)
    assert result["fp32"]["criterion"]["atol"] == 1e-4
    assert result["fp32"]["criterion"]["rtol"] == 1e-4
    # A single FP16 ULP also fails, even though the raw FP32 delta is allowed.
    left, right = np.array([1.00048], np.float32), np.array([1.00049], np.float32)
    result = qualification.boundary_metrics(left, right)
    assert result["fp32_within_tolerance"] is True and result["status"] == "FAIL"
    assert result["fp16_numeric_unequal"] == result["fp16_bit_unequal"] == 1
    assert result["max_float16_ulp"] == 1


def test_signed_zero_is_reported_separately_and_fails_strict_bits(qualification):
    left, right = np.array([0.], np.float32), np.array([-0.], np.float32)
    result = qualification.boundary_metrics(left, right)
    assert result["status"] == "FAIL" and result["fp32_within_tolerance"]
    assert result["fp16_numeric_unequal"] == 0 and result["fp16_bit_unequal"] == 1
    assert result["signed_zero_bit_unequal"] == 1 and result["max_float16_ulp"] == 0


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, 1e10])
def test_nonfinite_or_fp16_overflow_is_never_accepted(qualification, value):
    operand = np.array([value], np.float32)
    assert qualification.boundary_metrics(operand, operand)["status"] == "FAIL"


def test_empty_and_mismatched_outputs_do_not_pass(qualification):
    assert qualification.boundary_metrics(np.array([], np.float32),
                                         np.array([], np.float32))["status"] == "FAIL"
    assert qualification.boundary_metrics(np.zeros(1, np.float32),
                                         np.zeros(2, np.float32))["status"] == "FAIL"
    assert qualification.boundary_metrics(np.zeros(2, np.float32),
                                         np.zeros(2, np.float32))["status"] == "PASS"


@pytest.fixture
def model_run(monkeypatch):
    run, core, calls = candidate_fixture.__wrapped__(monkeypatch)
    core.array, core.zeros = np.array, np.zeros
    core.eval = lambda *values: None
    core.clear_cache = lambda: None
    transformer = ModuleType("mlx_one.models.shared.transformer")
    transformer.forced_actual_selected = None
    transformer.suppress_actual_dispatch = False

    def original(q, k, v, **options):
        from mlx_one.engine.optimization_policy import current_math
        from mlx_one.ops.profiling import dispatch

        decision = options.pop("decision", None)
        selected = transformer.forced_actual_selected or getattr(decision, "selected", "reference")
        if not transformer.suppress_actual_dispatch:
            dispatch("attention", current_math().policy.attention, selected,
                     "fixture actual dispatcher revalidation", q=list(q.shape), k=list(k.shape),
                     accumulation_operands="float32", query_tile=128, native_sdpa=False)
        if selected == "native_gather_tiled":
            return run(q, k, v, options["scale"], mask=options.get("mask"), query_tile=128)
        saved, core.repeat = core.repeat, np.repeat
        try:
            return attention_reference(q, k, v, **options)
        finally:
            core.repeat = saved

    transformer.attention = original
    import mlx_one.models.shared as shared

    monkeypatch.setitem(sys.modules, "mlx_one.models.shared.transformer", transformer)
    monkeypatch.setattr(shared, "transformer", transformer, raising=False)

    class Model:
        def __init__(self):
            self.created = []
            self.forced_preappend_selected = None
            self.use_array_mask = False

        def make_cache(self):
            cache = [KVCache() for _ in range(28)]
            self.created.append(cache)
            return cache

        def __call__(self, ids, *, cache):
            from mlx_one.engine.optimization_policy import current_math

            batch, length = ids.shape
            hidden = np.broadcast_to(ids[..., None] / np.float32(32),
                                     (batch, length, 8)).astype(np.float16)
            for state in cache:
                q = hidden.reshape(batch, length, 2, 4).transpose(0, 2, 1, 3)
                k = q[:, :1].copy()
                v = (k * np.float16(.25) + np.float16(.125)).astype(np.float16)
                selected = ("native_gather_tiled" if current_math().policy.attention
                            == "native_gather_tiled" else "reference")
                decision = KernelDecision(self.forced_preappend_selected or selected,
                                          "fixture preappend selection")
                mask = (np.triu(np.full((length, state.offset + length), -np.inf, np.float32),
                                k=state.offset + 1) if self.use_array_mask else "causal")
                k, v = state.update(k, v)
                output = transformer.attention(q, k, v, scale=.5, mask=mask, stable=True,
                    promote=True, decision=decision)
                hidden = output.transpose(0, 2, 1, 3).reshape(batch, length, 8).astype(np.float16)
            logits = hidden.sum(-1, keepdims=True) + np.arange(16, dtype=np.float16)
            return SimpleNamespace(logits=logits)

    bundle = SimpleNamespace(model=Model(), architecture="qwen3", tokenizer=SimpleNamespace(
        encode=lambda text: [3 + i % 17 for i in range(128)]))
    return bundle, run, core, transformer, original


def test_fake_model_exercises_all_original_and_boundary_gates(qualification, model_run):
    bundle, run, core, transformer, original = model_run
    report = qualification.qualify_model(bundle, run, 7, {})
    assert report["status"] == "PASS", report.get("exception")
    assert report["comparison_count"] == 3818 and report["greedy_passed"] == 66
    assert report["core_comparison_count"] == report["actual_candidate_calls"] == 2100
    assert report["actual_reference_calls"] == 2100
    assert report["total_fp16_bit_unequal"] == 0 and report["cleanup_passed"]
    assert report["core_snapshot_evaluations"] == 4200
    assert report["actual_native_sdpa_calls"] == 0
    assert {row["selected"] for row in report["dispatches"]} == {"gather_tiled", "reference"}
    assert transformer.attention is original
    assert all(layer.offset == 0 for lane in bundle.model.created for layer in lane)


def test_numerical_failure_runs_remaining_comparisons(qualification, model_run):
    bundle, run, core, transformer, original = model_run

    def inaccurate(*args, **options):
        return run(*args, **options) + np.float32(.05)

    report = qualification.qualify_model(bundle, inaccurate, 7, {})
    assert report["status"] == "FAIL" and "exception" not in report
    assert report["comparison_count"] == 3818 and report["core_comparison_count"] == 2100
    assert report["actual_candidate_calls"] == 2100
    assert report["failing_comparisons"] > 0 and report["failing_fp16_boundaries"] > 0
    assert report["cleanup_passed"] and transformer.attention is original


def test_executed_exception_fails_and_restores_patch_and_all_caches(qualification, model_run):
    bundle, run, core, transformer, original = model_run
    attempted = 0

    def broken(*args, **options):
        nonlocal attempted
        attempted += 1
        if attempted == 2:
            raise RuntimeError("executed candidate failure")
        return run(*args, **options)

    report = qualification.qualify_model(bundle, broken, 7, {})
    assert report["status"] == "FAIL" and report["exception"]["type"] == "RuntimeError"
    assert report["candidate_calls_attempted"] == 2 and report["actual_candidate_calls"] == 1
    assert report["cleanup_passed"] and transformer.attention is original
    assert all(layer.offset == 0 for lane in bundle.model.created for layer in lane)


def test_integrated_mode_uses_observed_production_dispatch_not_diagnostic_callable(
        qualification, model_run):
    bundle, run, core, transformer, original = model_run

    def forbidden(*args, **options):
        raise AssertionError("integrated mode must not execute the diagnostic callable")

    report = qualification.qualify_model(bundle, forbidden, 128, {}, integrated=True)
    assert report["status"] == "PASS", report.get("exception")
    assert report["comparison_count"] == 3818 and report["core_comparison_count"] == 2100
    assert report["actual_candidate_backend"] == "native_gather_tiled"
    assert report["actual_candidate_calls"] == report["actual_reference_calls"] == 2100
    assert report["actual_fallback_calls"] == report["unverified_candidate_dispatch_calls"] == 0
    assert report["candidate_identity"]["provided_callable_executed"] is False
    assert report["production_integrated"] is True
    assert all(row["diagnostic_override"] is False for row in report["dispatches"])
    assert all(row["preappend_selected"] == "native_gather_tiled"
               and row["actual_dispatch"]["selected"] == "native_gather_tiled"
               for row in report["core_comparisons"])
    assert transformer.attention is original and report["cleanup_passed"]


def test_integrated_fallback_after_preappend_selection_is_not_candidate_qualification(
        qualification, model_run):
    bundle, run, core, transformer, original = model_run
    transformer.forced_actual_selected = "reference"
    report = qualification.qualify_model(bundle, run, 128, {}, integrated=True)
    assert report["status"] == "FAIL" and "exception" not in report
    assert report["failing_comparisons"] == 0 and report["failing_fp16_boundaries"] == 0
    assert report["actual_candidate_calls"] == 0
    assert report["actual_fallback_calls"] == 2100 and report["actual_reference_calls"] == 4200
    assert report["actual_candidate_backend"] == "reference"
    assert report["requested_candidate_backend"] == "native_gather_tiled"
    assert report["core_comparison_count"] == 2100
    assert all(row["preappend_selected"] == "native_gather_tiled"
               and row["actual_dispatch"]["selected"] == "reference"
               and row["actual_dispatch"]["reason"] == "fixture actual dispatcher revalidation"
               for row in report["core_comparisons"])
    assert report["cleanup_passed"] and transformer.attention is original


def test_integrated_core_requires_actual_preappend_decision(qualification, model_run):
    bundle, run, core, transformer, original = model_run
    bundle.model.forced_preappend_selected = "reference"
    transformer.forced_actual_selected = "native_gather_tiled"
    report = qualification.qualify_model(bundle, run, 128, {}, integrated=True)
    assert report["status"] == "FAIL"
    assert report["actual_candidate_calls"] == 2100 and report["actual_fallback_calls"] == 0
    assert report["dispatch_issues"]  # Even reference lanes must retain actual reference dispatch.
    assert all(row["preappend_match"] is False for row in report["core_comparisons"])


def test_missing_dispatch_telemetry_never_becomes_a_candidate_pass(qualification, model_run):
    bundle, run, core, transformer, original = model_run
    transformer.suppress_actual_dispatch = True
    report = qualification.qualify_model(bundle, run, 128, {}, integrated=True)
    assert report["status"] == "FAIL" and "exception" not in report
    assert report["actual_candidate_calls"] == 0
    assert report["unverified_candidate_dispatch_calls"] == 2100
    assert report["actual_candidate_backend"] is None and report["dispatch_issues"]
    assert report["comparison_count"] == 3818 and report["core_comparison_count"] == 2100
    assert report["cleanup_passed"] and transformer.attention is original


def test_integrated_route_rejects_an_unreported_query_tile(qualification, model_run):
    bundle, run, core, transformer, original = model_run
    report = qualification.qualify_model(bundle, run, 7, {}, integrated=True)
    assert report["status"] == "FAIL"
    assert report["exception"]["type"] == "ValueError"
    assert report["actual_candidate_calls"] == 0


def test_no_core_snapshots_preserves_model_and_dispatch_gates_without_attention_evaluation(
        qualification, model_run):
    bundle, run, core, transformer, original = model_run
    bundle.model.use_array_mask = True
    evaluated_shapes = []
    core.eval = lambda *values: evaluated_shapes.extend(value.shape for value in values)

    def forbidden(*args, **options):
        raise AssertionError("integrated mode must not execute the diagnostic callable")

    report = qualification.qualify_model(bundle, forbidden, 128, {}, integrated=True,
                                         core_snapshots=False)
    assert report["status"] == "PASS", report.get("exception")
    assert report["comparison_count"] == 3818 and report["greedy_passed"] == 66
    assert report["actual_candidate_calls"] == report["actual_reference_calls"] == 2100
    assert report["core_dispatch_comparison_count"] == 2100
    assert report["core_comparison_count"] == report["fp16_boundary_comparison_count"] == 0
    assert report["core_numerical_scope"] == "NOT_RUN"
    assert report["core_snapshot_evaluations"] == report["mask_content_snapshot_evaluations"] == 0
    assert report["total_fp16_bit_unequal"] is None and report["max_float16_ulp"] is None
    assert report["failing_core_comparisons"] is report["failing_fp16_boundaries"] is None
    assert all("fp32" not in row and "cast_boundary" not in row and row["evaluated"]
               and row["numerical_scope"] == "NOT_RUN" for row in report["core_comparisons"])
    assert all(row["completed"] and row["execution_status"] == "EVALUATED_WITH_FORWARD"
               for row in report["dispatches"])
    assert not any(len(shape) == 2 or len(shape) == 4 and shape[1] == 2
                   for shape in evaluated_shapes)
    assert report["cleanup_passed"] and transformer.attention is original


def test_no_core_snapshots_still_finishes_model_comparisons_after_numerical_failure(
        qualification, model_run):
    bundle, run, core, transformer, original = model_run

    def inaccurate(*args, **options):
        return run(*args, **options) + np.float32(.05)

    report = qualification.qualify_model(bundle, inaccurate, 7, {}, core_snapshots=False)
    assert report["status"] == "FAIL" and "exception" not in report
    assert report["comparison_count"] == 3818 and report["failing_comparisons"] > 0
    assert report["actual_candidate_calls"] == report["core_dispatch_comparison_count"] == 2100
    assert report["core_comparison_count"] == report["fp16_boundary_comparison_count"] == 0
    assert report["total_fp16_numeric_unequal"] is None
    assert report["total_fp16_bit_unequal"] is None
    assert report["cleanup_passed"] and transformer.attention is original


def test_whole_forward_evaluation_failure_does_not_complete_built_candidate_graphs(
        qualification, model_run):
    bundle, run, core, transformer, original = model_run
    forwards = 0

    def fail_second_forward(*values):
        nonlocal forwards
        if any(value.shape == (1, 1, 16) for value in values):
            forwards += 1
            if forwards == 2:
                raise RuntimeError("candidate forward evaluation failed")

    core.eval = fail_second_forward
    report = qualification.qualify_model(bundle, run, 128, {}, integrated=True,
                                         core_snapshots=False)
    assert report["status"] == "FAIL" and report["exception"]["type"] == "RuntimeError"
    assert report["candidate_calls_attempted"] == 28 and report["actual_candidate_calls"] == 0
    assert report["completed_candidate_lane_calls"] == 0
    assert report["actual_reference_calls"] == 28 and report["unevaluated_core_rows"] == 28
    assert report["actual_candidate_backend"] is None
    assert all(row["status"] == "GRAPH_BUILT" and row["evaluated"] is False
               for row in report["core_comparisons"])
    candidate_dispatches = [row for row in report["dispatches"] if row["lane"] == "candidate_full"]
    assert len(candidate_dispatches) == 28 and not any(row["completed"]
                                                    for row in candidate_dispatches)
    assert report["cleanup_passed"] and transformer.attention is original
