"""Whole-model qualification of the isolated gathered, tiled attention candidate.

This is a diagnostic override of attention only. Native model and cache classes,
their reference arithmetic, and production dispatch policy remain unchanged.
"""
from __future__ import annotations

import gc
import hashlib
import traceback
from unittest.mock import patch


def boundary_metrics(reference, candidate):
    """Compare raw FP32 core results at their next FP16 cast, without correction.

    Numerical equality, bit equality, and signed-zero-only differences are separate
    facts. Strict acceptance requires finite original/cast values and zero unequal
    FP16 bits. Ordered-half ULP distance folds the two signed zeros together.
    """
    import numpy as np

    from mlx_one.ops.numerical_trace import compare_tensors

    left, right = np.asarray(reference), np.asarray(candidate)
    fp32 = compare_tensors(left, right, dtype="float32")
    result = {"status": "FAIL", "shape_match": left.shape == right.shape,
              "reference_shape": list(left.shape), "candidate_shape": list(right.shape),
              "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype),
              "elements": int(left.size), "finite": False, "fp16_finite": False,
              "nonfinite_elements": None, "fp16_nonfinite_elements": None,
              "fp16_numeric_unequal": None, "fp16_bit_unequal": None,
              "signed_zero_bit_unequal": None, "max_float16_ulp": None,
              "fp32": fp32, "fp32_within_tolerance": fp32["status"] == "PASS",
              "max_abs_error": fp32["max_abs_error"],
              "mean_abs_error": fp32["mean_abs_error"],
              "max_relative_error": fp32["max_relative_error"],
              "mean_relative_error": fp32["mean_relative_error"],
              "failing_elements": fp32["failing_elements"],
              "criterion": "frozen FP32 gate; finite FP32 and FP16; zero unequal FP16 bits"}
    if not result["shape_match"]:
        result["reason"] = "shape mismatch"
        return result
    if not left.size:
        result["reason"] = "empty core output"
        return result
    finite = np.isfinite(left) & np.isfinite(right)
    with np.errstate(over="ignore", invalid="ignore"):
        left_half, right_half = left.astype(np.float16), right.astype(np.float16)
    half_finite = np.isfinite(left_half) & np.isfinite(right_half)
    left_bits, right_bits = left_half.view(np.uint16), right_half.view(np.uint16)
    bit_unequal = left_bits != right_bits
    signed_zero = bit_unequal & (left_half == 0) & (right_half == 0)

    def ordered(bits):
        values = bits.astype(np.int32)
        magnitude = values & 0x7fff
        return np.where(values & 0x8000, 0x8000 - magnitude, 0x8000 + magnitude)

    ulps = np.abs(ordered(left_bits)[half_finite] - ordered(right_bits)[half_finite])
    result.update(finite=bool(np.all(finite)), fp16_finite=bool(np.all(half_finite)),
                  nonfinite_elements=int(np.count_nonzero(~finite)),
                  fp16_nonfinite_elements=int(np.count_nonzero(~half_finite)),
                  fp16_numeric_unequal=int(np.count_nonzero(left_half != right_half)),
                  fp16_bit_unequal=int(np.count_nonzero(bit_unequal)),
                  signed_zero_bit_unequal=int(np.count_nonzero(signed_zero)),
                  max_float16_ulp=int(ulps.max()) if ulps.size else None)
    if (result["fp32_within_tolerance"] and result["finite"] and result["fp16_finite"]
            and not result["fp16_bit_unequal"]):
        result["status"] = "PASS"
    return result


def qualify_model(bundle, candidate, query_tile, report, *, integrated=False, core_snapshots=True):
    """Update and return ``report`` with the exact Qwen3 whole-model gates.

    Reference-full and reference-chunk lanes provide schedule-matched core
    snapshots immediately before each candidate forward. The original 3818
    logits/KV comparisons still use the all-reference full-prefill lane. No
    numerical mismatch stops remaining comparisons. Executed exceptions fail.
    Integrated execution observes production dispatch and never invokes the
    supplied diagnostic candidate callable or overrides its backend decisions.
    Disabling core snapshots retains whole-model gates and dispatch checks while
    evaluating only complete forwards. Core numerical gates then remain NOT_RUN.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.inference_baseline import TOLERANCES
    from mlx_one.models.shared import transformer
    from mlx_one.ops.numerical_trace import compare_tensors, tensor_metadata

    expected_backend = "native_gather_tiled" if integrated else "gather_tiled"
    report.update(schema_version="reference-tiled-model-qualification-v1", status="RUNNING",
                  qualification_promoted=False, actual_candidate_backend=None,
                  requested_candidate_backend=expected_backend, integrated=integrated,
                  production_integrated=integrated,
                  diagnostic_attention_override=not integrated,
                  core_snapshots=core_snapshots,
                  core_numerical_scope="MEASURED" if core_snapshots else "NOT_RUN",
                  core_snapshot_evaluations=0, mask_content_snapshot_evaluations=0,
                  actual_native_sdpa_calls=0, candidate_calls_attempted=0,
                  actual_candidate_calls=0, actual_reference_calls=0,
                  completed_candidate_lane_calls=0, baseline_reference_calls=0,
                  actual_fallback_calls=0, unverified_candidate_dispatch_calls=0,
                  query_tile=query_tile, comparisons=[], core_comparisons=[], dispatches=[],
                  pairing_issues=[], dispatch_issues=[], cleanup_errors=[],
                  candidate_identity={"module": getattr(candidate, "__module__", None),
                                      "name": getattr(candidate, "__qualname__", repr(candidate)),
                                      "provided_callable_executed": not integrated},
                  frozen_tolerances={key: list(value) for key, value in TOLERANCES.items()},
                  scope="B1 native Qwen3; full/chunk7 prefill and 32 forced continuation steps",
                  performance="NOT_RUN; numerical qualification only")
    caches = {}
    original_attention = transformer.attention
    comparisons, cores, dispatches = (report["comparisons"], report["core_comparisons"],
                                     report["dispatches"])

    def snapshot(value):
        mx.eval(value)
        return np.array(value.astype(mx.float32), copy=True)

    def mask_identity(mask):
        if mask is None or isinstance(mask, str):
            return {"kind": "none" if mask is None else mask}
        if not core_snapshots:
            return {"kind": "array", "dtype": str(mask.dtype), "shape": list(mask.shape),
                    "content_capture": "NOT_RUN"}
        report["mask_content_snapshot_evaluations"] += 1
        value = snapshot(mask)
        return {"kind": "array", "dtype": str(mask.dtype), "shape": list(mask.shape),
                "float32_content_sha256": hashlib.sha256(value.tobytes()).hexdigest()}

    def forward(lane, tokens, stage, expected=None):
        is_candidate = lane.startswith("candidate_")
        captured, core_index = [], 0
        history = caches[lane][0].offset
        observed = []
        completed_routes, forward_dispatches, forward_cores = [], [], []

        def complete_route(candidate_lane, selected):
            if not candidate_lane:
                report["baseline_reference_calls"] += 1
            else:
                report["completed_candidate_lane_calls"] += 1
                report["actual_candidate_calls"] += int(selected == expected_backend)
                report["actual_fallback_calls"] += int(selected is not None
                                                        and selected != expected_backend)
                report["unverified_candidate_dispatch_calls"] += int(selected is None)
            report["actual_reference_calls"] += int(selected == "reference")
            report["actual_native_sdpa_calls"] += int(selected == "mlx_sdpa")

        def completed_or_deferred(candidate_lane, selected):
            if core_snapshots:
                complete_route(candidate_lane, selected)
            else:
                completed_routes.append((candidate_lane, selected))

        def record_dispatch(record):
            record.update(completed=core_snapshots,
                          execution_status=("EVALUATED_PER_CORE" if core_snapshots
                                            else "GRAPH_BUILT"))
            dispatches.append(record)
            forward_dispatches.append(record)

        def observe(event):
            if event.get("kind") == "math_dispatch" and event.get("operation") == "attention":
                observed.append(event)

        observe.capture_math_dispatch = True
        observe.profile_math = False

        def observed_route(start):
            events = observed[start:]
            if len(events) != 1:
                report["dispatch_issues"].append({"lane": lane, "stage": stage,
                    "layer": core_index - 1, "expected_dispatches": 1,
                    "actual_dispatches": len(events), "events": events})
                return {"selected": None,
                        "reason": "missing or ambiguous actual attention dispatch"}
            return events[0]

        def attention(q, k, v, *, scale, mask=None, stable=False, promote=False, decision=None):
            nonlocal core_index
            index = core_index
            core_index += 1
            metadata = {"stage": stage, "lane": lane, "layer": index,
                        "q_shape": list(q.shape), "k_shape": list(k.shape),
                        "v_shape": list(v.shape), "history_tokens": history,
                        "scale": float(scale), "mask": mask_identity(mask),
                        "preappend_selected": getattr(decision, "selected", None),
                        "preappend_reason": getattr(decision, "reason", None)}
            dispatch_start = len(observed)
            if not is_candidate:
                output = original_attention(q, k, v, scale=scale, mask=mask, stable=stable,
                                            promote=promote, decision=decision)
                metadata["output_dtype"] = str(output.dtype).rsplit(".", 1)[-1]
                if core_snapshots:
                    report["core_snapshot_evaluations"] += 1
                    captured.append({"tensor": snapshot(output), "metadata": metadata})
                else:
                    captured.append({"metadata": metadata})
                actual_route = (observed_route(dispatch_start) if integrated else
                                {"requested": "reference", "selected": "reference",
                                 "reason": "unchanged reference policy"})
                completed_or_deferred(False, actual_route["selected"])
                if actual_route["selected"] != "reference":
                    report["dispatch_issues"].append({**metadata, "actual_dispatch": actual_route,
                        "reason": "reference lane did not select reference"})
                record_dispatch({**metadata, **actual_route,
                                 "diagnostic_override": not integrated})
                return output
            report["candidate_calls_attempted"] += 1
            # Diagnostic mode runs the candidate graph. Integrated mode runs the
            # actual production decision, including an explicitly failed fallback.
            # Reference host evidence is never a candidate correction path.
            if integrated:
                output = original_attention(q, k, v, scale=scale, mask=mask, stable=stable,
                                            promote=promote, decision=decision)
            else:
                output = candidate(q, k, v, scale=scale, mask=mask, query_tile=query_tile)
            metadata["output_dtype"] = str(output.dtype).rsplit(".", 1)[-1]
            actual = None
            if core_snapshots:
                report["core_snapshot_evaluations"] += 1
                actual = snapshot(output)
            actual_route = (observed_route(dispatch_start) if integrated else
                            {"requested": "gather_tiled", "selected": "gather_tiled",
                             "reason": "isolated diagnostic candidate graph"})
            selected = actual_route["selected"]
            completed_or_deferred(True, selected)
            record_dispatch({**metadata, **actual_route,
                             "diagnostic_override": not integrated,
                             "operands": "float32" if selected == expected_backend else None,
                             "native_sdpa": selected == "mlx_sdpa"})
            call = (report["completed_candidate_lane_calls"] if core_snapshots else
                    report["completed_candidate_lane_calls"] + index + 1)
            row = {**metadata, "call": call,
                   "actual_backend": selected, "actual_dispatch": actual_route,
                   "evaluated": core_snapshots,
                   "numerical_scope": "MEASURED" if core_snapshots else "NOT_RUN",
                   "preappend_match": (not integrated or metadata["preappend_selected"]
                                       == "native_gather_tiled"),
                   "actual_dispatch_match": selected == expected_backend,
                   "query_tile_match": (not integrated or selected != expected_backend
                       or actual_route.get("shape", {}).get("query_tile") == query_tile)}
            if expected is None or index >= len(expected):
                row.update(status="FAIL", reason="missing schedule-matched reference core")
            else:
                baseline = expected[index]
                if core_snapshots:
                    row["fp32"] = compare_tensors(baseline["tensor"], actual, dtype="float32")
                    row["cast_boundary"] = boundary_metrics(baseline["tensor"], actual)
                row["schedule_match"] = all(
                    metadata[key] == baseline["metadata"][key]
                    for key in ("layer", "q_shape", "k_shape", "v_shape", "history_tokens",
                                "scale", "mask"))
                row["output_dtype_match"] = (metadata["output_dtype"]
                    == baseline["metadata"]["output_dtype"] == "float32")
                row["status"] = "PASS" if (row["schedule_match"]
                    and row["output_dtype_match"]
                    and row["preappend_match"] and row["actual_dispatch_match"]
                    and row["query_tile_match"]
                    and (not core_snapshots or (row["fp32"]["status"] == "PASS"
                         and row["cast_boundary"]["status"] == "PASS"))) else "FAIL"
            if not core_snapshots:
                row["dispatch_check_status"] = row["status"]
                if row["status"] == "PASS":
                    row["status"] = "GRAPH_BUILT"
            cores.append(row)
            forward_cores.append(row)
            return output

        policy = "native_gather_tiled" if integrated and is_candidate else "reference"
        with patch.object(transformer, "attention", attention), math_execution(
                NativeMathPolicy(attention=policy), family=bundle.architecture,
                observer=observe if integrated else None):
            value = bundle.model(mx.array([tokens]), cache=caches[lane]).logits[:, -1:, :]
            mx.eval(value)
        if not core_snapshots:
            for candidate_lane, selected in completed_routes:
                complete_route(candidate_lane, selected)
            for record in forward_dispatches:
                record.update(completed=True, execution_status="EVALUATED_WITH_FORWARD")
            for row in forward_cores:
                row["evaluated"] = True
                row["status"] = row["dispatch_check_status"]
        if core_index != 28 or (is_candidate and len(expected or []) != core_index):
            report["pairing_issues"].append({"lane": lane, "stage": stage,
                "core_calls": core_index, "expected_layers": 28,
                "reference_core_calls": len(expected) if expected is not None else None})
        if integrated and len(observed) != 28:
            report["dispatch_issues"].append({"lane": lane, "stage": stage,
                "expected_forward_dispatches": 28, "actual_forward_dispatches": len(observed)})
        return value, captured

    def compare(left, right, stage, *, state=False, reference_offset=None, candidate_offset=None):
        left_value, right_value = snapshot(left), snapshot(right)
        row = compare_tensors(left_value, right_value, dtype="float16",
                              reference_metadata=tensor_metadata(left),
                              candidate_metadata=tensor_metadata(right))
        row.update(stage=stage, kind="state" if state else "logits")
        if state:
            row.update(reference_offset=reference_offset, candidate_offset=candidate_offset,
                       offset_match=reference_offset == candidate_offset)
            if not row["offset_match"]:
                row.update(status="FAIL", reason="cache offset mismatch")
        else:
            row["greedy_agreement"] = bool(np.all(left_value.argmax(-1)
                                                  == right_value.argmax(-1)))
        comparisons.append(row)

    def compare_states(lanes, stage):
        for lane in lanes:
            for layer, (reference_cache, candidate_cache) in enumerate(
                    zip(caches["reference_full"], caches[lane], strict=True)):
                for component, (left, right) in enumerate(zip(reference_cache.state(),
                        candidate_cache.state(), strict=True)):
                    compare(left, right, f"{stage}/lane={lane}/layer={layer}/component={component}",
                            state=True, reference_offset=reference_cache.offset,
                            candidate_offset=candidate_cache.offset)

    try:
        if type(query_tile) is not int or query_tile < 1:
            raise ValueError("query_tile must be a positive integer")
        if type(integrated) is not bool:
            raise ValueError("integrated must be boolean")
        if type(core_snapshots) is not bool:
            raise ValueError("core_snapshots must be boolean")
        if integrated and query_tile != 128:
            raise ValueError("integrated production route requires query_tile=128")
        if not callable(candidate):
            raise TypeError("candidate attention must be callable")
        ids = bundle.tokenizer.encode(
            "An engineer measures a quiet river, then records seven distinct observations. " * 8
        )[:64]
        forced = bundle.tokenizer.encode(" The measured result remains reproducible." * 8)[:32]
        report.update(prompt_ids=ids, forced_ids=forced, prompt_tokens=len(ids),
                      forced_steps=len(forced), chunk_size=7)
        if len(ids) != 64 or len(forced) != 32:
            raise ValueError("exact workload requires 64 prompt tokens and 32 continuation tokens")
        for lane in ("reference_full", "candidate_full", "reference_chunk", "candidate_chunk"):
            caches[lane] = bundle.model.make_cache()
            if len(caches[lane]) != 28:
                raise ValueError("pinned Qwen3 qualification requires 28 native transformer layers")

        reference, baseline = forward("reference_full", ids, "full_prefill")
        actual, _ = forward("candidate_full", ids, "full_prefill", baseline)
        baseline.clear()
        compare(reference, actual, "full_prefill")
        compare_states(["candidate_full"], "full_prefill_kv")
        for start in range(0, len(ids), 7):
            tokens = ids[start:start + 7]
            _, baseline = forward("reference_chunk", tokens, f"chunk/{start}")
            chunked, _ = forward("candidate_chunk", tokens, f"chunk/{start}", baseline)
            baseline.clear()
        compare(reference, chunked, "chunked_prefill")
        compare_states(["candidate_full", "candidate_chunk"], "chunked_prefill_kv")
        for step, token in enumerate(forced):
            reference, baseline = forward("reference_full", [token], f"continuation/{step}")
            actual, _ = forward("candidate_full", [token], f"continuation/{step}", baseline)
            baseline.clear()
            compare(reference, actual, f"continuation/{step}/candidate_full")
            _, baseline = forward("reference_chunk", [token], f"continuation/{step}")
            actual, _ = forward("candidate_chunk", [token], f"continuation/{step}", baseline)
            baseline.clear()
            compare(reference, actual, f"continuation/{step}/candidate_chunk")
            compare_states(["candidate_full", "candidate_chunk"], f"continuation_kv/{step}")
        logits = [row for row in comparisons if row["kind"] == "logits"]
        good = (len(comparisons) == 3818 and len(logits) == 66 and len(cores) == 2100
                and report["actual_candidate_calls"] == report["candidate_calls_attempted"] == 2100
                and report["completed_candidate_lane_calls"] == 2100
                and report["baseline_reference_calls"] == 2100
                and report["actual_reference_calls"] == 2100 and not report["pairing_issues"]
                and report["actual_fallback_calls"] == 0
                and report["unverified_candidate_dispatch_calls"] == 0
                and not report["dispatch_issues"]
                and sum("cast_boundary" in row for row in cores) == (2100 if core_snapshots else 0)
                and all(row["status"] == "PASS" for row in comparisons)
                and all(row["greedy_agreement"] for row in logits)
                and all(row["status"] == "PASS" for row in cores))
        report["status"] = "PASS" if good else "FAIL"
    except Exception as exc:
        report.update(status="FAIL", exception={"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()})
    finally:
        for lane, cache in caches.items():
            for index, layer in enumerate(cache):
                try:
                    layer.reset()
                except Exception as exc:
                    report["cleanup_errors"].append({"lane": lane, "layer": index,
                        "type": type(exc).__name__, "message": str(exc)})
        report["cleanup_offsets"] = {lane: [layer.offset for layer in cache]
                                     for lane, cache in caches.items()}
        report["cleanup_passed"] = (not report["cleanup_errors"] and len(caches) == 4
            and all(offset == 0 for offsets in report["cleanup_offsets"].values()
                    for offset in offsets))
        if not report["cleanup_passed"]:
            report["status"] = "FAIL"
        caches.clear()
        gc.collect()
        try:
            mx.clear_cache()
        except Exception as exc:
            report["cleanup_errors"].append({"operation": "clear_cache",
                "type": type(exc).__name__, "message": str(exc)})
            report.update(status="FAIL", cleanup_passed=False)

    cast_rows = [row["cast_boundary"] for row in cores if "cast_boundary" in row]
    logits = [row for row in comparisons if row["kind"] == "logits"]
    actual_backends = {row["actual_backend"] for row in cores if row["evaluated"]}
    report["actual_candidate_backend"] = (next(iter(actual_backends)) if len(actual_backends) == 1
                                          else "mixed" if actual_backends else None)
    report.update(model_qualification_status=report["status"], comparison_count=len(comparisons),
                  failing_comparisons=sum(row["status"] != "PASS" for row in comparisons),
                  greedy_comparisons=len(logits),
                  greedy_passed=sum(row["greedy_agreement"] for row in logits),
                  core_comparison_count=len(cores) if core_snapshots else 0,
                  core_dispatch_comparison_count=len(cores),
                  unevaluated_core_rows=sum(not row["evaluated"] for row in cores),
                  failing_core_dispatch_comparisons=sum(row["status"] != "PASS" for row in cores),
                  failing_core_comparisons=(sum(row["status"] != "PASS" for row in cores)
                                            if core_snapshots else None),
                  fp16_boundary_comparison_count=len(cast_rows),
                  failing_fp16_boundaries=(sum(row["status"] != "PASS" for row in cast_rows)
                                           if core_snapshots else None),
                  total_fp16_numeric_unequal=sum(
                      row["fp16_numeric_unequal"] or 0 for row in cast_rows)
                      if core_snapshots else None,
                  total_fp16_bit_unequal=(sum(row["fp16_bit_unequal"] or 0 for row in cast_rows)
                                          if core_snapshots else None),
                  total_signed_zero_bit_unequal=sum(
                      row["signed_zero_bit_unequal"] or 0 for row in cast_rows)
                      if core_snapshots else None,
                  max_float16_ulp=max((row["max_float16_ulp"] for row in cast_rows
                                      if row["max_float16_ulp"] is not None), default=None),
                  max_error=max((row["max_abs_error"] for row in comparisons
                                 if row["max_abs_error"] is not None), default=None),
                  first_failing_core=next((row for row in cores if row["status"] != "PASS"), None),
                  first_failing_comparison=next(
                      (row for row in comparisons if row["status"] != "PASS"), None))
    return report
