"""Opt-in, bounded tensor diagnostics with the frozen numerical comparison gates.

Hooks never evaluate an array. The supplied callback owns evaluation and snapshotting,
so the normal model path only performs a ContextVar lookup when tracing is disabled.
This module deliberately has no MLX import and imports NumPy only for comparisons.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceState:
    callback: Callable[[dict[str, Any]], None]
    max_layers: int
    max_events: int
    metadata: dict[str, Any] = field(default_factory=dict)
    emitted: int = 0
    dropped: int = 0


_TRACE: ContextVar[TraceState | None] = ContextVar("numerical_trace", default=None)
_SCOPE: ContextVar[dict[str, Any] | None] = ContextVar("numerical_trace_scope", default=None)


@contextmanager
def numerical_trace(callback, *, max_layers=3, max_events=512, metadata=None):
    """Capture at most ``max_events`` events from layers ``[0, max_layers)``.

    A new context has its own budget, allowing a caller to bound each forward pass.
    Callback failures propagate: incomplete diagnostics must never look successful.
    """
    if not callable(callback):
        raise TypeError("trace callback must be callable")
    if (type(max_layers) is not int or type(max_events) is not int
            or min(max_layers, max_events) < 1):
        raise ValueError("trace bounds must be positive integers")
    state = TraceState(callback, max_layers, max_events, dict(metadata or {}))
    token = _TRACE.set(state)
    try:
        yield state
    finally:
        _TRACE.reset(token)


@contextmanager
def trace_scope(name, **metadata):
    """Attach a forward name and caller metadata to events in this context."""
    token = _SCOPE.set({**(_SCOPE.get() or {}), "forward": name, **metadata})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def emit(layer, name, tensor, metadata=None):
    """Emit an unevaluated tensor only when a bounded trace is active."""
    state = _TRACE.get()
    if state is None:
        return
    if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < state.max_layers:
        return
    if state.emitted >= state.max_events:
        state.dropped += 1
        return
    event = {"sequence": state.emitted, "layer": layer, "name": name, "tensor": tensor,
             "metadata": {**state.metadata, **(_SCOPE.get() or {}), **dict(metadata or {})}}
    state.emitted += 1
    state.callback(event)


def _dtype_name(dtype):
    return str(dtype).rsplit(".", 1)[-1]


def tensor_metadata(tensor):
    """Return layout facts without converting, copying, or evaluating a tensor.

    MLX versions that do not expose strides are reported as unavailable. NumPy
    exposes byte strides, which are normalized to elements for a clear comparison.
    """
    shape = getattr(tensor, "shape", None)
    strides = getattr(tensor, "strides", None)
    itemsize = getattr(tensor, "itemsize", None)
    is_numpy = type(tensor).__module__.startswith("numpy")
    if strides is not None:
        strides = list(strides)
        if is_numpy and itemsize:
            strides = [stride // itemsize for stride in strides]
    return {"shape": list(shape) if shape is not None else None,
            "dtype": _dtype_name(getattr(tensor, "dtype", type(tensor).__name__)),
            "strides": strides,
            "strides_unit": "elements" if strides is not None else None,
            "strides_available": strides is not None}


def criterion_for_dtype(dtype):
    """Declare the unchanged baseline gate before inspecting tensor differences."""
    from mlx_one.inference_baseline import TOLERANCES

    dtype = _dtype_name(dtype)
    if dtype in TOLERANCES:
        atol, rtol = TOLERANCES[dtype]
        return {"dtype": dtype, "atol": atol, "rtol": rtol,
                "rule": "abs(candidate-reference) <= atol + rtol*abs(reference)",
                "source": "mlx_one.inference_baseline.TOLERANCES",
                "finite_required": True}
    if dtype.startswith(("int", "uint", "bool", "str", "<U", "|S")) or dtype in {
        "NoneType", "object", "None", "list", "tuple"
    }:
        return {"dtype": dtype, "atol": 0, "rtol": 0, "rule": "exact equality",
                "source": "exact metadata comparison", "finite_required": True}
    raise ValueError(f"no frozen numerical tolerance for dtype {dtype!r}")


def compare_tensors(reference, candidate, *, dtype=None, reference_metadata=None,
                    candidate_metadata=None, float16_ulp=None):
    """Compare evaluated operands; ``dtype`` explicitly declares a frozen gate.

    Dtype and stride differences are reported, but acceptance uses the declared
    numerical gate and identical shape. This allows controlled FP16/FP32 replay.
    Relative errors are diagnostic only, with a float64 epsilon denominator floor;
    that floor never enters the frozen acceptance criterion.
    """
    import numpy as np

    ref_meta = dict(reference_metadata or tensor_metadata(reference))
    got_meta = dict(candidate_metadata or tensor_metadata(candidate))
    ref, got = np.asarray(reference), np.asarray(candidate)
    criterion = criterion_for_dtype(dtype or ref_meta["dtype"])
    result = {"reference": ref_meta, "candidate": got_meta, "criterion": criterion,
              "shape_match": ref.shape == got.shape,
              "dtype_match": ref_meta["dtype"] == got_meta["dtype"],
              "strides_match": (ref_meta.get("strides") == got_meta.get("strides")
                                if ref_meta.get("strides_available")
                                and got_meta.get("strides_available") else None),
              "elements": int(ref.size), "failing_elements": None, "changed_elements": None,
              "max_abs_error": None, "mean_abs_error": None,
              "max_relative_error": None, "mean_relative_error": None,
              "relative_error_denominator_floor": float(np.finfo(np.float64).eps),
              "nonfinite_elements": None, "max_float16_ulp": None,
              "float16_ulp_elements": None, "status": "FAIL"}
    if not result["shape_match"]:
        result["reason"] = "shape mismatch"
        return result
    if ref.dtype.kind not in "biufc" or got.dtype.kind not in "biufc":
        equal = np.equal(ref, got)
        result.update(failing_elements=int(np.count_nonzero(~equal)),
                      changed_elements=int(np.count_nonzero(~equal)),
                      nonfinite_elements=0,
                      status="PASS" if bool(np.all(equal)) else "FAIL")
        return result
    # Float64 diagnostics avoid losing FP16/FP32 deltas during subtraction.
    left, right = ref.astype(np.float64), got.astype(np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    with np.errstate(invalid="ignore", over="ignore"):
        error = np.abs(right - left)
        limit = criterion["atol"] + criterion["rtol"] * np.abs(left)
        failed = ~finite | (error > limit)
    if criterion["rule"] == "exact equality":
        failed = ~finite | ~np.equal(ref, got)
    finite_errors = error[finite]
    relative = finite_errors / np.maximum(np.abs(left[finite]), np.finfo(np.float64).eps)
    result.update(failing_elements=int(np.count_nonzero(failed)),
                  changed_elements=int(np.count_nonzero(~np.equal(ref, got))),
                  nonfinite_elements=int(np.count_nonzero(~finite)),
                  max_abs_error=float(np.max(finite_errors)) if finite_errors.size else 0.,
                  mean_abs_error=float(np.mean(finite_errors)) if finite_errors.size else 0.,
                  max_relative_error=float(np.max(relative)) if relative.size else 0.,
                  mean_relative_error=float(np.mean(relative)) if relative.size else 0.,
                  status="PASS" if not bool(np.any(failed)) else "FAIL")
    if float16_ulp is None:
        float16_ulp = ref_meta["dtype"] == got_meta["dtype"] == "float16"
    if float16_ulp:
        with np.errstate(over="ignore", invalid="ignore"):
            half_left, half_right = left.astype(np.float16), right.astype(np.float16)
        valid = np.isfinite(half_left) & np.isfinite(half_right)

        def ordered_half(values):
            bits = values.view(np.uint16).astype(np.int32)
            magnitude = bits & 0x7fff
            return np.where(bits & 0x8000, 0x8000 - magnitude, 0x8000 + magnitude)

        ulps = np.abs(ordered_half(half_right)[valid] - ordered_half(half_left)[valid])
        result["max_float16_ulp"] = int(np.max(ulps)) if ulps.size else None
        result["float16_ulp_elements"] = int(ulps.size)
    return result


def _event_keys(events):
    seen = defaultdict(int)
    for event in events:
        key = (event.get("metadata", {}).get("forward"), event["layer"], event["name"])
        occurrence = seen[key]
        seen[key] += 1
        yield (*key, occurrence), event


def compare_traces(reference: Iterable[Mapping], candidate: Iterable[Mapping]):
    """Compare ordered snapshots and identify the first violated declared gate.

    Tensor events are matched by forward, layer, name and occurrence. Missing or
    unexpected events fail explicitly rather than silently shortening comparison.
    Each event can declare ``criterion_dtype`` in its metadata before comparison.
    """
    references = list(_event_keys(reference))
    candidates = dict(_event_keys(candidate))
    if not references:
        missing = {"status": "FAIL", "reason": "no captured reference events"}
        return {"status": "FAIL", "comparisons": [], "comparison_count": 0,
                "failing_comparisons": 0, "earliest_violation": missing,
                "reference_event_count": 0, "candidate_event_count": len(candidates)}
    rows = []
    for key, event in references:
        other = candidates.pop(key, None)
        row = {"forward": key[0], "layer": key[1], "name": key[2],
               "occurrence": key[3], "reference_sequence": event.get("sequence")}
        if other is None:
            row.update(status="FAIL", reason="missing candidate event")
        else:
            row.update(compare_tensors(
                event["tensor"], other["tensor"],
                dtype=event.get("metadata", {}).get("criterion_dtype"),
                reference_metadata=event.get("tensor_metadata"),
                candidate_metadata=other.get("tensor_metadata")))
            row["candidate_sequence"] = other.get("sequence")
        rows.append(row)
    for key, event in candidates.items():
        rows.append({"forward": key[0], "layer": key[1], "name": key[2],
                     "occurrence": key[3], "candidate_sequence": event.get("sequence"),
                     "status": "FAIL", "reason": "unexpected candidate event"})
    failures = [row for row in rows if row["status"] != "PASS"]
    return {"status": "FAIL" if failures else "PASS", "comparisons": rows,
            "comparison_count": len(rows), "failing_comparisons": len(failures),
            "earliest_violation": failures[0] if failures else None,
            "reference_event_count": len(references),
            "candidate_event_count": len(rows) - sum(
                row.get("reason") == "missing candidate event" for row in rows)}
