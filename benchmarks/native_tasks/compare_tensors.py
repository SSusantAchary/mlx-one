#!/usr/bin/env python3
"""Compare named, pickle-free tensor traces without relaxing project tolerances.

Traces are NPZ archives. Their keys, shapes, finite values and every element are
checked before an individual tensor can pass. This tool is development-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Same gates as mlx_one.inference_baseline.TOLERANCES; import that module when
# running the CLI so the project remains the single source of truth.


def compare(reference: np.ndarray, actual: np.ndarray, *, atol: float, rtol: float) -> dict:
    ref, got = np.asarray(reference), np.asarray(actual)
    result = {
        "reference_shape": list(ref.shape),
        "actual_shape": list(got.shape),
        "reference_dtype": str(ref.dtype),
        "actual_dtype": str(got.dtype),
    }
    if ref.shape != got.shape:
        return {**result, "passed": False, "reason": "shape_mismatch"}
    if ref.dtype.kind not in "biuf" or got.dtype.kind not in "biuf":
        return {**result, "passed": False, "reason": "unsupported_dtype"}
    ref, got = ref.astype(np.float64), got.astype(np.float64)
    finite = np.isfinite(ref) & np.isfinite(got)
    result.update(count=int(ref.size), finite_count=int(finite.sum()))
    if not finite.all():
        return {**result, "passed": False, "reason": "nonfinite"}
    if ref.size == 0:
        return {**result, "passed": False, "reason": "empty_tensor"}
    diff = np.abs(ref - got)
    failure = diff > atol + rtol * np.abs(ref)
    relative = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
    result.update(
        max_abs_error=float(diff.max()),
        max_relative_error=float(relative.max()),
        rms_error=float(np.sqrt(np.mean(diff * diff))),
        failing_elements=int(failure.sum()),
        atol=atol,
        rtol=rtol,
        passed=not bool(failure.any()),
    )
    if ref.ndim and ref.shape[-1] > 0:
        result["top1_agreement"] = bool(
            np.array_equal(np.argmax(ref, axis=-1), np.argmax(got, axis=-1))
        )
    return result


def compare_archives(reference: Path, actual: Path, *, atol: float, rtol: float) -> dict:
    with np.load(reference, allow_pickle=False) as ref, np.load(actual, allow_pickle=False) as got:
        missing = sorted(set(ref.files) - set(got.files))
        unexpected = sorted(set(got.files) - set(ref.files))
        keys = [key for key in ref.files if key in got.files]
        tensors = {key: compare(ref[key], got[key], atol=atol, rtol=rtol) for key in keys}
    first = next((name for name, report in tensors.items() if not report["passed"]), None)
    return {
        "passed": bool(tensors) and not missing and not unexpected and first is None,
        "missing": missing,
        "unexpected": unexpected,
        "first_divergent": first,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--output", type=Path, required=True, help="new report path; never overwritten"
    )
    args = parser.parse_args()
    from mlx_one.inference_baseline import TOLERANCES

    atol, rtol = TOLERANCES[args.dtype]
    report = compare_archives(args.reference, args.actual, atol=atol, rtol=rtol)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
