#!/usr/bin/env python3
"""Compare structured task outputs: exact semantics and project-gated scores."""

import argparse
import json
import math
from pathlib import Path


def compare(reference, actual, *, atol, rtol):
    errors = []
    max_error = 0.0
    scores = 0

    def visit(a, b, path):
        nonlocal max_error, scores
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append({"path": path, "reason": "keys"})
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append({"path": path, "reason": "length"})
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, f"{path}[{i}]")
        elif isinstance(a, float) and isinstance(b, (float, int)) and not isinstance(b, bool):
            scores += 1
            if not math.isfinite(a) or not math.isfinite(b):
                errors.append({"path": path, "reason": "nonfinite"})
            else:
                error = abs(a - b)
                max_error = max(max_error, error)
                if error > atol + rtol * abs(a):
                    errors.append({"path": path, "reason": "numerical_gate"})
        elif type(a) is not type(b) or a != b:
            errors.append({"path": path, "reason": "semantic_mismatch"})

    visit(reference, actual, "$")
    return {
        "passed": not errors,
        "score_count": scores,
        "max_abs_score_error": max_error,
        "error_count": len(errors),
        "first_errors": errors[:50],
        "atol": atol,
        "rtol": rtol,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from mlx_one.inference_baseline import TOLERANCES

    atol, rtol = TOLERANCES["float32"]
    result = compare(
        json.loads(args.reference.read_text()),
        json.loads(args.actual.read_text()),
        atol=atol,
        rtol=rtol,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
