#!/usr/bin/env python3
"""Source-relative depth map errors; not independent ground-truth depth quality."""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("matrix", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
summary = []
for path in sorted(args.matrix.glob("*/comparison.json")):
    report = json.loads(path.read_text())
    row = {
        "image": path.parent.name,
        "whole_trace_pass": report["passed"],
        "first_divergent": report["first_divergent"],
        "failed": {
            k: {a: v[a] for a in ("max_abs_error", "failing_elements")}
            for k, v in report["tensors"].items()
            if not v["passed"]
        },
        "maps": {},
    }
    with (
        np.load(path.parent / "reference.npz", allow_pickle=False) as ref,
        np.load(path.parent / "native.npz", allow_pickle=False) as native,
    ):
        for key in ("predicted_depth", "resized_depth"):
            difference = native[key].astype(np.float64) - ref[key].astype(np.float64)
            row["maps"][key] = {
                "shape": list(ref[key].shape),
                "mae": float(np.abs(difference).mean()),
                "rmse": float(np.sqrt((difference**2).mean())),
                "max_abs_error": float(np.max(np.abs(difference))),
                "gate_pass": report["tensors"][key]["passed"],
            }
    summary.append(row)
with args.output.open("x") as stream:
    json.dump(summary, stream, indent=2)
