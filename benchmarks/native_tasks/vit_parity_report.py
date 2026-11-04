#!/usr/bin/env python3
"""Generate numerical summaries of saved ViT investigation traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlx_one.inference_baseline import TOLERANCES


def metrics(reference, actual, atol, rtol):
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite trace")
    a = reference.astype(np.float64)
    b = actual.astype(np.float64)
    error = np.abs(a - b)
    allowed = atol + rtol * np.abs(a)
    return {
        "shape": list(a.shape),
        "reference_dtype": str(reference.dtype),
        "actual_dtype": str(actual.dtype),
        "max_abs_error": float(error.max()),
        "max_normalized_error": float((error / allowed).max()),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "failed_elements": int((error > allowed).sum()),
        "passed": bool(np.all(error <= allowed)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    atol, rtol = TOLERANCES["float32"]
    with np.load(args.run / "hf.npz", allow_pickle=False) as ref:
        with np.load(args.run / "operator-replay.npz", allow_pickle=False) as replay:
            operations = {key: metrics(ref[key], replay[key], atol, rtol) for key in replay.files}
        variants = {}
        layers = [
            "layer_00",
            *[f"layer_{i:02}" for i in range(1, 13)],
            "final_hidden",
            "logits",
            "probabilities",
        ]
        for name in (
            "tf32-off",
            "default",
            "eager",
            "approx_gelu",
            "explicit_ln",
            "transposed_linear",
            "residual_reverse",
            "materialize",
            "cpu",
        ):
            with np.load(args.run / f"mlx-{name}.npz", allow_pickle=False) as actual:
                entry = {key: metrics(ref[key], actual[key], atol, rtol) for key in layers}
                entry["isolated_blocks"] = {
                    key: metrics(ref[key], actual["isolated_" + key], atol, rtol)
                    for key in ref.files
                    if key.startswith(("block_04_", "block_05_"))
                    and "isolated_" + key in actual.files
                }
                variants[name] = entry
        with np.load(args.run / "mlx-tf32-off.npz", allow_pickle=False) as actual:
            chained = {
                key: metrics(ref[key], actual[key], atol, rtol)
                for key in ref.files
                if key.startswith(("block_04_", "block_05_")) and key in actual.files
            }
    report = {
        "atol": atol,
        "rtol": rtol,
        "normalized_error_definition": "abs(actual-reference)/(atol+rtol*abs(reference))",
        "variants": variants,
        "chained_operations": chained,
        "independent_operator_replay": operations,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
