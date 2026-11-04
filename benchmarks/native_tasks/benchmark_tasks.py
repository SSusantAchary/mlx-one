#!/usr/bin/env python3
"""Diagnostic local ViT timings; no cross-runtime speedup claim."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_one.vision.vit_classification import load_vit_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 5:
        raise ValueError("at least five timed samples are required")
    start = time.perf_counter()
    model = load_vit_classifier(args.checkpoint)
    load_ms = (time.perf_counter() - start) * 1000
    model.predict(args.image)  # one excluded warmup
    results = []
    times = []
    for _ in range(args.samples):
        start = time.perf_counter()
        results.append(model.predict(args.image))
        times.append((time.perf_counter() - start) * 1000)
    if any(result != results[0] for result in results[1:]):
        raise ValueError("outputs changed between repeated benchmark requests")
    ordered = sorted(times)
    import math

    report = {
        "kind": "full Python task, one image, includes preprocessing and postprocessing",
        "model": model.model_id,
        "revision": model.revision,
        "warmups": 1,
        "samples": args.samples,
        "latencies_ms": times,
        "median_ms": statistics.median(times),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "p95_warning": "diagnostic only; small sample count" if args.samples < 20 else None,
        "load_ms": load_ms,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "image": str(args.image),
        "result": results[0],
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
