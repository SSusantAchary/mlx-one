#!/usr/bin/env python3
"""Bounded DistilBERT request and component timing, one process, native FP32."""

import argparse
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_one.classification import ClassificationEngine
from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.task_runners.classification import DistilBertClassificationAdapter


def summary(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1],
        "values_ms": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 5:
        raise ValueError("at least five samples required")
    text = "The film tells a warm, thoughtful story with excellent performances."
    start = time.perf_counter()
    with ClassificationEngine(args.checkpoint) as engine:
        expected = engine.classify(text).to_dict()  # load plus excluded warmup
        load_warmup = (time.perf_counter() - start) * 1000
        full, prepare, forward = [], [], []
        actor, lease = engine.service._task_lease("classification", "text-classification")
        bundle = lease.runtime.value
        adapter = DistilBertClassificationAdapter(bundle)
        try:
            for _ in range(args.samples):
                start = time.perf_counter()
                result = engine.classify(text).to_dict()
                full.append((time.perf_counter() - start) * 1000)
                if result != expected:
                    raise ValueError("repeated output differs")
                start = time.perf_counter()
                prepared = adapter.prepare("text-classification", {"texts": text})
                prepare.append((time.perf_counter() - start) * 1000)
                sequence = prepared.payload["sequences"][0]

                def run(sequence=sequence):
                    ids = mx.array([sequence])
                    mask = mx.ones_like(ids, dtype=mx.bool_)
                    logits = bundle.model(ids, mask)
                    mx.eval(logits)

                start = time.perf_counter()
                DeviceExecutor.shared().call("sentiment-benchmark", run)
                forward.append((time.perf_counter() - start) * 1000)
        finally:
            lease.release()
        report = {
            "model": bundle.model_id,
            "revision": bundle.revision,
            "dtype": "float32",
            "batch_size": 1,
            "warmups": 1,
            "processes": 1,
            "load_and_warmup_ms": load_warmup,
            "python_task": summary(full),
            "preparation": summary(prepare),
            "forward_with_device_dispatch": summary(forward),
            "requests_per_second_from_median": 1000 / statistics.median(full),
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mlx_peak_bytes": mx.get_peak_memory(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "text": text,
            "warning": (
                "Diagnostic one-process timings; no speedup comparison. "
                "Forward includes array creation and dispatch; "
                "components do not sum to full latency."
            ),
        }
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
