#!/usr/bin/env python3
"""Diagnostic NER/CLIP/VAD task, preparation and native-forward latency."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx

from mlx_one.classification import TokenClassificationEngine, ZeroShotImageEngine
from mlx_one.engine.adapter_registry import get_runtime_registration
from mlx_one.engine.device_executor import DeviceExecutor


def stats(values):
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[math.ceil(0.95 * len(values)) - 1],
        "values_ms": values,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["ner", "clip", "vad"])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--samples", type=int, default=20)
    args = p.parse_args()
    if args.samples < 5:
        raise ValueError("at least 5 timed samples required")
    if args.task == "ner":
        engine = TokenClassificationEngine(args.checkpoint)
        task = "token-classification"
        payload = {"texts": "Marie Curie lived in Paris.", "aggregation_strategy": "simple"}

        def run_full():
            return engine.tag(payload["texts"], aggregation_strategy="simple")
    elif args.task == "clip":
        if args.image is None:
            raise ValueError("--image required for CLIP")
        engine = ZeroShotImageEngine(args.checkpoint)
        task = "zero-shot-image-classification"
        payload = {"image": args.image, "candidate_labels": ["cat", "dog"]}

        def run_full():
            return engine.classify(args.image, payload["candidate_labels"])

    else:
        from mlx_one.audio.vad import VADEngine, prepare_vad, validate_options

        pcm = prepare_vad(args.checkpoint / "test.wav")[:16000]
        engine = VADEngine(args.checkpoint)
        task = "voice-activity-detection"
        payload = {"pcm": pcm, "options": validate_options({})}

        def run_full():
            return engine.detect(pcm)

    start = time.perf_counter()
    with engine:
        expected = run_full().to_dict()
        load_warmup_ms = (time.perf_counter() - start) * 1000
        actor, lease = engine.service._task_lease(
            "vad" if args.task == "vad" else "classification", task
        )
        bundle = lease.runtime.value
        adapter = get_runtime_registration(bundle.architecture).adapter(task)(bundle)
        full, prepare, forward = [], [], []
        try:
            for _ in range(args.samples):
                start = time.perf_counter()
                actual = run_full().to_dict()
                full.append((time.perf_counter() - start) * 1000)
                if actual != expected:
                    raise ValueError("repeated task outputs differ")
                start = time.perf_counter()
                prepared = adapter.prepare(task, payload)
                prepare.append((time.perf_counter() - start) * 1000)

                def run(prepared=prepared):
                    data = prepared.payload
                    if args.task == "clip":
                        output = bundle.model(
                            mx.array(data["pixels"][None]),
                            mx.array(data["ids"]),
                            mx.array(data["mask"]),
                        )
                        mx.eval(output["logits_per_image"])
                    elif args.task == "ner":
                        encoded = data["encoded"][0]
                        output = bundle.model(
                            mx.array([encoded.ids]), mx.array([encoded.attention_mask])
                        )
                        mx.eval(output.logits)
                    else:
                        import numpy as np

                        pcm = np.pad(data["pcm"], (0, -len(data["pcm"]) % 512))
                        context, state = mx.zeros((1, 64)), mx.zeros((2, 1, 128))
                        for offset in range(0, len(pcm), 512):
                            chunk = mx.array(pcm[offset : offset + 512][None])
                            output, state = bundle.model(
                                mx.concatenate((context, chunk), axis=1), state
                            )
                            context = chunk[:, -64:]
                            mx.eval(output, state, context)

                start = time.perf_counter()
                DeviceExecutor.shared().call("native-task-benchmark", run)
                forward.append((time.perf_counter() - start) * 1000)
        finally:
            lease.release()
        report = {
            "task": task,
            "model": bundle.model_id,
            "revision": bundle.revision,
            "dtype": "float32",
            "batch": 1,
            "processes": 1,
            "warmups": 1,
            "load_and_warmup_ms": load_warmup_ms,
            "python_task": stats(full),
            "preparation": stats(prepare),
            "forward_with_dispatch": stats(forward),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "platform": platform.platform(),
            "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32", "unset"),
            "input": (
                str(args.image)
                if args.task == "clip"
                else "First 16000 samples of pinned Silero test.wav"
                if args.task == "vad"
                else payload["texts"]
            ),
            "warning": (
                "One-process diagnostic; forward includes array creation/dispatch; "
                "components timed separately."
            ),
        }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
