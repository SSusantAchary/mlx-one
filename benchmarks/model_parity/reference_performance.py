"""B1 reference-dense characterization with inference/conversion memory separated."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path


def run(path, precision, contexts, outputs, repeats, progress):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one import Engine, TextGenerationOptions
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.observation import observe_request
    from mlx_one.engine.timing import RequestTiming
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    started = time.perf_counter()
    manager = ModelManager()
    bundle = manager.load(path, offline=True)
    report = {
        "checkpoint": bundle.model_id,
        "revision": bundle.revision,
        "precision": precision,
        "cold_load_seconds": time.perf_counter() - started,
        "mode": "reference_dense",
        "batch_size": 1,
        "repeats": repeats,
        "qualification": "CHARACTERIZATION_ONLY_REQUIRES_NUMERICAL_GATES",
        "rows": [],
    }
    execution = "float16" if precision.startswith("affine_q") else None

    def prepare():
        nonlocal bundle
        before = time.perf_counter()
        if precision.startswith("affine_q"):
            bits = int(precision[-1])
            nn.quantize(bundle.model, group_size=64, bits=bits, mode="affine")
            bundle = replace(
                bundle, quantization={"bits": bits, "group_size": 64, "mode": "affine"}
            )
            manager._bundle = bundle
        configure_execution_dtype(bundle, execution)
        mx.eval(bundle.model.parameters())
        report["conversion_seconds"] = time.perf_counter() - before
        report["load_conversion_peak_mlx_bytes"] = mx.get_peak_memory()
        report["load_conversion_process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        gc.collect()
        mx.clear_cache()

    try:
        manager._pool.executor.call(bundle.model_id, prepare)
        with Engine(
            bundle,
            execution_dtype=execution,
            inference_mode="reference_dense",
            cache_config=CacheConfig(prefix_cache_enabled=False),
        ) as engine:
            engine.generate("Warmup", options=TextGenerationOptions(max_tokens=1))
            for context in contexts:
                prompt = " a" * context
                encoded = len(bundle.tokenizer.encode(prompt))
                for output in outputs:
                    row = {
                        "requested_context": context,
                        "encoded_tokens": encoded,
                        "output_budget": output,
                        "samples": [],
                    }
                    if encoded + output > bundle.context_length:
                        row.update(status="UNSUPPORTED", reason="checkpoint context limit")
                        report["rows"].append(row)
                        progress(report)
                        continue
                    # Warm this prefill shape before recording warm TTFT.
                    engine.generate(prompt, options=TextGenerationOptions(max_tokens=1))
                    for _ in range(repeats):
                        sampled_rss, stop = [], threading.Event()

                        def sample_rss(stop=stop, sampled_rss=sampled_rss):
                            while not stop.is_set():
                                sample = subprocess.run(
                                    ["ps", "-o", "rss=", "-p", str(os.getpid())],
                                    capture_output=True,
                                    text=True,
                                    check=False,
                                )
                                if sample.stdout.strip():
                                    sampled_rss.append(int(sample.stdout.strip()) * 1024)
                                stop.wait(0.05)

                        thread = threading.Thread(target=sample_rss, daemon=True)
                        thread.start()
                        timing = RequestTiming(time.perf_counter(), trace=True)
                        peak_cache = 0

                        def observe(event, timing=timing):
                            nonlocal peak_cache
                            timing.observe(event)
                            if event["kind"] == "cache_bytes":
                                peak_cache = max(peak_cache, event["bytes"])

                        observe.capture_bytes = True
                        mx.reset_peak_memory()
                        try:
                            with observe_request(observe):
                                result = engine.generate(
                                    prompt,
                                    options=TextGenerationOptions(max_tokens=output, temperature=0),
                                )
                        finally:
                            stop.set()
                            thread.join(timeout=2)
                        sample = timing.summary()
                        prefill = timing.forward_ms["prefill"]
                        sample.update(
                            input_tokens=timing.prefill_tokens,
                            generated_tokens=result.generation_tokens,
                            prefill_ms=prefill,
                            prefill_tokens_per_second=timing.prefill_tokens * 1000 / prefill
                            if prefill
                            else None,
                            peak_inference_mlx_bytes=mx.get_peak_memory(),
                            sampled_inference_peak_rss_bytes=max(sampled_rss, default=None),
                            peak_cache_state_bytes=peak_cache,
                            finish_reason=result.finish_reason,
                            token_ids=list(timing.token_ids),
                            actual_logits_dtypes=sorted(timing.dtypes),
                            cleanup=engine.runtime_stats()["cache_stats"],
                        )
                        row["samples"].append(sample)
                    row["status"] = (
                        "PASS"
                        if all(
                            s["cleanup"]["active_handles"] == 0
                            and s["cleanup"]["reserved_bytes"] == 0
                            for s in row["samples"]
                        )
                        else "FAIL_STATE"
                    )
                    report["rows"].append(row)
                    progress(report)
        report["status"] = (
            "PASS"
            if all(row["status"] in {"PASS", "UNSUPPORTED"} for row in report["rows"])
            else "FAIL_STATE"
        )
        report["memory_contract"] = (
            "Peak MLX resets after conversion/warmup and before each inference request. "
            "RSS is sampled at 50ms intervals during inference; lifetime process RSS peaks "
            "are separately labelled load/conversion and cannot be reset. Cache bytes are "
            "observed during forwards, before cleanup. No reference model is loaded."
        )
        return report
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--contexts", default="16,128,512,1024,2048,4096")
    parser.add_argument("--outputs", default="16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence path")
    if args.repeats < 1:
        parser.error("repeats must be positive")

    def save(report):
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    result = run(
        args.checkpoint,
        args.precision,
        [int(x) for x in args.contexts.split(",")],
        [int(x) for x in args.outputs.split(",")],
        args.repeats,
        save,
    )
    save(result)
    print(result["checkpoint"], result["status"], len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
