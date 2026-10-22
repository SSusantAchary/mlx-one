"""Repeated real-model requests, cancellation recovery and bounded memory evidence."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path


def run(path, precision):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one import CacheConfig, Engine
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.observation import observe_request
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)
    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    execution = "float16" if precision.startswith("affine") else None

    def prepare():
        nonlocal bundle
        if execution:
            from dataclasses import replace

            bits = int(precision[-1])
            nn.quantize(bundle.model, group_size=64, bits=bits, mode="affine")
            bundle = replace(
                bundle, quantization={"bits": bits, "group_size": 64, "mode": "affine"}
            )
        configure_execution_dtype(bundle, execution)
        mx.eval(bundle.model.parameters())

    report = {
        "timestamp": time.time(),
        "checkpoint": bundle.model_id,
        "revision": bundle.revision,
        "precision": precision,
        "requests": [],
    }
    try:
        manager._pool.executor.call(bundle.model_id, prepare)
        with Engine(
            bundle,
            inference_mode="reference_dense",
            execution_dtype=execution,
            cache_config=CacheConfig(prefix_cache_enabled=False),
        ) as engine:
            prompt = "The city was"
            expected = engine.generate(prompt, max_tokens=8)
            for index in range(12):
                output = engine.generate(prompt, max_tokens=8)
                stats = engine.runtime_stats()["cache_stats"]
                report["requests"].append(
                    {
                        "index": index,
                        "text": output.text,
                        "finish_reason": output.finish_reason,
                        "tokens": output.generation_tokens,
                        "same_output": output.text == expected.text,
                        "cache_stats": stats,
                        "active_mlx_bytes": mx.get_active_memory(),
                        "cached_mlx_bytes": mx.get_cache_memory(),
                    }
                )
                assert stats["active_handles"] == stats["reserved_bytes"] == 0
            try:
                engine.generate(" a" * bundle.context_length, max_tokens=8)
            except ValueError:
                report["failed_request_rejected"] = True
            else:
                raise AssertionError("context overflow was accepted")
            report["after_failure_same"] = (
                engine.generate(prompt, max_tokens=8).text == expected.text
            )
            ready = threading.Event()

            def observer(event):
                if event["kind"] == "token":
                    ready.set()

            with observe_request(observer):
                handle = engine.submit(prompt, max_tokens=256)
            assert ready.wait(30), "generation did not start"
            was_running = not handle.done()
            cancelled = handle.cancel()
            cancelled_output = handle.result(timeout=60)
            report["cancellation"] = {
                "was_running": was_running,
                "cancel_returned": cancelled,
                "generated_tokens": cancelled_output.generation_tokens,
                "finish_reason": cancelled_output.finish_reason,
            }
            report["after_cancel_same"] = (
                engine.generate(prompt, max_tokens=8).text == expected.text
            )
            report["cleanup"] = engine.runtime_stats()["cache_stats"]
            memory = [r["active_mlx_bytes"] for r in report["requests"]]
            report["active_memory_growth_bytes"] = max(memory[-3:]) - max(memory[:3])
            report["memory_bound_bytes"] = 8 * 1024**2
            report["status"] = (
                "PASS"
                if all(r["same_output"] for r in report["requests"])
                and (
                    report["after_failure_same"]
                    and report["after_cancel_same"]
                    and report["cleanup"]["active_handles"]
                    == report["cleanup"]["reserved_bytes"]
                    == 0
                    and report["active_memory_growth_bytes"] <= report["memory_bound_bytes"]
                    and was_running
                    and cancelled
                )
                else "FAIL"
            )
        return report
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    try:
        result = run(args.checkpoint, args.precision)
    except Exception as exc:
        import traceback

        result = {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
