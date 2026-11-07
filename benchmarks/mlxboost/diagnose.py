"""Cached-checkpoint B1 timings and a retained device-thread Python profile.

Run with the MLX-One development interpreter and Metal access. This is a native
diagnostic, separate from the frozen HTTP protocol. No weight conversion occurs.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen2", "qwen3", "smollm2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mlx.core as mx

    from mlx_one.engine import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.text import TextGenerationOptions, load_text_model
    from mlx_one.text.generation import _stream_bundle

    manifest = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())
    model = next(m for m in manifest["models"] if m["key"] == args.model)
    from evidence import snapshot

    snapshot(args.output, model=model)
    bundle = load_text_model(model["path"], offline=True)
    configure_execution_dtype(bundle, model["execution_dtype"])
    prompt = model["prompts"][1]
    if bundle.tokenizer.encode(prompt["text"]) != prompt["token_ids"]:
        raise ValueError("prompt token identity changed")
    options = TextGenerationOptions(max_tokens=128, temperature=0, seed=0)
    cache_type = "f16" if model["execution_dtype"] == "float16" else "bf16"
    engine = Engine(
        bundle,
        inference_mode="reference_dense",
        execution_dtype=model["execution_dtype"],
        cache_config=CacheConfig(
            key_type=cache_type, value_type=cache_type, prefix_cache_enabled=False
        ),
    )

    def direct():
        cache = bundle.model.make_cache()
        tokens, times = [], []
        current = mx.array([prompt["token_ids"]])
        mx.synchronize()
        started = time.perf_counter()
        for _ in range(128):
            logits = bundle.model(current, cache=cache).logits
            token = mx.argmax(logits[0, -1])
            mx.eval(token)
            value = int(token.item())
            times.append(time.perf_counter() - started)
            tokens.append(value)
            if value in bundle.eos_token_ids:
                break
            current = mx.array([[value]])
        mx.synchronize()
        return {
            "wall_seconds": time.perf_counter() - started,
            "tokens": tokens,
            "first_model_token_seconds": times[0],
            "decode_tokens_per_second": (len(times) - 1) / (times[-1] - times[0]),
            "token_times_seconds": times,
            "offsets": [c.offset for c in cache],
        }

    def generation(managed):
        events = []
        mx.synchronize()
        started = time.perf_counter()
        chunks = list(
            engine.stream(prompt["text"], options=options)
            if managed
            else _stream_bundle(
                bundle,
                prompt["text"],
                options,
                None,
                cache_type_k=cache_type,
                cache_type_v=cache_type,
                on_execution=events.append,
            )
        )
        mx.synchronize()
        return {
            "wall_seconds": time.perf_counter() - started,
            "tokens": [c.token_id for c in chunks if c.token_id is not None],
            "text": "".join(c.text for c in chunks),
            "finish_reason": chunks[-1].finish_reason,
            "events": events,
        }

    results = {
        "command": sys.argv,
        "model": model,
        "device": mx.device_info(),
        "versions": {p: importlib.metadata.version(p) for p in ["mlx", "mlx-metal"]},
        "samples": {},
        "thermal": "unavailable",
        "gpu_capture": "not performed",
    }
    try:
        for label, operation in [
            ("direct_forward", direct),
            ("native_generation", lambda: generation(False)),
            ("engine_generation", lambda: generation(True)),
        ]:
            engine._device.call("warmup", operation)
            samples = []
            for _ in range(args.repetitions):
                sample = engine._device.call("measurement", operation)
                sample["tokens_per_second"] = len(sample["tokens"]) / sample["wall_seconds"]
                samples.append(sample)
            results["samples"][label] = samples
            print(label, statistics.median(s["tokens_per_second"] for s in samples), flush=True)
        profile = cProfile.Profile()
        engine._device.call("profile", lambda: profile.runcall(generation, True))
        profile.dump_stats(str(args.output / "device-thread.pstats"))
        with (args.output / "device-thread.txt").open("w") as handle:
            pstats.Stats(profile, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(
                70
            )
        results["active_bytes"] = mx.get_active_memory()
        results["peak_bytes"] = mx.get_peak_memory()
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        snapshot(args.output, model=model, status="MEASURED")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
