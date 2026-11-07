"""Public Engine flag-off/on timings, exact outputs and per-forward state parity."""

from __future__ import annotations

import argparse
import json
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
    import numpy as np
    from evidence import snapshot

    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.observation import observe_request
    from mlx_one.mlxboost import MLXBoostConfig
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.registry import InferenceFeatureRegistry
    from mlx_one.text import TextGenerationOptions, load_text_model

    models = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())["models"]
    model = next(m for m in models if m["key"] == args.model)
    snapshot(args.output, model=model)
    bundle = load_text_model(model["path"], offline=True)
    prompt = model["prompts"][1]["text"]
    cache_type = "f16" if model["execution_dtype"] == "float16" else "bf16"
    options = TextGenerationOptions(max_tokens=128, temperature=0, seed=0)
    paths, traces = {}, {}

    class Observer:
        capture_state = True
        profile_math = False

        def __init__(self):
            self.tensors, self.widths, self.offsets = {}, [], []

        def __call__(self, event):
            if event["kind"] != "state":
                return
            step = len(self.widths)
            self.widths.append(len(event["input_ids"]))
            self.offsets.append([c.offset for c in event["cache"]])
            self.tensors[f"logits.{step}"] = np.array(event["logits"][:, -1].astype(mx.float32))
            if step in {0, 15, 127}:
                for layer, cache in enumerate(event["cache"]):
                    state = (
                        cache.state()
                        if callable(getattr(cache, "state", None))
                        else (
                            cache._keys[:, :, : cache.offset],
                            cache._values[:, :, : cache.offset],
                        )
                    )
                    for kind, tensor in zip(["K", "V"], state, strict=True):
                        self.tensors[f"{kind}.{step}.{layer}"] = np.array(tensor.astype(mx.float32))

    for label, mode, enabled in [
        ("reference", "reference_dense", False),
        ("boost_off", "mlxboost_b1", False),
        ("fast_decode", "mlxboost_b1", True),
    ]:
        engine = Engine(
            bundle,
            inference_mode=mode,
            execution_dtype=model["execution_dtype"],
            cache_config=CacheConfig(
                key_type=cache_type, value_type=cache_type, prefix_cache_enabled=False
            ),
            qualification_registry=InferenceFeatureRegistry(),
            mlxboost_config=MLXBoostConfig(fast_decode=enabled),
        )
        try:
            engine.capabilities()  # Mirrors serving's non-calibration feature records.
            list(engine.stream(prompt, options=options))
            samples = []
            for _ in range(args.repetitions):
                started = time.perf_counter()
                chunks = list(engine.stream(prompt, options=options))
                elapsed = time.perf_counter() - started
                ids = [c.token_id for c in chunks if c.token_id is not None]
                samples.append(
                    {
                        "wall_seconds": elapsed,
                        "tokens_per_second": len(ids) / elapsed,
                        "tokens": ids,
                        "text": "".join(c.text for c in chunks),
                        "finish_reason": chunks[-1].finish_reason,
                        "terminal_token_id": chunks[-1].terminal_token_id,
                        "generated_tokens": chunks[-1].generated_tokens,
                    }
                )
            observer = Observer()

            def capture(observer=observer, engine=engine):
                with observe_request(observer):
                    return list(engine.stream(prompt, options=options))

            engine._device.call("parity", capture)
            traces[label] = observer.tensors
            paths[label] = {
                "samples": samples,
                "widths": observer.widths,
                "offsets": observer.offsets,
                "selection": engine._selection,
                "flags": vars(engine._cache_manager.mlxboost_config),
                "active_bytes": mx.get_active_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }
            print(label, statistics.median(s["tokens_per_second"] for s in samples), flush=True)
        finally:
            engine.close()
    comparisons = []
    for label in ["boost_off", "fast_decode"]:
        for name, reference in traces["reference"].items():
            comparisons.append(
                {
                    "candidate": label,
                    "name": name,
                    **compare_tensors(
                        reference, traces[label][name], dtype=model["execution_dtype"]
                    ),
                }
            )
    parity = all(c["status"] == "PASS" for c in comparisons)
    output_equal = all(
        paths["reference"]["samples"][0][key] == sample[key]
        for label in paths
        for sample in paths[label]["samples"]
        for key in ["tokens", "text", "finish_reason", "terminal_token_id", "generated_tokens"]
    )
    result = {
        "paths": paths,
        "comparisons": comparisons,
        "state_parity": parity,
        "exact_output_parity": output_equal,
        "model": model,
        "scope": "self comparison; source/cached-fresh baseline gates remain independent",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    # Retain raw final logits and selected K/V states for independent replay.
    for label, tensors in traces.items():
        np.savez_compressed(
            args.output / f"{label}.npz",
            **{
                k: v
                for k, v in tensors.items()
                if not k.startswith("logits.") or k in {"logits.0", "logits.15", "logits.127"}
            },
        )
    snapshot(args.output, model=model, status="PASS" if parity and output_equal else "FAIL")
    if not parity or not output_equal:
        raise SystemExit("flag comparison failed; rollback required")


if __name__ == "__main__":
    main()
