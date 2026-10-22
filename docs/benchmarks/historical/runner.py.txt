"""Single-model diagnostic performance with natural versus production FP16 KV."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path


def run(args):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)

    def compute():
        nonlocal bundle
        if args.precision.startswith("affine"):
            bits = int(args.precision[-1])
            nn.quantize(bundle.model, group_size=64, bits=bits)
            bundle = replace(bundle, quantization={"bits": bits, "group_size": 64})
            configure_execution_dtype(bundle, "float16")
        model = bundle.model
        model.eval()
        mx.eval(model.parameters())
        seed = bundle.tokenizer.encode("The red book was on the table. The city is calm. ")
        result = {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": args.precision,
            "timestamp": time.time(),
            "reference_resident": False,
            "diagnostic_direct_model": True,
            "public_engine_benchmark": False,
            "output_budget": 16,
            "rows": [],
        }

        def request(n, mode):
            cache = model.make_cache()
            if mode == "production_fp16":
                cache = configure_kv_caches(cache, "f16", "f16")
            ids = (seed * ((n + len(seed) - 1) // len(seed)))[:n]
            mx.clear_cache()
            mx.reset_peak_memory()
            begin = time.perf_counter()
            output = model(mx.array([ids]), cache=cache).logits
            token = mx.argmax(output[0, -1])
            mx.eval(token)
            prefill = time.perf_counter() - begin
            generated = [int(token)]
            del output
            first = time.perf_counter()
            peak_state = sum(c.nbytes for c in cache)
            for _ in range(15):
                output = model(mx.array([[generated[-1]]]), cache=cache).logits
                token = mx.argmax(output[0, -1])
                mx.eval(token)
                generated.append(int(token))
                del output
                peak_state = max(peak_state, sum(c.nbytes for c in cache))
                if generated[-1] in bundle.eos_token_ids:
                    break
            decode = time.perf_counter() - first
            peak = int(mx.get_peak_memory())
            for c in cache:
                c.reset()
            clean = all(c.offset == 0 and c.nbytes == 0 for c in cache)
            if not clean:
                raise AssertionError("cache cleanup failed")
            return {
                "input_tokens": n,
                "generated_tokens": len(generated),
                "token_ids": generated,
                "ttft_ms": prefill * 1000,
                "prefill_tokens_per_second": n / prefill,
                "decode_tokens_per_second": (len(generated) - 1) / decode,
                "peak_inference_mlx_bytes": peak,
                "peak_state_bytes": peak_state,
                "cleanup": clean,
            }

        for n in [128, 512, 1024]:
            if n + 16 > bundle.context_length:
                result["rows"].append({"context": n, "status": "UNSUPPORTED"})
                continue
            for mode in ["natural", "production_fp16"]:
                request(n, mode)
                samples = [request(n, mode) for _ in range(3)]
                result["rows"].append({"context": n, "mode": mode, "samples": samples})
        result["status"] = "MEASURED_NOT_RELEASE_APPROVED"
        return result

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--precision", default="stored")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    a.output.write_text(json.dumps(run(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
