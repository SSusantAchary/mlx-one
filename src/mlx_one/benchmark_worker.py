"""Isolated MLX inference benchmark worker."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.benchmark_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    import mlx.core as mx

    mx.reset_peak_memory()
    loaded_at = time.perf_counter()
    from mlx_one.text import load_text_model

    bundle = load_text_model(request["model_id"], revision=request["revision"])
    load_seconds = time.perf_counter() - loaded_at
    load_peak = int(mx.get_peak_memory())

    def once(prompt: str) -> dict[str, float | int]:
        mx.reset_peak_memory()
        started = time.perf_counter()
        from mlx_one.text import TextGenerationOptions, stream_generate

        iterator = stream_generate(
            bundle,
            prompt,
            options=TextGenerationOptions(max_tokens=request["max_tokens"]),
        )
        first = next(iterator)
        ttft_seconds = time.perf_counter() - started
        chunks = [first, *iterator]
        wall_seconds = time.perf_counter() - started
        prompt_tokens = max(len(bundle.tokenizer.encode(prompt)), 1)
        generation_tokens = max(chunks[-1].generated_tokens, 0)
        prompt_tps = prompt_tokens / ttft_seconds
        decode_seconds = max(wall_seconds - ttft_seconds, 1e-12)
        generation_tps = max(generation_tokens - 1, 0) / decode_seconds
        return {
            "wall_seconds": wall_seconds,
            "ttft_seconds": ttft_seconds,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "prompt_tokens_per_second": prompt_tps,
            "decode_tokens_per_second": generation_tps,
            "peak_metal_bytes": int(mx.get_peak_memory()),
        }

    once(request["prompts"][0])
    samples = [once(prompt) for prompt in request["prompts"] for _ in range(request["repeats"])]
    result = {
        "load_seconds": load_seconds,
        "load_peak_metal_bytes": load_peak,
        "sample_count": len(samples),
        "mean_wall_seconds": statistics.mean(item["wall_seconds"] for item in samples),
        "median_ttft_seconds": statistics.median(item["ttft_seconds"] for item in samples),
        "mean_prompt_tokens_per_second": statistics.mean(
            item["prompt_tokens_per_second"] for item in samples
        ),
        "mean_decode_tokens_per_second": statistics.mean(
            item["decode_tokens_per_second"] for item in samples
        ),
        "median_prompt_tokens_per_second": statistics.median(
            item["prompt_tokens_per_second"] for item in samples
        ),
        "median_decode_tokens_per_second": statistics.median(
            item["decode_tokens_per_second"] for item in samples
        ),
        "peak_metal_bytes": max(load_peak, *(item["peak_metal_bytes"] for item in samples)),
    }
    Path(request["result_path"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
