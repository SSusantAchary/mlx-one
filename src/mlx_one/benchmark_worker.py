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

    from mlx_one.text.loading import resolve_text_model_type

    mx.reset_peak_memory()
    loaded_at = time.perf_counter()
    native = resolve_text_model_type(request["model_id"], revision=request["revision"]) == "gpt2"
    if native:
        from mlx_one.text import load_text_model

        bundle = load_text_model(request["model_id"], revision=request["revision"])
    else:
        from mlx_lm import load

        model, tokenizer = load(request["model_id"], revision=request["revision"])
    load_seconds = time.perf_counter() - loaded_at
    load_peak = int(mx.get_peak_memory())

    def once(prompt: str) -> dict[str, float | int]:
        mx.reset_peak_memory()
        started = time.perf_counter()
        if native:
            from mlx_one.text import TextGenerationOptions, generate

            response = generate(
                bundle,
                prompt,
                options=TextGenerationOptions(max_tokens=request["max_tokens"]),
            )
            wall_seconds = time.perf_counter() - started
            prompt_tokens = response.prompt_tokens
            generation_tokens = response.generation_tokens
            prompt_tps = prompt_tokens / wall_seconds
            generation_tps = generation_tokens / wall_seconds
        else:
            from mlx_lm.generate import stream_generate

            responses = list(
                stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=request["max_tokens"],
                )
            )
            if not responses:
                raise RuntimeError("generation produced no response records")
            final = responses[-1]
            wall_seconds = time.perf_counter() - started
            prompt_tokens = int(final.prompt_tokens)
            generation_tokens = int(final.generation_tokens)
            prompt_tps = float(final.prompt_tps)
            generation_tps = float(final.generation_tps)
        return {
            "wall_seconds": wall_seconds,
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
        "mean_prompt_tokens_per_second": statistics.mean(
            item["prompt_tokens_per_second"] for item in samples
        ),
        "mean_decode_tokens_per_second": statistics.mean(
            item["decode_tokens_per_second"] for item in samples
        ),
        "peak_metal_bytes": max(load_peak, *(item["peak_metal_bytes"] for item in samples)),
    }
    Path(request["result_path"]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
