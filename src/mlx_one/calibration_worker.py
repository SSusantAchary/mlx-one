"""Metal-initializing worker used only by the calibration subprocess."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> None:
    """Run one isolated calibration request and write its result."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m mlx_one.calibration_worker REQUEST.json")
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    try:
        workload = request["workload"]
        if workload["kind"] == "inference":
            result = _run_inference(request)
        else:
            result = _run_training(request)
        _write_json(result_path, result)
    except Exception as exc:
        _write_json(result_path, {"error": f"{type(exc).__name__}: {exc}"})
        raise


def _run_inference(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_one.text import TextGenerationOptions, generate, load_text_model

    _set_memory_limit(mx, request["memory_limit_bytes"])
    mx.reset_peak_memory()
    started = time.perf_counter()
    bundle = load_text_model(request["model_path"])
    load_seconds = time.perf_counter() - started
    load_peak = int(mx.get_peak_memory())
    workload = request["workload"]
    tokens = _exact_tokens(bundle.tokenizer, workload["context_length"])
    prompt = bundle.tokenizer.decode(tokens, skip_special_tokens=False)

    def generate_once() -> dict[str, float | int]:
        mx.reset_peak_memory()
        started_at = time.perf_counter()
        response = generate(
            bundle,
            prompt,
            options=TextGenerationOptions(max_tokens=workload["generation_length"]),
        )
        elapsed = time.perf_counter() - started_at
        return {
            "wall_seconds": elapsed,
            "prompt_tokens": int(response.prompt_tokens),
            "generation_tokens": int(response.generation_tokens),
            "prompt_tokens_per_second": float(response.prompt_tokens / elapsed),
            "decode_tokens_per_second": float(response.generation_tokens / elapsed),
            "peak_metal_bytes": int(mx.get_peak_memory()),
        }

    cold = generate_once()
    warm = [generate_once() for _ in range(request["warm_repetitions"])]
    peak = max(load_peak, cold["peak_metal_bytes"], *(item["peak_metal_bytes"] for item in warm))
    return {
        "metrics": {
            "load_seconds": load_seconds,
            "cold": cold,
            "warm_repetitions": len(warm),
            "warm_wall_seconds_mean": mean(item["wall_seconds"] for item in warm),
            "warm_prompt_tokens_per_second_mean": mean(
                item["prompt_tokens_per_second"] for item in warm
            ),
            "warm_decode_tokens_per_second_mean": mean(
                item["decode_tokens_per_second"] for item in warm
            ),
        },
        "memory": {
            "load_peak_metal_bytes": load_peak,
            "peak_metal_bytes": int(peak),
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
    }


def _run_training(request: dict[str, Any]) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_one.schemas import TrainConfig, TrainingMethod
    from mlx_one.text import load_text_model
    from mlx_one.training import MLXTrainingBackend

    _set_memory_limit(mx, request["memory_limit_bytes"])
    workload = request["workload"]
    started = time.perf_counter()
    bundle = load_text_model(request["model_path"])
    load_seconds = time.perf_counter() - started
    load_peak = int(mx.get_peak_memory())
    tokens = _exact_tokens(bundle.tokenizer, workload["context_length"])
    text = bundle.tokenizer.decode(tokens, skip_special_tokens=False)
    data_dir = Path(request["scratch_dir"]) / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps({"text": text}) + "\n" for _ in range(16)), encoding="utf-8"
    )
    total_steps = request["warmup_steps"] + request["measured_steps"]
    config = TrainConfig(
        output_dir=request["scratch_dir"],
        method=TrainingMethod(workload["method"]),
        max_seq_length=workload["context_length"],
        batch_size=workload["batch_size"],
        max_steps=total_steps,
        learning_rate=1e-5,
        lora_rank=workload["lora_rank"],
        lora_alpha=request["lora_scale"] * workload["lora_rank"],
        lora_dropout=request["lora_dropout"],
        lora_layers=workload["lora_layers"],
        gradient_checkpointing=workload["gradient_checkpointing"],
        save_steps=total_steps,
        metadata={"base_quantized": bool(bundle.quantization)},
    )
    del bundle
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    result = MLXTrainingBackend().train(
        model_id=request["model_path"],
        revision="0" * 40,
        data_dir=data_dir,
        config=config,
        resume_adapter=None,
    )
    wall_seconds = time.perf_counter() - started
    measured = max(total_steps - request["warmup_steps"], 1)
    return {
        "metrics": {
            "load_seconds": load_seconds,
            "training_wall_seconds": wall_seconds,
            "warmup_steps": request["warmup_steps"],
            "measured_steps": measured,
            "warm_iterations_per_second_mean": measured / wall_seconds,
            "warm_tokens_per_second_mean": result.tokens_per_second,
        },
        "memory": {
            "load_peak_metal_bytes": load_peak,
            "peak_metal_bytes": result.peak_memory_bytes or int(mx.get_peak_memory()),
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
    }


def _exact_tokens(tokenizer: Any, length: int) -> list[int]:
    seed = tokenizer.encode("mlx-one deterministic calibration sample", add_special_tokens=True)
    if not seed:
        raise ValueError("tokenizer returned no tokens")
    repeated = (seed * ((length // len(seed)) + 1))[:length]
    return [int(token) for token in repeated]


def _set_memory_limit(mx: Any, requested: int) -> None:
    recommended = int(mx.device_info().get("max_recommended_working_set_size", requested))
    mx.set_wired_limit(min(requested, recommended))


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
