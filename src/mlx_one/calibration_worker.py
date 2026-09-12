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

    try:
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
    except ModuleNotFoundError as exc:
        from mlx_one.compat.dependencies import legacy_mlx_lm_error

        raise legacy_mlx_lm_error("Legacy calibration") from exc

    _set_memory_limit(mx, request["memory_limit_bytes"])
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer = load(request["model_path"])
    load_seconds = time.perf_counter() - started
    load_peak = int(mx.get_peak_memory())
    workload = request["workload"]
    prompt = _exact_tokens(tokenizer, workload["context_length"])

    def generate_once() -> dict[str, float | int]:
        mx.reset_peak_memory()
        started_at = time.perf_counter()
        responses = list(
            stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=workload["generation_length"],
            )
        )
        elapsed = time.perf_counter() - started_at
        final = responses[-1]
        return {
            "wall_seconds": elapsed,
            "prompt_tokens": int(final.prompt_tokens),
            "generation_tokens": int(final.generation_tokens),
            "prompt_tokens_per_second": float(final.prompt_tps),
            "decode_tokens_per_second": float(final.generation_tps),
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
    import mlx.optimizers as optim

    try:
        from mlx_lm import load
        from mlx_lm.tuner.callbacks import TrainingCallback
        from mlx_lm.tuner.datasets import CacheDataset
        from mlx_lm.tuner.trainer import TrainingArgs, train
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ModuleNotFoundError as exc:
        from mlx_one.compat.dependencies import legacy_mlx_lm_error

        raise legacy_mlx_lm_error("Legacy training calibration") from exc

    _set_memory_limit(mx, request["memory_limit_bytes"])
    workload = request["workload"]
    started = time.perf_counter()
    model, tokenizer = load(request["model_path"])
    load_seconds = time.perf_counter() - started
    load_peak = int(mx.get_peak_memory())
    model.freeze()
    linear_to_lora_layers(
        model,
        workload["lora_layers"],
        {
            "rank": workload["lora_rank"],
            "dropout": request["lora_dropout"],
            "scale": request["lora_scale"],
        },
    )
    tokens = _exact_tokens(tokenizer, workload["context_length"])
    dataset = CacheDataset(_TokenDataset(tokens, count=16))
    adapter_file = Path(request["scratch_dir"]) / "adapters.safetensors"
    total_steps = request["warmup_steps"] + request["measured_steps"]
    args = TrainingArgs(
        batch_size=workload["batch_size"],
        iters=total_steps,
        val_batches=0,
        steps_per_report=1,
        steps_per_eval=total_steps + 1,
        steps_per_save=total_steps + 1,
        adapter_file=adapter_file,
        max_seq_length=workload["context_length"],
        grad_checkpoint=workload["gradient_checkpointing"],
        grad_accumulation_steps=1,
    )
    callback = _TrainingMeasurements(mx, TrainingCallback, request["warmup_steps"])
    optimizer = optim.AdamW(learning_rate=1e-5)
    mx.reset_peak_memory()
    started = time.perf_counter()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=dataset,
        val_dataset=None,
        args=args,
        training_callback=callback,
    )
    wall_seconds = time.perf_counter() - started
    warm = callback.reports[request["warmup_steps"] :]
    peak = max((int(item["peak_memory"] * 1e9) for item in callback.reports), default=0)
    return {
        "metrics": {
            "load_seconds": load_seconds,
            "training_wall_seconds": wall_seconds,
            "warmup_steps": request["warmup_steps"],
            "measured_steps": len(warm),
            "warm_iterations_per_second_mean": mean(
                item["iterations_per_second"] for item in warm
            ),
            "warm_tokens_per_second_mean": mean(item["tokens_per_second"] for item in warm),
        },
        "memory": {
            "load_peak_metal_bytes": load_peak,
            "peak_metal_bytes": peak,
            "peak_process_rss_bytes": _peak_rss_bytes(),
        },
    }


class _TokenDataset:
    def __init__(self, tokens: list[int], *, count: int):
        self.tokens = tokens
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, _index: int) -> list[int]:
        return self.tokens

    def process(self, tokens: list[int]) -> tuple[list[int], int]:
        return tokens, 0


def _TrainingMeasurements(mx: Any, base: type, warmup_steps: int) -> Any:
    class Callback(base):
        def __init__(self) -> None:
            self.reports: list[dict[str, Any]] = []

        def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
            self.reports.append(dict(train_info))
            if len(self.reports) == warmup_steps:
                mx.reset_peak_memory()

    return Callback()


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
