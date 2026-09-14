"""Run pinned native real-model gates and emit a promotion-compatible report."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import median

QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
QWEN_VL_REVISION = "01af461cdb9574acc09084a0ef94e216e142b085"
WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _timed(function: Callable[[], None], repeats: int = 1) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return median(values)


def _qualify_minicpm(target: str) -> dict[str, float]:
    selector = "1b" if target.endswith("1b") else "2b"
    elapsed = _timed(
        lambda: subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_native_minicpm5_integration.py",
                "-k",
                selector,
            ],
            check=True,
        )
    )
    return {f"{target}_checkpoint_smoke_seconds": elapsed}


def _qualify_whisper() -> dict[str, float]:
    from mlx_one.models.audio.whisper.loading import load_whisper

    elapsed = _timed(
        lambda: load_whisper("openai/whisper-tiny", revision=WHISPER_REVISION)
    )
    return {"whisper_load_seconds": elapsed}


def _qualify_vlm() -> dict[str, float]:
    from PIL import Image

    from mlx_one.text import TextGenerationOptions
    from mlx_one.vision import load_vlm_model, stream_vlm

    started = time.perf_counter()
    bundle = load_vlm_model(
        "mlx-community/Qwen2-VL-2B-Instruct-4bit", revision=QWEN_VL_REVISION
    )
    load_seconds = time.perf_counter() - started
    image = Image.new("RGB", (56, 56), color=(127, 127, 127))
    decode_seconds = _timed(
        lambda: tuple(
            stream_vlm(
                bundle,
                "<image>Describe this image.",
                [image],
                options=TextGenerationOptions(max_tokens=2),
            )
        )
    )
    return {"vlm_load_seconds": load_seconds, "vlm_decode_seconds": decode_seconds}


def _qualify_retrieval() -> dict[str, float]:
    from mlx_one.retrieval import embed, load_retrieval_model, rerank

    timings = {}
    started = time.perf_counter()
    embedding = load_retrieval_model(
        "Qwen/Qwen3-Embedding-0.6B", revision=EMBEDDING_REVISION, task="embedding"
    )
    timings["qwen_embedding_load_seconds"] = time.perf_counter() - started
    embed(embedding, ["native MLX"], max_length=32)
    started = time.perf_counter()
    reranker = load_retrieval_model(
        "Qwen/Qwen3-Reranker-0.6B", revision=RERANKER_REVISION, task="reranking"
    )
    timings["qwen_reranker_load_seconds"] = time.perf_counter() - started
    rerank(reranker, "MLX", ["Apple MLX"], max_length=128)
    started = time.perf_counter()
    minilm = load_retrieval_model(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision=MINILM_REVISION,
        task="embedding",
    )
    timings["minilm_load_seconds"] = time.perf_counter() - started
    embed(minilm, ["native MLX"], max_length=32)
    return timings


def _generation_metrics(model_path: Path) -> dict[str, float]:
    import mlx.core as mx

    from mlx_one.text import TextGenerationOptions, load_text_model, stream_generate

    mx.reset_peak_memory()
    started = time.perf_counter()
    bundle = load_text_model(model_path)
    load_seconds = time.perf_counter() - started
    prompt = "Apple MLX native qualification"
    prompt_tokens = max(len(bundle.tokenizer.encode(prompt)), 1)
    ttft, prefill, decode = [], [], []
    for _ in range(3):
        started = time.perf_counter()
        iterator = stream_generate(
            bundle, prompt, options=TextGenerationOptions(max_tokens=8)
        )
        first = next(iterator)
        first_seconds = time.perf_counter() - started
        chunks = [first, *iterator]
        wall = time.perf_counter() - started
        generated = chunks[-1].generated_tokens
        ttft.append(first_seconds)
        prefill.append(prompt_tokens / first_seconds)
        decode.append(max(generated - 1, 0) / max(wall - first_seconds, 1e-12))
    return {
        "text_load_seconds": load_seconds,
        "text_ttft_seconds": median(ttft),
        "text_prefill_tokens_per_second": median(prefill),
        "text_decode_tokens_per_second": median(decode),
        "text_peak_memory_bytes": float(mx.get_peak_memory()),
    }


def _qualify_training() -> dict[str, float]:
    from mlx_one.conversion import convert_text_model
    from mlx_one.schemas import TrainConfig, TrainingMethod
    from mlx_one.training import MLXTrainingBackend

    with tempfile.TemporaryDirectory(prefix="mlx-one-qualification-") as temporary:
        root = Path(temporary)
        data = root / "data"
        data.mkdir()
        (data / "train.jsonl").write_text(
            "".join(
                json.dumps({"text": "Apple MLX native training sample."}) + "\n"
                for _ in range(2)
            ),
            encoding="utf-8",
        )
        timings = {}
        for method, quantize in ((TrainingMethod.LORA, False), (TrainingMethod.QLORA, True)):
            converted = root / method.value
            started = time.perf_counter()
            convert_text_model(
                "Qwen/Qwen2.5-0.5B",
                converted,
                revision=QWEN_REVISION,
                quantize=quantize,
            )
            timings[f"{method.value}_conversion_seconds"] = time.perf_counter() - started
            if method is TrainingMethod.LORA:
                timings.update(_generation_metrics(converted))
            config = TrainConfig(
                output_dir=str(root / f"{method.value}-output"),
                method=method,
                max_seq_length=32,
                batch_size=1,
                max_steps=2,
                lora_layers=1,
                lora_rank=4,
                lora_alpha=8,
            )
            started = time.perf_counter()
            result = MLXTrainingBackend().train(
                model_id=str(converted),
                revision="0" * 40,
                data_dir=data,
                config=config,
                resume_adapter=None,
            )
            timings[f"{method.value}_training_step_seconds"] = (
                time.perf_counter() - started
            ) / config.max_steps
            timings[f"{method.value}_peak_memory_bytes"] = float(
                result.peak_memory_bytes or 0
            )
        return timings


QUALIFIERS = {
    "qwen2.5-0.5b-native-training": _qualify_training,
    "minicpm5-1b": lambda: _qualify_minicpm("minicpm5-1b"),
    "minicpm5-2b": lambda: _qualify_minicpm("minicpm5-2b"),
    "qwen2-vl-2b": _qualify_vlm,
    "whisper-tiny": _qualify_whisper,
    "retrieval": _qualify_retrieval,
}


def _performance_verdict(root: Path, metrics: dict[str, float]) -> tuple[str, float]:
    baseline_path = root / "compatibility/baselines/m4-32gb.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
    comparable = set(metrics) & set(baseline)
    if not comparable:
        return "investigation-required", 0.0
    regressions = []
    for name in comparable:
        if baseline[name] <= 0:
            raise ValueError(f"performance baseline must be positive: {name}")
        difference = (
            baseline[name] - metrics[name]
            if name.endswith("tokens_per_second")
            else metrics[name] - baseline[name]
        )
        regressions.append(difference / baseline[name] * 100)
    maximum = max(regressions)
    if maximum > 10:
        return "degraded", maximum
    if maximum >= 5:
        return "investigation-required", maximum
    return "acceptable", maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["full", *QUALIFIERS])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    targets = tuple(QUALIFIERS) if args.target == "full" else (args.target,)
    capabilities = {}
    metrics = {}
    failure = None
    for target in targets:
        try:
            metrics.update(QUALIFIERS[target]())
            capabilities[target] = True
        except Exception as exc:
            capabilities[target] = False
            failure = f"{target}: {exc}"
            break
    verdict, regression = _performance_verdict(root, metrics)
    try:
        mlx_version = version("mlx")
    except Exception:
        mlx_version = "unavailable"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True, cwd=root
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "mlx_version": mlx_version,
        "qualified_at": datetime.now(UTC).isoformat(),
        "repository_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "hardware": platform.machine(),
            "profile": "Apple M4 32 GB",
        },
        "capabilities": capabilities,
        "performance": {
            "verdict": verdict,
            "maximum_regression_percent": regression,
            "medians": metrics,
        },
        "failure": failure,
        "promotion_eligible": args.target == "full" and failure is None and verdict == "acceptable",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failure:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
