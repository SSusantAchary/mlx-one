"""Reproducible isolated inference benchmarking."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlx_one.hardware import detect_hardware
from mlx_one.run_store import RunStore
from mlx_one.schemas import (
    Failure,
    MetricProvenance,
    MetricValue,
    Modality,
    ModelSpec,
    Operation,
    RunResult,
    RunSpec,
    RunStatus,
)


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot produce valid measured evidence."""


def benchmark_inference(
    model_id: str,
    revision: str,
    prompts: list[str],
    *,
    store: RunStore,
    max_tokens: int = 32,
    repeats: int = 3,
    run_id: str | None = None,
    executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> RunResult:
    """Run warm text inference in an isolated worker and persist measured metrics."""
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise BenchmarkError("benchmarking requires an immutable model revision")
    if not prompts or any(not isinstance(item, str) or not item for item in prompts):
        raise BenchmarkError("at least one non-empty prompt is required")
    if max_tokens < 1 or repeats < 1:
        raise BenchmarkError("max_tokens and repeats must be positive")
    identifier = run_id or f"benchmark-{uuid.uuid4().hex[:12]}"
    spec = RunSpec(
        run_id=identifier,
        operation=Operation.BENCHMARK,
        model=ModelSpec(model_id=model_id, revision=revision, modality=Modality.TEXT),
        profile="text-inference-v1",
        hardware=detect_hardware(),
        generation={"max_tokens": max_tokens, "temperature": 0.0},
        limits={"repeats": repeats, "prompt_count": len(prompts)},
        metadata={"metric_policy": "explicit-provenance-v1"},
    )
    store.create(spec)
    started = _now()
    request = {
        "model_id": model_id,
        "revision": revision,
        "prompts": prompts,
        "max_tokens": max_tokens,
        "repeats": repeats,
    }
    try:
        measured = (executor or _execute_worker)(request)
        required = {
            "load_seconds",
            "sample_count",
            "mean_wall_seconds",
            "mean_prompt_tokens_per_second",
            "mean_decode_tokens_per_second",
            "peak_metal_bytes",
        }
        if not required <= set(measured):
            raise BenchmarkError("benchmark worker omitted required metrics")
    except Exception as exc:
        result = RunResult(
            run_id=identifier,
            status=RunStatus.FAILED,
            spec=spec,
            failures=(Failure(code="benchmark-failed", message=str(exc), retryable=True),),
            started_at=started,
            ended_at=_now(),
        )
        store.save_result(result)
        if isinstance(exc, BenchmarkError):
            raise
        raise BenchmarkError(str(exc)) from exc
    metrics = {
        "performance.prompt_tokens_per_second": _measured(
            measured["mean_prompt_tokens_per_second"], "tokens/second"
        ),
        "performance.decode_tokens_per_second": _measured(
            measured["mean_decode_tokens_per_second"], "tokens/second"
        ),
    }
    result = RunResult(
        run_id=identifier,
        status=RunStatus.COMPLETED,
        spec=spec,
        metrics=metrics,
        timings={
            "load_seconds": _measured(measured["load_seconds"], "seconds"),
            "mean_wall_seconds": _measured(measured["mean_wall_seconds"], "seconds"),
        },
        memory={"peak_metal_bytes": _measured(measured["peak_metal_bytes"], "bytes")},
        started_at=started,
        ended_at=_now(),
    )
    store.save_result(result)
    return result


def load_prompts(path: str | Path) -> list[str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"cannot read prompt file: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("prompts")
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise BenchmarkError("prompt file must contain a JSON string array")
    return payload


def _execute_worker(request: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mlx-one-benchmark-") as temporary:
        root = Path(temporary)
        request_path = root / "request.json"
        result_path = root / "result.json"
        request["result_path"] = str(result_path)
        request_path.write_text(json.dumps(request), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "mlx_one.benchmark_worker", str(request_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "worker failed"
            )
            raise BenchmarkError(f"MLX benchmark failed: {detail}")
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkError(f"invalid benchmark worker result: {exc}") from exc


def _measured(value: int | float, unit: str) -> dict[str, Any]:
    return MetricValue(
        value=value,
        provenance=MetricProvenance.MEASURED,
        unit=unit,
        protocol="text-inference-v1",
    ).to_dict()


def _now() -> str:
    return datetime.now(UTC).isoformat()
