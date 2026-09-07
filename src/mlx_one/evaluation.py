"""Deterministic text evaluation with pluggable generation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlx_one.run_store import RunStore
from mlx_one.schemas import (
    MetricProvenance,
    MetricValue,
    RunResult,
    RunSpec,
    RunStatus,
    TaskResult,
)

TEXT_EXACT_MATCH_PROFILE = "text-exact-match-v1"


class EvaluationError(RuntimeError):
    """Raised when an evaluation protocol or input is invalid."""


def evaluate_text(
    spec: RunSpec,
    dataset: str | Path,
    *,
    store: RunStore,
    predictions: str | Path | None = None,
    predictor: Callable[[str], str] | None = None,
    generated_predictions: list[str] | None = None,
) -> RunResult:
    """Evaluate prompt/reference JSON(L) using predictions or a supplied predictor."""
    if spec.operation.value != "evaluate":
        raise EvaluationError("RunSpec operation must be evaluate")
    if spec.profile != TEXT_EXACT_MATCH_PROFILE:
        raise EvaluationError(f"unsupported evaluation profile: {spec.profile!r}")
    records = load_evaluation_records(dataset)
    supplied = _predictions(predictions) if predictions is not None else None
    if supplied is not None and generated_predictions is not None:
        raise EvaluationError("provide only one prediction source")
    if generated_predictions is not None:
        supplied = generated_predictions
    if supplied is not None and len(supplied) != len(records):
        raise EvaluationError("prediction count must match dataset record count")
    if supplied is None and predictor is None:
        raise EvaluationError("provide predictions or a predictor")
    started = _now()
    outputs = supplied or [predictor(item["prompt"]) for item in records]  # type: ignore[misc]
    matches = [
        normalize_answer(output) == normalize_answer(item["reference"])
        for item, output in zip(records, outputs, strict=True)
    ]
    accuracy = sum(matches) / len(matches)
    protocol = {
        "profile": TEXT_EXACT_MATCH_PROFILE,
        "normalization": "unicode-nfkc-lower-collapse-whitespace-v1",
        "dataset_revisions": dict(spec.dataset_revisions),
    }
    task = TaskResult(
        task="exact-match",
        metrics={
            "exact_match": MetricValue(
                value=accuracy,
                provenance=MetricProvenance.MEASURED,
                unit="ratio",
                protocol=TEXT_EXACT_MATCH_PROFILE,
            ).to_dict()
        },
        sample_count=len(records),
        metadata={"protocol": protocol, "failed_samples": matches.count(False)},
    )
    result = RunResult(
        run_id=spec.run_id,
        status=RunStatus.COMPLETED,
        spec=spec,
        metrics={"quality.exact_match": task.metrics["exact_match"]},
        task_results=(task,),
        timings={
            "wall_seconds": MetricValue(
                value=None,
                provenance=MetricProvenance.NOT_AVAILABLE,
                unit="seconds",
                protocol=TEXT_EXACT_MATCH_PROFILE,
            ).to_dict()
        },
        started_at=started,
        ended_at=_now(),
    )
    store.save_result(result)
    return result


def normalize_answer(value: str) -> str:
    """Apply the pinned exact-match normalization protocol."""
    normalized = unicodedata.normalize("NFKC", value).lower().strip()
    return re.sub(r"\s+", " ", normalized)


def load_evaluation_records(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            payload: Any = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("data")
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read evaluation dataset: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvaluationError("evaluation dataset must contain records")
    result = []
    for item in payload:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("prompt"), str)
            or not isinstance(item.get("reference"), str)
        ):
            raise EvaluationError("each evaluation record requires prompt and reference strings")
        result.append({"prompt": item["prompt"], "reference": item["reference"]})
    return result


def _predictions(path: str | Path) -> list[str]:
    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            payload: Any = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read predictions: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("predictions")
    if not isinstance(payload, list):
        raise EvaluationError("predictions must be a JSON array or JSONL")
    values = [item.get("prediction") if isinstance(item, dict) else item for item in payload]
    if any(not isinstance(item, str) for item in values):
        raise EvaluationError("each prediction must be a string")
    return values


def _now() -> str:
    return datetime.now(UTC).isoformat()
