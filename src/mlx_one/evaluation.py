"""Deterministic text evaluation with pluggable generation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlx_one.provenance import collect_software_provenance
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
TEXT_CODING_PROFILE = "text-coding-v1"
TEXT_INSTRUCTION_PROFILE = "text-instruction-v1"
TEXT_REASONING_PROFILE = "text-reasoning-v1"
TEXT_PROFILES = (
    TEXT_EXACT_MATCH_PROFILE,
    TEXT_CODING_PROFILE,
    TEXT_INSTRUCTION_PROFILE,
    TEXT_REASONING_PROFILE,
)


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
    if spec.profile not in TEXT_PROFILES:
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
    sample_results = []
    scores: list[float] = []
    for index, (item, output) in enumerate(zip(records, outputs, strict=True)):
        excluded = bool(item.get("excluded", False))
        score = _score(spec.profile, output, item["reference"])
        if not excluded:
            scores.append(score)
        sample_results.append(
            {
                "sample_id": item.get("id") or _sample_id(item["prompt"], index),
                "prompt": item["prompt"],
                "reference": item["reference"],
                "prediction": output,
                "score": score,
                "passed": score == 1.0,
                "excluded": excluded,
                "exclusion_reason": item.get("exclusion_reason"),
            }
        )
    if not scores:
        raise EvaluationError("evaluation has no included samples")
    accuracy = sum(scores) / len(scores)
    metric_name, task_name = _profile_metric(spec.profile)
    interval = _wilson_interval(sum(score == 1.0 for score in scores), len(scores))
    protocol = {
        "profile": spec.profile,
        "normalization": "unicode-nfkc-lower-collapse-whitespace-v1",
        "dataset_revisions": dict(spec.dataset_revisions),
    }
    task = TaskResult(
        task=task_name,
        metrics={
            metric_name: {
                **MetricValue(
                value=accuracy,
                provenance=MetricProvenance.MEASURED,
                unit="ratio",
                protocol=spec.profile,
                ).to_dict(),
                "confidence_interval_95": {"lower": interval[0], "upper": interval[1]},
            }
        },
        sample_count=len(scores),
        metadata={
            "protocol": protocol,
            "failed_samples": sum(score < 1.0 for score in scores),
            "excluded_samples": len(records) - len(scores),
            "samples": sample_results,
        },
    )
    result = RunResult(
        run_id=spec.run_id,
        status=RunStatus.COMPLETED,
        spec=spec,
        software=collect_software_provenance(),
        metrics={f"quality.{metric_name}": task.metrics[metric_name]},
        task_results=(task,),
        timings={
            "wall_seconds": MetricValue(
                value=None,
                provenance=MetricProvenance.NOT_AVAILABLE,
                unit="seconds",
                protocol=spec.profile,
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


def load_evaluation_records(path: str | Path) -> list[dict[str, Any]]:
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
    result: list[dict[str, Any]] = []
    for item in payload:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("prompt"), str)
            or not isinstance(item.get("reference"), str)
        ):
            raise EvaluationError("each evaluation record requires prompt and reference strings")
        normalized = {"prompt": item["prompt"], "reference": item["reference"]}
        if "id" in item:
            if not isinstance(item["id"], str) or not item["id"].strip():
                raise EvaluationError("evaluation record id must be a non-empty string")
            normalized["id"] = item["id"]
        if "excluded" in item:
            if not isinstance(item["excluded"], bool):
                raise EvaluationError("evaluation record excluded must be boolean")
            normalized["excluded"] = item["excluded"]
        if "exclusion_reason" in item:
            if not isinstance(item["exclusion_reason"], str):
                raise EvaluationError("exclusion_reason must be a string")
            normalized["exclusion_reason"] = item["exclusion_reason"]
        result.append(normalized)
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


def _profile_metric(profile: str) -> tuple[str, str]:
    return {
        TEXT_EXACT_MATCH_PROFILE: ("exact_match", "exact-match"),
        TEXT_CODING_PROFILE: ("code_exact_match", "coding"),
        TEXT_INSTRUCTION_PROFILE: ("token_f1", "instruction-following"),
        TEXT_REASONING_PROFILE: ("answer_accuracy", "reasoning"),
    }[profile]


def _score(profile: str, prediction: str, reference: str) -> float:
    if profile == TEXT_INSTRUCTION_PROFILE:
        expected = normalize_answer(reference).split()
        actual = normalize_answer(prediction).split()
        if not expected or not actual:
            return float(expected == actual)
        common = 0
        remaining = list(actual)
        for token in expected:
            if token in remaining:
                common += 1
                remaining.remove(token)
        precision = common / len(actual)
        recall = common / len(expected)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if profile == TEXT_CODING_PROFILE:
        return float(_normalize_code(prediction) == _normalize_code(reference))
    if profile == TEXT_REASONING_PROFILE:
        return float(_final_answer(prediction) == _final_answer(reference))
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _normalize_code(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()[1:-1]
        stripped = "\n".join(lines)
    return "\n".join(line.rstrip() for line in stripped.strip().splitlines())


def _final_answer(value: str) -> str:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return numbers[-1] if numbers else normalize_answer(value)


def _sample_id(prompt: str, index: int) -> str:
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    return f"sample-{index:04d}-{digest}"


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z**2 / count
    centre = (proportion + z**2 / (2 * count)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * count)) / count)
    return max(0.0, centre - margin / denominator), min(1.0, centre + margin / denominator)
