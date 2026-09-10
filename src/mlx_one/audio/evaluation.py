"""Pinned WER and CER evaluation for automatic speech recognition."""

from __future__ import annotations

import json
import re
import unicodedata
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

WHISPER_WER_PROFILE = "whisper-wer-v1"
WHISPER_CER_PROFILE = "whisper-cer-v1"
ASR_PROFILES = (WHISPER_WER_PROFILE, WHISPER_CER_PROFILE)


class ASREvaluationError(RuntimeError):
    """Raised when an ASR evaluation input or profile is invalid."""


def normalize_transcript(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\s']", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, prediction: str) -> float:
    expected = normalize_transcript(reference).split()
    actual = normalize_transcript(prediction).split()
    return edit_distance(expected, actual) / max(len(expected), 1)


def character_error_rate(reference: str, prediction: str) -> float:
    expected = list(normalize_transcript(reference).replace(" ", ""))
    actual = list(normalize_transcript(prediction).replace(" ", ""))
    return edit_distance(expected, actual) / max(len(expected), 1)


def load_asr_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        if source.suffix.lower() == ".jsonl":
            value: Any = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            value = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("data")
    except (OSError, json.JSONDecodeError) as exc:
        raise ASREvaluationError(f"cannot read ASR dataset: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ASREvaluationError("ASR dataset must contain records")
    records = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("audio"), str):
            raise ASREvaluationError("each ASR record requires an audio path")
        if not isinstance(item.get("reference"), str):
            raise ASREvaluationError("each ASR record requires reference text")
        record = dict(item)
        audio = Path(record["audio"])
        if not audio.is_absolute():
            record["audio"] = str((source.parent / audio).resolve())
        records.append(record)
    return records


def score_asr_records(
    records: list[dict[str, Any]], predictions: list[str], profile: str
) -> dict[str, Any]:
    if profile not in ASR_PROFILES:
        raise ASREvaluationError(f"unsupported ASR profile: {profile}")
    if len(records) != len(predictions):
        raise ASREvaluationError("prediction count must match ASR record count")
    scorer = word_error_rate if profile == WHISPER_WER_PROFILE else character_error_rate
    scores = [
        scorer(record["reference"], prediction)
        for record, prediction in zip(records, predictions, strict=True)
        if not record.get("excluded", False)
    ]
    if not scores:
        raise ASREvaluationError("ASR evaluation has no included records")
    return {
        "profile": profile,
        "value": sum(scores) / len(scores),
        "sample_count": len(scores),
        "normalization": "unicode-nfkc-lower-word-punctuation-v1",
        "sample_scores": scores,
    }


def load_asr_predictions(path: str | Path) -> list[str]:
    try:
        value: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ASREvaluationError(f"cannot read ASR predictions: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("predictions")
    if not isinstance(value, list):
        raise ASREvaluationError("ASR predictions must be a JSON array")
    result = [item.get("prediction") if isinstance(item, dict) else item for item in value]
    if any(not isinstance(item, str) for item in result):
        raise ASREvaluationError("each ASR prediction must be text")
    return result


def evaluate_asr(
    spec: RunSpec,
    dataset: str | Path,
    *,
    store: RunStore,
    predictions: str | Path | None = None,
    generated_predictions: list[str] | None = None,
) -> RunResult:
    records = load_asr_records(dataset)
    supplied = (
        load_asr_predictions(predictions)
        if predictions is not None
        else generated_predictions
    )
    if supplied is None:
        raise ASREvaluationError("provide ASR predictions or generated predictions")
    score = score_asr_records(records, supplied, spec.profile or "")
    metric_name = "wer" if spec.profile == WHISPER_WER_PROFILE else "cer"
    started = datetime.now(UTC).isoformat()
    metric = MetricValue(
        value=score["value"],
        provenance=MetricProvenance.MEASURED,
        unit="ratio",
        protocol=spec.profile,
    ).to_dict()
    task = TaskResult(
        task="automatic-speech-recognition",
        metrics={metric_name: metric},
        sample_count=score["sample_count"],
        metadata={
            "normalization": score["normalization"],
            "sample_scores": score["sample_scores"],
        },
    )
    result = RunResult(
        run_id=spec.run_id,
        status=RunStatus.COMPLETED,
        spec=spec,
        software=collect_software_provenance(),
        metrics={f"quality.{metric_name}": metric},
        task_results=(task,),
        started_at=started,
        ended_at=datetime.now(UTC).isoformat(),
    )
    store.save_result(result)
    return result
