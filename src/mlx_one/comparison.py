"""Protocol-safe run comparison and quality gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlx_one.schemas import RunResult


class ComparisonError(RuntimeError):
    """Raised when result protocols are incompatible or gates are malformed."""


@dataclass(frozen=True)
class ComparisonReport:
    compatible: bool
    passed: bool
    baseline_run_id: str
    candidate_run_id: str
    metrics: dict[str, dict[str, float | None]]
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "compatible": self.compatible,
            "passed": self.passed,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "metrics": self.metrics,
            "failures": list(self.failures),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        rows = ["| Metric | Baseline | Candidate | Delta |", "| --- | ---: | ---: | ---: |"]
        for name, values in self.metrics.items():
            baseline = _fmt(values["baseline"])
            candidate = _fmt(values["candidate"])
            delta = _fmt(values["delta"])
            rows.append(f"| {name} | {baseline} | {candidate} | {delta} |")
        status = "PASS" if self.passed else "FAIL"
        return f"# Run comparison: {status}\n\n" + "\n".join(rows) + "\n"


def compare_results(
    baseline: RunResult,
    candidate: RunResult,
    *,
    gates: dict[str, Any] | None = None,
    allow_incompatible: bool = False,
) -> ComparisonReport:
    """Compare common numeric metrics after enforcing protocol identity."""
    compatible = _protocol(baseline) == _protocol(candidate)
    if not compatible and not allow_incompatible:
        raise ComparisonError("evaluation protocols differ; use an explicit override to compare")
    metrics: dict[str, dict[str, float | None]] = {}
    for name in sorted(set(baseline.metrics) & set(candidate.metrics)):
        old = _numeric_value(baseline.metrics[name])
        new = _numeric_value(candidate.metrics[name])
        if old is not None and new is not None:
            metrics[name] = {"baseline": old, "candidate": new, "delta": new - old}
    failures: list[str] = []
    rules = (gates or {}).get("metrics", {})
    if not isinstance(rules, dict):
        raise ComparisonError("gate metrics must be an object")
    for name, rule in rules.items():
        if name not in metrics:
            failures.append(f"required metric {name!r} is unavailable")
            continue
        if not isinstance(rule, dict):
            raise ComparisonError(f"gate for {name!r} must be an object")
        candidate_value = metrics[name]["candidate"]
        delta = metrics[name]["delta"]
        if "minimum" in rule and candidate_value < float(rule["minimum"]):  # type: ignore[operator]
            failures.append(f"{name} is below minimum {rule['minimum']}")
        if "maximum" in rule and candidate_value > float(rule["maximum"]):  # type: ignore[operator]
            failures.append(f"{name} exceeds maximum {rule['maximum']}")
        if "max_regression" in rule and delta < -float(rule["max_regression"]):  # type: ignore[operator]
            failures.append(f"{name} regressed by more than {rule['max_regression']}")
    return ComparisonReport(
        compatible,
        compatible and not failures,
        baseline.run_id,
        candidate.run_id,
        metrics,
        tuple(failures),
    )


def load_result(path: str | Path) -> RunResult:
    try:
        return RunResult.from_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"invalid result file {path}: {exc}") from exc


def load_gates(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ComparisonError(f"invalid gate file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError("gate file must contain an object")
    return payload


def _protocol(result: RunResult) -> str:
    generation = {
        key: value for key, value in result.spec.generation.items() if key != "adapter"
    }
    return json.dumps(
        {
            "profile": result.spec.profile,
            "datasets": result.spec.dataset_revisions,
            "generation": generation,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict) and value.get("provenance") != "not-available":
        nested = value.get("value")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            return float(nested)
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"
