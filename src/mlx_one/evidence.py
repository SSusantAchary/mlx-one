"""Sanitize and publish checksummed lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mlx_one.compatibility import validate_run_evidence
from mlx_one.schemas import RunResult


class EvidenceError(RuntimeError):
    """Raised when run evidence is invalid or unsafe to publish."""


def publish_run_evidence(source: str | Path, destination: str | Path) -> tuple[Path, str]:
    """Write a privacy-safe run record and return its SHA-256 digest."""
    try:
        result = RunResult.from_json(Path(source).read_text(encoding="utf-8"))
        sanitized = RunResult.from_dict(_sanitize(result.to_dict(), result.run_id))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid run evidence: {exc}") from exc
    report = validate_run_evidence(sanitized)
    if not report.valid:
        raise EvidenceError("; ".join(issue.message for issue in report.issues))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(sanitized.to_json() + "\n", encoding="utf-8")
    temporary.replace(target)
    return target, hashlib.sha256(target.read_bytes()).hexdigest()


def _sanitize(value: Any, run_id: str, key: str = "") -> Any:
    if isinstance(value, dict):
        result = {child: _sanitize(item, run_id, child) for child, item in value.items()}
        if key == "artifacts" or {"uri", "kind"} <= set(result):
            kind = str(result.get("kind", "artifact"))
            result["uri"] = f"artifact://{run_id}/{kind}"
        return result
    if isinstance(value, list):
        return [_sanitize(item, run_id, key) for item in value]
    if isinstance(value, str):
        if key in {"prompt", "prediction", "reference"}:
            return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
        if key in {"output_dir", "source"} and Path(value).expanduser().is_absolute():
            return "redacted-local-path"
    return value
