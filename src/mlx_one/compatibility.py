"""Validation for evidence-backed compatibility and calibration records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_one.schemas import (
    CalibrationRecord,
    CalibrationStatus,
    CapabilitySpec,
    CapabilityStatus,
    CompatibilityResult,
    CompatibilityStatus,
)

_IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_PRIVATE_KEYS = {
    "hardware_uuid",
    "private_path",
    "raw_audio",
    "raw_prompt",
    "serial_number",
    "username",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable evidence validation finding."""

    code: str
    message: str


@dataclass(frozen=True)
class EvidenceValidationReport:
    """Result of validating one evidence record for publication."""

    valid: bool
    issues: tuple[ValidationIssue, ...] = ()

    def require_valid(self) -> None:
        """Raise one combined error when publication is unsafe."""
        if self.issues:
            raise ValueError("; ".join(issue.message for issue in self.issues))


def validate_calibration_evidence(record: CalibrationRecord) -> EvidenceValidationReport:
    """Validate whether a calibration record may support a verified claim."""
    issues: list[ValidationIssue] = []
    if record.status is not CalibrationStatus.COMPLETED:
        issues.append(
            ValidationIssue(
                "non-success-status",
                f"calibration status {record.status.value!r} is not verified success",
            )
        )
    _validate_identity(record.model.revision, record.hardware.profile_id, issues)
    if not record.software.packages:
        issues.append(ValidationIssue("missing-software", "software package versions are required"))
    if record.started_at is None or record.ended_at is None:
        issues.append(
            ValidationIssue("missing-timestamps", "start and end timestamps are required")
        )
    if record.status is CalibrationStatus.COMPLETED and not record.metrics:
        issues.append(
            ValidationIssue("missing-metrics", "completed evidence requires measured metrics")
        )
    _scan_private(record.to_dict(), issues)
    return EvidenceValidationReport(not issues, tuple(issues))


def validate_compatibility_evidence(result: CompatibilityResult) -> EvidenceValidationReport:
    """Validate a compatibility claim before registry ingestion."""
    issues: list[ValidationIssue] = []
    verified = result.status in {
        CompatibilityStatus.OFFICIALLY_VERIFIED,
        CompatibilityStatus.COMMUNITY_VERIFIED,
    }
    if verified:
        _validate_identity(result.model.revision, result.hardware.profile_id, issues)
        if result.workload is None:
            issues.append(ValidationIssue("missing-workload", "verified claims require a workload"))
        if not result.evidence:
            issues.append(ValidationIssue("missing-evidence", "verified claims require evidence"))
        for artifact in result.evidence:
            if artifact.sha256 is None:
                issues.append(
                    ValidationIssue(
                        "missing-checksum", f"evidence artifact {artifact.uri!r} requires sha256"
                    )
                )
    _scan_private(result.to_dict(), issues)
    return EvidenceValidationReport(not issues, tuple(issues))


def validate_capability(spec: CapabilitySpec) -> EvidenceValidationReport:
    """Validate promotion requirements for an operation-level capability."""
    issues: list[ValidationIssue] = []
    verified = spec.status in {
        CapabilityStatus.INTEGRATION_TESTED,
        CapabilityStatus.HARDWARE_VERIFIED,
        CapabilityStatus.QUALITY_VERIFIED,
    }
    if verified and not _IMMUTABLE_REVISION.fullmatch(spec.model.revision):
        issues.append(
            ValidationIssue("mutable-revision", "verified capabilities require an exact revision")
        )
    if verified and not spec.evidence:
        issues.append(ValidationIssue("missing-evidence", "verified capabilities require evidence"))
    if spec.status in {CapabilityStatus.HARDWARE_VERIFIED, CapabilityStatus.QUALITY_VERIFIED}:
        if not spec.hardware_profile_ids:
            issues.append(
                ValidationIssue(
                    "missing-hardware", "hardware or quality verification requires a profile"
                )
            )
    if spec.status is CapabilityStatus.QUALITY_VERIFIED and spec.operation.value != "evaluate":
        issues.append(
            ValidationIssue(
                "invalid-quality-operation", "quality verification is scoped to evaluation"
            )
        )
    for artifact in spec.evidence:
        if artifact.sha256 is None:
            issues.append(ValidationIssue("missing-checksum", "registry evidence requires sha256"))
    _scan_private(spec.to_dict(), issues)
    return EvidenceValidationReport(not issues, tuple(issues))


def _validate_identity(revision: str, profile_id: str, issues: list[ValidationIssue]) -> None:
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        issues.append(
            ValidationIssue("mutable-revision", "verified evidence requires an exact revision")
        )
    if not profile_id.strip():
        issues.append(ValidationIssue("missing-hardware", "a hardware profile is required"))


def _scan_private(value: Any, issues: list[ValidationIssue], key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            normalized = str(child_key).lower().replace("-", "_")
            if normalized in _PRIVATE_KEYS and child not in (None, "", [], {}):
                issues.append(
                    ValidationIssue("private-field", f"private field {child_key!r} must be removed")
                )
            _scan_private(child, issues, normalized)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _scan_private(child, issues, key)
    elif isinstance(value, str) and key.endswith(("path", "uri")):
        path = Path(value).expanduser()
        if path.is_absolute() and not value.startswith(("artifact://", "hf://")):
            issues.append(ValidationIssue("private-path", f"absolute private path in {key!r}"))
