"""Explicit, loss-aware migrations for persisted lifecycle records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CURRENT_SCHEMA_VERSION = "1.0"
LEGACY_SCHEMA_VERSIONS = {None, "0.1"}

_ALIASES: dict[str, dict[str, str]] = {
    "artifact": {"location": "uri", "type": "kind", "hash": "sha256"},
    "capability": {
        "model_spec": "model",
        "hardware_profiles": "hardware_profile_ids",
    },
    "checkpoint": {
        "base_model": "base_model_id",
        "base_model_revision": "base_revision",
        "states": "resumable_state",
    },
    "dataset": {"id": "dataset_id", "path": "source", "hash": "sha256"},
    "run": {"id": "run_id", "parents": "parent_run_ids"},
    "run_result": {"id": "run_id"},
}

_FIELDS: dict[str, set[str]] = {
    "artifact": {"schema_version", "uri", "kind", "sha256", "media_type", "metadata"},
    "capability": {
        "schema_version",
        "model",
        "backend",
        "operation",
        "status",
        "constraints",
        "hardware_profile_ids",
        "evidence",
        "notes",
    },
    "checkpoint": {
        "schema_version",
        "run_id",
        "base_model_id",
        "base_revision",
        "method",
        "step",
        "adapter",
        "config_sha256",
        "resumable_state",
        "resume_semantics",
        "state_artifacts",
        "created_at",
    },
    "dataset": {
        "schema_version",
        "dataset_id",
        "source",
        "revision",
        "split",
        "format",
        "layout",
        "columns",
        "response_only",
        "private",
        "sha256",
        "metadata",
    },
    "run": {
        "schema_version",
        "run_id",
        "operation",
        "model",
        "profile",
        "hardware",
        "dataset_revisions",
        "generation",
        "seed",
        "limits",
        "cache_policy",
        "parent_run_ids",
        "metadata",
        "created_at",
    },
    "run_result": {
        "schema_version",
        "run_id",
        "status",
        "spec",
        "software",
        "metrics",
        "task_results",
        "timings",
        "failures",
        "artifacts",
        "started_at",
        "ended_at",
    },
}


def migrate_record(record_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate a supported persisted record while retaining unknown legacy fields."""
    payload = dict(data)
    version = payload.get("schema_version")
    if version == CURRENT_SCHEMA_VERSION:
        return payload
    if version not in LEGACY_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION!r}"
        )
    if record_type not in _FIELDS:
        raise ValueError(f"unsupported migration record type: {record_type}")
    for old, new in _ALIASES.get(record_type, {}).items():
        if old in payload and new not in payload:
            payload[new] = payload.pop(old)
    payload["schema_version"] = CURRENT_SCHEMA_VERSION
    unknown = {key: payload.pop(key) for key in tuple(payload) if key not in _FIELDS[record_type]}
    if unknown and "metadata" in _FIELDS[record_type]:
        metadata = dict(payload.get("metadata") or {})
        metadata["legacy_fields"] = unknown
        payload["metadata"] = metadata
    return payload
