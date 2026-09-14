"""Offline Apple MLX runtime compatibility resolution."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import metadata, resources
from typing import Any


class MLXCompatibilityStatus(str, Enum):
    """Qualification state for an installed Apple MLX release."""

    VERIFIED = "verified"
    SUPPORTED = "supported"
    UNVERIFIED = "unverified"
    DEGRADED = "degraded"
    INCOMPATIBLE = "incompatible"
    LEGACY = "legacy"


@dataclass(frozen=True)
class MLXCompatibility:
    """Resolved compatibility information for the local MLX runtime."""

    runtime_version: str | None
    latest_upstream: str
    latest_verified: str | None
    supported_versions: tuple[str, ...]
    state: MLXCompatibilityStatus
    capabilities: dict[str, bool | None]
    reason: str
    observed_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        value["supported_versions"] = list(self.supported_versions)
        return value


class MLXCompatibilityManifestError(RuntimeError):
    """Raised when the packaged compatibility manifest is invalid."""


def load_mlx_compatibility_manifest() -> dict[str, Any]:
    """Load and validate the package-owned compatibility manifest."""
    resource = resources.files("mlx_one").joinpath("data", "compatibility", "mlx.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MLXCompatibilityManifestError(
            f"cannot read MLX compatibility manifest: {exc}"
        ) from exc
    _validate_manifest(payload)
    return payload


def get_mlx_compatibility(*, runtime_version: str | None = None) -> MLXCompatibility:
    """Resolve compatibility without importing MLX or accessing the network."""
    manifest = load_mlx_compatibility_manifest()
    versions = manifest["mlx_versions"]
    supported = tuple(sorted(versions, key=_version_key, reverse=True))
    detect_runtime = runtime_version is None
    if detect_runtime:
        try:
            runtime_version = metadata.version("mlx")
        except metadata.PackageNotFoundError:
            runtime_version = None
    latest = manifest["latest_upstream"]
    latest_verified = manifest["latest_verified"]
    observed_at = manifest["observed_at"]
    if runtime_version is None:
        return MLXCompatibility(
            None,
            latest,
            latest_verified,
            supported,
            MLXCompatibilityStatus.INCOMPATIBLE,
            {},
            "Apple MLX is not installed.",
            observed_at,
        )
    if detect_runtime and not _mlx_importable():
        return MLXCompatibility(
            runtime_version,
            latest,
            latest_verified,
            supported,
            MLXCompatibilityStatus.INCOMPATIBLE,
            {},
            f"Apple MLX {runtime_version} is installed but cannot be imported.",
            observed_at,
        )
    entry = versions.get(runtime_version)
    if entry is not None:
        state = MLXCompatibilityStatus(entry["status"])
        capabilities = dict(entry.get("capabilities", {}))
        return MLXCompatibility(
            runtime_version,
            latest,
            latest_verified,
            supported,
            state,
            capabilities,
            _entry_reason(runtime_version, state),
            observed_at,
        )
    newest_recorded = supported[0]
    oldest_recorded = supported[-1]
    if (
        not _STABLE_VERSION.fullmatch(runtime_version)
        or _version_key(runtime_version) > _version_key(newest_recorded)
        or _version_key(runtime_version) >= _version_key(oldest_recorded)
    ):
        state = MLXCompatibilityStatus.UNVERIFIED
        reason = f"MLX {runtime_version} is newer than or outside the qualified release list."
    else:
        state = MLXCompatibilityStatus.LEGACY
        reason = f"MLX {runtime_version} is outside the rolling five-release window."
    return MLXCompatibility(
        runtime_version,
        latest,
        latest_verified,
        supported,
        state,
        {},
        reason,
        observed_at,
    )


_STABLE_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){1,2}")


def _version_key(value: str) -> tuple[int, ...]:
    match = _STABLE_VERSION.fullmatch(value)
    return tuple(int(part) for part in value.split(".")) if match else (10**9,)


def _mlx_importable() -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import mlx.core"],
            capture_output=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _validate_manifest(payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise MLXCompatibilityManifestError("MLX manifest requires schema_version 1")
    policy = payload.get("policy")
    if not isinstance(policy, dict) or policy.get("release_window") != 5:
        raise MLXCompatibilityManifestError("MLX manifest release_window must be 5")
    for name in ("observed_at", "latest_upstream"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise MLXCompatibilityManifestError(f"MLX manifest requires {name}")
    latest_verified = payload.get("latest_verified")
    if latest_verified is not None and not isinstance(latest_verified, str):
        raise MLXCompatibilityManifestError("latest_verified must be a version or null")
    versions = payload.get("mlx_versions")
    if not isinstance(versions, dict) or len(versions) != 5:
        raise MLXCompatibilityManifestError("MLX manifest must contain exactly five releases")
    allowed = {status.value for status in MLXCompatibilityStatus} - {"legacy", "unverified"}
    for version, entry in versions.items():
        if not isinstance(version, str) or not _STABLE_VERSION.fullmatch(version):
            raise MLXCompatibilityManifestError(f"invalid MLX version key: {version!r}")
        if not isinstance(entry, dict) or entry.get("status") not in allowed:
            raise MLXCompatibilityManifestError(f"invalid status for MLX {version}")
        capabilities = entry.get("capabilities", {})
        if not isinstance(capabilities, dict) or any(
            not isinstance(key, str) or value is not None and not isinstance(value, bool)
            for key, value in capabilities.items()
        ):
            raise MLXCompatibilityManifestError(f"invalid capabilities for MLX {version}")
        for name in ("qualified_at", "repository_commit", "environment", "performance", "report"):
            if name not in entry:
                raise MLXCompatibilityManifestError(
                    f"MLX {version} entry requires nullable {name}"
                )
        if entry["qualified_at"] is not None and not isinstance(entry["qualified_at"], str):
            raise MLXCompatibilityManifestError(f"invalid qualified_at for MLX {version}")
        if entry["repository_commit"] is not None and not isinstance(
            entry["repository_commit"], str
        ):
            raise MLXCompatibilityManifestError(
                f"invalid repository_commit for MLX {version}"
            )
        if entry["environment"] is not None and not isinstance(entry["environment"], dict):
            raise MLXCompatibilityManifestError(f"invalid environment for MLX {version}")
        if entry["performance"] is not None and not isinstance(entry["performance"], dict):
            raise MLXCompatibilityManifestError(f"invalid performance for MLX {version}")
        if entry["report"] is not None and not isinstance(entry["report"], str):
            raise MLXCompatibilityManifestError(f"invalid report for MLX {version}")
    ordered = sorted(versions, key=_version_key, reverse=True)
    if not _STABLE_VERSION.fullmatch(payload["latest_upstream"]):
        raise MLXCompatibilityManifestError("latest_upstream must be a stable MLX version")
    if _version_key(payload["latest_upstream"]) < _version_key(ordered[0]):
        raise MLXCompatibilityManifestError(
            "latest_upstream cannot be older than the newest manifest release"
        )
    if latest_verified is not None:
        entry = versions.get(latest_verified)
        if entry is None or entry.get("status") != "verified":
            raise MLXCompatibilityManifestError("latest_verified must reference a verified entry")


def _entry_reason(version: str, state: MLXCompatibilityStatus) -> str:
    if state is MLXCompatibilityStatus.VERIFIED:
        return f"MLX {version} passed the complete mlx-one qualification contract."
    if state is MLXCompatibilityStatus.SUPPORTED:
        return f"MLX {version} is in the support window but is not fully qualified."
    if state is MLXCompatibilityStatus.DEGRADED:
        return f"MLX {version} has documented accepted regressions."
    return f"MLX {version} has documented incompatibilities."
