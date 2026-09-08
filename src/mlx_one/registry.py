"""Local, evidence-gated model capability registry."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path

from mlx_one.compatibility import validate_capability
from mlx_one.schemas import (
    ArtifactRef,
    CapabilitySpec,
    CapabilityStatus,
    Operation,
    RunResult,
    RunStatus,
)


class RegistryError(RuntimeError):
    """Raised when registry input is invalid or conflicts with an entry."""


class CapabilityRegistry:
    """In-memory registry keyed by exact revision, backend, operation, and constraints."""

    def __init__(self, entries: tuple[CapabilitySpec, ...] = ()) -> None:
        self._entries: dict[tuple[str, str, str, str, str], CapabilitySpec] = {}
        for entry in entries:
            self.add(entry)

    @classmethod
    def builtin(cls) -> CapabilityRegistry:
        resource = resources.files("mlx_one").joinpath("data", "capabilities.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
        return cls(tuple(CapabilitySpec.from_dict(item) for item in payload["entries"]))

    @classmethod
    def from_file(cls, path: str | Path) -> CapabilityRegistry:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise RegistryError("registry requires schema_version '1.0'")
        return cls(tuple(CapabilitySpec.from_dict(item) for item in payload["entries"]))

    def add(self, entry: CapabilitySpec) -> None:
        report = validate_capability(entry)
        if not report.valid:
            raise RegistryError("; ".join(issue.message for issue in report.issues))
        key = self._key(entry)
        if key in self._entries:
            raise RegistryError("duplicate capability entry")
        self._entries[key] = entry

    def upsert(self, entry: CapabilitySpec) -> None:
        """Insert an entry or replace the same scoped claim after validation."""
        report = validate_capability(entry)
        if not report.valid:
            raise RegistryError("; ".join(issue.message for issue in report.issues))
        self._entries[self._key(entry)] = entry

    def write(self, path: str | Path) -> Path:
        """Atomically persist a registry."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(target)
        return target

    def promote_from_result(
        self,
        entry: CapabilitySpec,
        result: RunResult,
        result_path: str | Path,
        status: CapabilityStatus | str,
    ) -> CapabilitySpec:
        """Promote a claim only from matching, successful, checksummed run evidence."""
        promoted_status = CapabilityStatus(status)
        if promoted_status not in {
            CapabilityStatus.INTEGRATION_TESTED,
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        }:
            raise RegistryError("promotion requires a verified capability status")
        if result.status is not RunStatus.COMPLETED:
            raise RegistryError("only completed runs can promote a capability")
        if result.spec.model != entry.model or result.spec.operation is not entry.operation:
            raise RegistryError("run model and operation must match the capability")
        if promoted_status in {
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        } and result.spec.hardware is None:
            raise RegistryError("hardware and quality promotion require run hardware provenance")
        source = Path(result_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        packages = result.software.packages if result.software is not None else {}
        constraints = {
            **entry.constraints,
            "evidence_run_id": result.run_id,
            "backend_version": packages.get(entry.backend, "not-recorded"),
        }
        promoted = CapabilitySpec(
            model=entry.model,
            backend=entry.backend,
            operation=entry.operation,
            status=promoted_status,
            constraints=constraints,
            hardware_profile_ids=(result.spec.hardware.profile_id,)
            if result.spec.hardware is not None
            else (),
            evidence=(
                *entry.evidence,
                ArtifactRef(
                    uri=f"run://{result.run_id}/result.json",
                    kind="run-result",
                    sha256=digest,
                    media_type="application/json",
                    metadata={"run_id": result.run_id},
                ),
            ),
            notes=entry.notes,
        )
        self._entries.pop(self._key(entry), None)
        self.upsert(promoted)
        return promoted

    def requiring_revalidation(
        self, backend_versions: dict[str, str]
    ) -> tuple[CapabilitySpec, ...]:
        """Return verified entries whose recorded backend version no longer matches."""
        verified = {
            CapabilityStatus.INTEGRATION_TESTED,
            CapabilityStatus.HARDWARE_VERIFIED,
            CapabilityStatus.QUALITY_VERIFIED,
        }
        return tuple(
            entry
            for entry in self.entries
            if entry.status in verified
            and entry.backend in backend_versions
            and entry.constraints.get("backend_version") not in {
                None,
                backend_versions[entry.backend],
            }
        )

    def find(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        backend: str | None = None,
        operation: Operation | str | None = None,
    ) -> tuple[CapabilitySpec, ...]:
        operation_value = Operation(operation).value if operation is not None else None
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.model.model_id == model_id
            and (revision is None or entry.model.revision == revision)
            and (backend is None or entry.backend == backend)
            and (operation_value is None or entry.operation.value == operation_value)
        )

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "entries": [item.to_dict() for item in self.entries]}

    @property
    def entries(self) -> tuple[CapabilitySpec, ...]:
        return tuple(self._entries.values())

    @staticmethod
    def _key(entry: CapabilitySpec) -> tuple[str, str, str, str, str]:
        constraints = json.dumps(entry.constraints, sort_keys=True, separators=(",", ":"))
        return (
            entry.model.model_id,
            entry.model.revision,
            entry.backend,
            entry.operation.value,
            constraints,
        )
