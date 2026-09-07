"""Local, evidence-gated model capability registry."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from mlx_one.compatibility import validate_capability
from mlx_one.schemas import CapabilitySpec, Operation


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
        return cls(tuple(CapabilitySpec.from_dict(item) for item in payload["entries"]))

    def add(self, entry: CapabilitySpec) -> None:
        report = validate_capability(entry)
        if not report.valid:
            raise RegistryError("; ".join(issue.message for issue in report.issues))
        key = self._key(entry)
        if key in self._entries:
            raise RegistryError("duplicate capability entry")
        self._entries[key] = entry

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
