"""Atomic local-first persistence for lifecycle runs."""

from __future__ import annotations

import json
from pathlib import Path

from mlx_one.schemas import RunResult, RunSpec


class RunStoreError(RuntimeError):
    """Raised when a run is missing, corrupt, or would be overwritten."""


class RunStore:
    """Store each run in an isolated directory using atomic JSON replacement."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def create(self, spec: RunSpec) -> Path:
        path = self.path_for(spec.run_id)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunStoreError(f"run already exists: {spec.run_id}") from exc
        self._write(path / "manifest.json", spec.to_dict())
        return path

    def save_result(self, result: RunResult) -> Path:
        path = self.path_for(result.run_id)
        if not path.is_dir():
            self.create(result.spec)
        else:
            stored = self.load_spec(result.run_id)
            if stored != result.spec:
                raise RunStoreError("result spec does not match the stored run manifest")
        target = path / "result.json"
        self._write(target, result.to_dict())
        return target

    def load_spec(self, run_id: str) -> RunSpec:
        return RunSpec.from_dict(self._read(self.path_for(run_id) / "manifest.json"))

    def load_result(self, run_id: str) -> RunResult:
        return RunResult.from_dict(self._read(self.path_for(run_id) / "result.json"))

    def path_for(self, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise RunStoreError("run_id must be a safe single path component")
        return self.root / run_id

    @staticmethod
    def _read(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunStoreError(f"cannot read run record {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RunStoreError(f"run record must be a JSON object: {path}")
        return payload

    @staticmethod
    def _write(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
