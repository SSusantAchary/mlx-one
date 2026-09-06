"""JSON checkpoint persistence for long-running MLX One operations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_DIR = Path.home() / ".mlx-one" / "checkpoints"


class CheckpointManager:
    """Manage a single JSON checkpoint file."""

    def __init__(
        self,
        name: str = "checkpoint.json",
        *,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        if not name:
            raise ValueError("checkpoint name cannot be empty")
        self.checkpoint_dir = (
            Path(checkpoint_dir).expanduser() if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
        )
        self.path = self.checkpoint_dir / name

    def save(
        self,
        state: dict[str, Any],
        *,
        model_name: str | None = None,
        operation_type: str | None = None,
        progress_percentage: float | None = None,
    ) -> Path:
        """Save JSON-serializable state and metadata."""
        metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "operation_type": operation_type,
            "progress_percentage": progress_percentage,
        }
        payload = {"metadata": metadata, "state": state}

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
        except TypeError as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise TypeError("checkpoint state must be JSON serializable") from exc

        tmp_path.replace(self.path)
        return self.path

    def load(self) -> dict[str, Any] | None:
        """Load checkpoint payload, or None when no checkpoint exists."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def clear(self) -> None:
        """Delete the checkpoint file when present."""
        if self.path.exists():
            self.path.unlink()
