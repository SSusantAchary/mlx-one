"""Isolated deterministic generation used by text evaluation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class GenerationError(RuntimeError):
    """Raised when an MLX generation worker fails."""


def generate_predictions(
    model_id: str,
    revision: str,
    prompts: list[str],
    *,
    max_tokens: int = 128,
    adapter_path: str | Path | None = None,
) -> list[str]:
    """Load once and generate greedy predictions in an isolated process."""
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise GenerationError("generation requires an immutable model revision")
    if not prompts or any(not isinstance(item, str) or not item for item in prompts):
        raise GenerationError("at least one non-empty prompt is required")
    if max_tokens < 1:
        raise GenerationError("max_tokens must be positive")
    adapter = Path(adapter_path).expanduser().resolve() if adapter_path else None
    if adapter is not None and not adapter.exists():
        raise GenerationError(f"adapter path does not exist: {adapter}")
    with tempfile.TemporaryDirectory(prefix="mlx-one-generate-") as temporary:
        root = Path(temporary)
        request_path = root / "request.json"
        result_path = root / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "adapter_path": str(adapter) if adapter else None,
                    "result_path": str(result_path),
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-m", "mlx_one.generation_worker", str(request_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "worker failed"
            )
            raise GenerationError(f"MLX generation failed: {detail}")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenerationError(f"invalid generation worker result: {exc}") from exc
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or any(not isinstance(item, str) for item in predictions):
        raise GenerationError("generation worker returned invalid predictions")
    return predictions
