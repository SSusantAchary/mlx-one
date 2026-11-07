"""Immutable evidence helpers for local native-task experiments."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

_SHA = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, metadata: dict, files: dict[str, Path]) -> None:
    for name in ("source_commit", "model_revision"):
        if not isinstance(metadata.get(name), str) or not _SHA.fullmatch(metadata[name]):
            raise ValueError(f"{name} must be a full immutable Git SHA")
    if not files:
        raise ValueError("at least one evidence file is required")
    report = {
        **metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            name: {"path": str(file), "sha256": sha256_file(file)} for name, file in files.items()
        },
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
