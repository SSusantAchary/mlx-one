"""Source inventory validation without importing any upstream runtime."""

from __future__ import annotations

import hashlib
import re
from importlib.resources import files
from pathlib import Path


def source_manifest():
    import yaml

    value = yaml.safe_load(files(__package__).joinpath("sources.yaml").read_text())
    if value.get("schema_version") != 1:
        raise ValueError("unsupported source manifest schema")
    entries = value.get("upstream_repositories", [])
    if len(entries) != 4 or len({e["name"] for e in entries}) != 4:
        raise ValueError("source inventory must contain four distinct pins")
    for entry in entries:
        if not re.fullmatch(r"[a-f0-9]{40}", entry["revision"]):
            raise ValueError("source revision must be a full Git object ID")
        for source in entry["files"]:
            if not re.fullmatch(r"[a-f0-9]{64}", source["sha256"]):
                raise ValueError("source file requires SHA-256")
    return value


def verify_captures(root: str | Path):
    """Fail on missing or modified captures; never infer verification from a pin."""
    root = Path(root).resolve()
    checked = 0
    for entry in source_manifest()["upstream_repositories"]:
        for source in entry["files"]:
            path = (root / source["local_capture"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("capture path escapes the repository")
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("source hash mismatch: " + source["path"])
            checked += 1
    return checked
