"""Local run metadata shared by the MLXBoost development tools."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def snapshot(output, *, model=None, status="RUNNING", flags=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()

    paths = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(paths))
        if name and (ROOT / name).is_file()
    }
    sources = ROOT / "local/mlxboost/sources/sources.json"
    value = {
        "schema_version": "mlxboost-run-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": status,
        "git_status": git("status", "--porcelain=v1"),
        "working_tree_sha256": hashes,
        "command": [sys.executable, *sys.argv],
        "model": model,
        "flags": flags or {},
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment_lock": {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        "environment_overrides": {
            k: v for k, v in os.environ.items() if k.startswith("MLXBOOST_") or k == "PYTHONPATH"
        },
        "source_provenance": json.loads(sources.read_text()) if sources.exists() else None,
        "temperature": None,
        "temperature_method": "no sensor sampled",
        "power_state": None,
        "power_state_method": "not sampled",
        "gpu_profile": "not captured; wall-clock synchronization is not GPU kernel timing",
    }
    (Path(output) / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return value
