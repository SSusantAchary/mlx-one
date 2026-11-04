#!/usr/bin/env python3
"""Snapshot current task sources and checksum a new milestone evidence directory."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root, run = Path.cwd(), args.run.resolve()
    if (run / "manifest.json").exists():
        raise FileExistsError("evidence is already sealed")
    baseline = json.loads(Path("local/native_tasks/runs/m1-20261103/baseline.json").read_text())
    protected = json.loads(
        Path("local/native_tasks/runs/m1-20261103/frozen-check.json").read_text()
    )["checked_files"]
    changed = [name for name in protected if sha(root / name) != baseline["sha256"][name]]
    if changed:
        raise ValueError(f"protected files changed: {changed}")
    (run / "frozen-check.json").write_text(
        json.dumps({"checked_files": protected, "changed": changed}, indent=2) + "\n"
    )
    paths = set()
    for prefix in (
        "src/mlx_one/classification",
        "src/mlx_one/models/classification",
        "src/mlx_one/models/audio/silero",
        "src/mlx_one/models/detection",
        "src/mlx_one/models/depth",
        "src/mlx_one/models/vision_language/qwen2_5_vl",
        "src/mlx_one/vision",
        "benchmarks/native_tasks",
        "tests/native_tasks",
        "docs/native_tasks",
        "examples/native_tasks",
    ):
        paths.update(
            p for p in (root / prefix).rglob("*") if p.is_file() and "__pycache__" not in p.parts
        )
    for name in (
        "README.md",
        "pyproject.toml",
        "src/mlx_one/__init__.py",
        "src/mlx_one/cli.py",
        "src/mlx_one/data/native_tasks.json",
        "src/mlx_one/vision/clip_processing.py",
        "src/mlx_one/vision/images.py",
        "src/mlx_one/vision/detr_processing.py",
        "src/mlx_one/audio/vad.py",
        "src/mlx_one/audio/vad_postprocessing.py",
    ):
        paths.add(root / name)
    for prefix in ("src/mlx_one/engine", "src/mlx_one/server"):
        paths.update((root / prefix).rglob("*.py"))
    for path in sorted(paths):
        target = run / "source-snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name, command in (
        ("native-pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
        (
            "reference-pip-freeze.txt",
            ["local/native_tasks/reference_venv/bin/python", "-m", "pip", "freeze"],
        ),
        ("git-status.txt", ["git", "status", "--porcelain=v1"]),
    ):
        with (run / name).open("x") as stream:
            subprocess.run(command, stdout=stream, check=True)
    files = {
        str(p.relative_to(run)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    profiles = json.loads(Path("src/mlx_one/data/native_tasks.json").read_text())["profiles"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc",
        "checkpoint_profiles": profiles,
        "hardware": "Apple M4 32 GiB; macOS15.8.1; Metal3; MLX0.32.2",
        "python": platform.python_version(),
        "files": files,
    }
    with (run / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({"sealed_files": len(files), "protected_unchanged": len(protected)}))


if __name__ == "__main__":
    main()
