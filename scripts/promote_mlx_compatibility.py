"""Validate and explicitly promote one completed native MLX qualification report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid stable MLX version: {version}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    required = {
        "mlx_version",
        "qualified_at",
        "repository_commit",
        "environment",
        "capabilities",
        "performance",
    }
    missing = sorted(required - set(report))
    if missing:
        raise SystemExit(f"qualification report is incomplete: {', '.join(missing)}")
    version = report["mlx_version"]
    _key(version)
    required_capabilities = {
        "qwen2.5-0.5b-native-training",
        "minicpm5-1b",
        "minicpm5-2b",
        "qwen2-vl-2b",
        "whisper-tiny",
        "retrieval",
    }
    if set(report["capabilities"]) != required_capabilities or not all(
        report["capabilities"].values()
    ):
        raise SystemExit("all required native capabilities must pass before promotion")
    if report.get("failure") is not None or report.get("promotion_eligible") is not True:
        raise SystemExit("qualification report is not marked promotion-eligible")
    if report["performance"].get("verdict") != "acceptable":
        raise SystemExit("performance verdict must be acceptable before promotion")
    manifest_path = root / "src/mlx_one/data/compatibility/mlx.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = root / "compatibility/history"
    history.mkdir(parents=True, exist_ok=True)
    archived = history / f"mlx-{version}.json"
    if archived.exists():
        raise SystemExit(f"immutable report already exists: {archived}")
    shutil.copy2(args.report, archived)
    entry = {
        "status": "verified",
        "qualified_at": report["qualified_at"],
        "repository_commit": report["repository_commit"],
        "environment": report["environment"],
        "capabilities": report["capabilities"],
        "performance": report["performance"],
        "report": str(archived.relative_to(root)),
    }
    versions = {**manifest["mlx_versions"], version: entry}
    ordered = sorted(versions, key=_key, reverse=True)[:5]
    manifest["mlx_versions"] = {item: versions[item] for item in ordered}
    manifest["latest_upstream"] = max(
        manifest["latest_upstream"], version, key=_key
    )
    manifest["latest_verified"] = version
    manifest["observed_at"] = report["qualified_at"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
