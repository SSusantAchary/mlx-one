"""Create an unverified candidate report and proposed observation manifest."""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--updated-manifest", type=Path)
    args = parser.parse_args()
    if (args.manifest is None) != (args.updated_manifest is None):
        parser.error("--manifest and --updated-manifest must be supplied together")
    with urllib.request.urlopen("https://pypi.org/pypi/mlx/json", timeout=30) as response:
        payload = json.load(response)
    stable = []
    for version, files in payload["releases"].items():
        parts = version.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        if files and any(not item.get("yanked", False) for item in files):
            stable.append(version)
    latest = max(stable, key=lambda item: tuple(int(part) for part in item.split(".")))
    observed_at = datetime.now(UTC).isoformat()
    report = {
        "schema_version": 1,
        "observed_at": observed_at,
        "mlx_version": latest,
        "status": "unverified",
        "source": "https://pypi.org/pypi/mlx/json",
        "promoted": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest["latest_upstream"] = latest
        manifest["observed_at"] = observed_at
        args.updated_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
