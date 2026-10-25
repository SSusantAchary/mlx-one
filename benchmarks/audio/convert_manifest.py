"""Register a verified existing Parakeet v2 conversion for local benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.audio.result_schema import digest_file
from mlx_one.models.audio.parakeet.loading import validate_conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = args.model.expanduser().resolve()
    manifest = root / "mlx-one-download.json"
    if manifest.exists():
        parser.error("download manifest exists; refusing overwrite")
    conversion = validate_conversion(root)
    assets = dict(conversion["assets"])
    assets["conversion-provenance.json"] = digest_file(root / "conversion-provenance.json")
    with manifest.open("x") as handle:
        json.dump(
            {
                "checkpoint": conversion["model"],
                "revision": conversion["revision"],
                "assets": assets,
                "conversion_archive_sha256": conversion["archive_sha256"],
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(manifest)


if __name__ == "__main__":
    main()
