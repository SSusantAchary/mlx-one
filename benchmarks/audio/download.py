"""Explicit manual checkpoint download with pinned revisions and file hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.audio.result_schema import digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(Path("benchmarks/audio/checkpoints.json").read_text())
    entry = next((item for item in catalog["entries"] if item["key"] == args.key), None)
    if entry is None:
        parser.error("unknown checkpoint key")
    if args.output.exists():
        parser.error("output must be a new directory; existing models are never overwritten")
    from huggingface_hub import snapshot_download

    snapshot_download(
        entry["checkpoint"],
        revision=entry["revision"],
        allow_patterns=entry["assets"],
        local_dir=args.output,
    )
    manifest = {
        "checkpoint": entry["checkpoint"],
        "revision": entry["revision"],
        "assets": {name: digest_file(args.output / name) for name in entry["assets"]},
    }
    (args.output / "mlx-one-download.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    if entry["key"] == "A08":
        print(
            "Original NeMo archive downloaded. "
            "Run the isolated conversion workflow in AUDIO_TESTING.md."
        )


if __name__ == "__main__":
    main()
