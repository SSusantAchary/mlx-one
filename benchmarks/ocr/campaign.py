"""Download, measure, then remove only this campaign's explicitly owned temporary assets."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.ocr.corpus import EVIDENCE, ROOT, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cohort", choices=["omni100", "hand10"], default="omni100")
    parser.add_argument("--ids", nargs="+", default=["O04", "O03", "O05", "O06", "O07"])
    args = parser.parse_args()
    hand = args.cohort == "hand10"
    evidence = EVIDENCE.parent / "hand10-20261025" if hand else EVIDENCE
    work = args.work.resolve()
    marker = work / ".mlx-one-ocr-benchmark-owned"
    if not marker.is_file() or marker.read_text().strip() != args.cohort + "-20261025":
        parser.error("scratch directory needs an explicit campaign ownership marker")
    if args.work.is_symlink() or work == ROOT or ROOT.is_relative_to(work):
        parser.error("unsafe scratch directory")
    # Set before importing Hub/Xet, which cache their environment configuration.
    os.environ["HF_XET_CACHE"] = str(work / "xet-cache")
    os.environ["HF_HUB_CACHE"] = str(work / "hub-cache")
    from huggingface_hub import snapshot_download

    inventory = json.loads((EVIDENCE.parent / "inventory.json").read_text())["checkpoints"]
    cleanup = {
        "schema_version": "mlx-one-ocr-cleanup-v1",
        "owned_directory": str(work),
        "policy": "only task-owned model/data/cache artifacts; pre-existing caches untouched",
        "models": [],
    }
    for entry in [] if hand else inventory[:2]:
        write(
            EVIDENCE / (entry["id"] + ".json"),
            {
                "schema_version": "mlx-one-ocr-omni100-v1",
                "id": entry["id"],
                "checkpoint": entry["checkpoint"],
                "revision": entry["revision"],
                "name": entry["name"],
                "precision_label": entry["precision"],
                "status": "INCOMPATIBLE",
                "reason": "Cropped-line recognizer; dataset has page images "
                "and no annotated line crops. No page benchmark or weight download performed.",
            },
        )
    for identifier in args.ids:
        entry = next(e for e in inventory if e["id"] == identifier)
        path = work / "models" / identifier
        output = evidence / (identifier + ".json")
        try:
            print("DOWNLOAD", identifier, entry["checkpoint"], flush=True)
            snapshot_download(
                entry["checkpoint"],
                revision=entry["revision"],
                local_dir=path,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                    "*.jinja",
                    "LICENSE*",
                    "README.md",
                ],
                max_workers=4,
            )
            assets = [
                {"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(path.rglob("*"))
                if p.is_file() and ".cache" not in p.parts
            ]
            write(EVIDENCE / (identifier + "-assets.json"), assets)
            print("RUN", identifier, flush=True)
            with (evidence / (identifier + ".log")).open("a") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.ocr.corpus",
                        "run",
                        "--id",
                        identifier,
                        "--work",
                        str(work),
                        "--cohort",
                        args.cohort,
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print("FINISHED", identifier, result.returncode, flush=True)
        except Exception as exc:
            write(
                output,
                {
                    "schema_version": (
                        "mlx-one-ocr-hand10-v1" if hand else "mlx-one-ocr-omni100-v1"
                    ),
                    "id": identifier,
                    "checkpoint": entry["checkpoint"],
                    "revision": entry["revision"],
                    "name": entry["name"],
                    "precision_label": entry["precision"],
                    "status": "FAILED",
                    "stage": "download/assets",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print("FAILED", identifier, type(exc).__name__, str(exc), flush=True)
        finally:
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path, ignore_errors=False) if path.exists() else None
            cleanup["models"].append(
                {"id": identifier, "removed_bytes": size, "removed": not path.exists()}
            )
            write(evidence / "cleanup.json", cleanup)
    # Dataset identities, ground truth and model file hashes are already retained as evidence.
    remaining = sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
    shutil.rmtree(work)
    cleanup.update(
        remaining_cache_and_dataset_removed_bytes=remaining,
        owned_directory_removed=not work.exists(),
        completed_unix_seconds=time.time(),
    )
    write(evidence / "cleanup.json", cleanup)
    print("CLEANUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
