"""Prepare portable evidence while retaining large raw oracle tensors locally."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/qualification/batch-1-sub-6b"
PRIVATE = ROOT / "local/evidence/batch1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    tensors = 0
    for source in PUBLIC.rglob("*.npz"):
        relative = source.relative_to(PUBLIC)
        target = PRIVATE / "reference-tensors" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("refuse to overwrite an archived oracle: " + str(relative))
        arrays = np.load(source, allow_pickle=False)
        records = {}
        for name in arrays.files:
            value = arrays[name]
            records[name] = dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                preview=value.reshape(-1)[:16].tolist(),
            )
            if name in {"output", "generation", "logits"} and value.size <= 10000:
                records[name]["values"] = value.tolist()
        arrays.close()
        receipt = dict(
            archive_sha256=digest(source),
            bytes=source.stat().st_size,
            local_archive=str(target.relative_to(ROOT)),
            tensors=records,
            note=(
                "Large real-checkpoint oracle outputs remain in ignored local evidence; "
                "rerun pinned workers to reproduce."
            ),
        )
        source.with_name(source.stem + "-artifact.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
        shutil.move(source, target)
        tensors += 1
    replacements = {
        str(ROOT): "<checkout>",
        "/Users/susant-achary/dev_work/git_codes/c_base": "<workspace>",
        "/Users/susant-achary": "<user>",
        "/private/tmp/mlx-one-batch1-20261109": "<campaign-work>",
        "/private/tmp/mlx-one-all40-20261108/reference-env": "<reference-env>",
        "/private/tmp/mlx-one-ci-repair-20261109": "<python-matrix-work>",
    }
    sanitized = 0
    for source in PUBLIC.rglob("*"):
        if not source.is_file() or source.suffix not in {".json", ".log", ".md", ".txt"}:
            continue
        original = source.read_text()
        updated = original
        for before, after in replacements.items():
            updated = updated.replace(before, after)
        if original == updated:
            continue
        raw = PRIVATE / "raw-text" / source.relative_to(PUBLIC)
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            shutil.copyfile(source, raw)
        source.write_text(updated)
        sanitized += 1
    print(f"Archived {tensors} oracle arrays; sanitized {sanitized} portable evidence files")


if __name__ == "__main__":
    main()
