"""Capture immutable Git source objects and the owner's pre-change workspace.

No model downloads or environment changes. Run with Python 3.12; --fetch requires
network access. Extracted bytes are retained outside the distributable package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = [
    ("mlx", "ml-explore/mlx", "1f8e74e3f12f31365464a6867c6579f0e9b29d85", "MIT"),
    ("mlx_comparison", "ml-explore/mlx", "36969c094354086e15d2a327a9c50feb20b9cf11", "MIT"),
    ("mlx_lm", "ml-explore/mlx-lm", "93e6d4a6447b3080861985947f0820e76fcb957d", "MIT"),
    ("omlx", "jundot/omlx", "9de79bc39a01c1f3fec1e849903aaf0dd5478d7a", "Apache-2.0"),
]
PATHS = {
    "mlx": [
        "LICENSE",
        "mlx/backend/metal/quantized.cpp",
        "mlx/backend/metal/kernels/quantized.h",
        "python/mlx/nn/layers/quantized.py",
    ],
    "mlx_lm": [
        "LICENSE",
        "mlx_lm/generate.py",
        "mlx_lm/server.py",
        "mlx_lm/sample_utils.py",
        "mlx_lm/tokenizer_utils.py",
        "mlx_lm/utils.py",
        "mlx_lm/models/cache.py",
        "mlx_lm/models/base.py",
        "mlx_lm/models/activations.py",
        "mlx_lm/models/pipeline.py",
        "mlx_lm/models/qwen2.py",
        "mlx_lm/models/qwen3.py",
        "mlx_lm/models/llama.py",
        "mlx_lm/models/lfm2.py",
        "mlx_lm/models/rope_utils.py",
    ],
    "omlx": [
        "LICENSE",
        "NOTICE",
        "docs/custom-kernels.md",
        "omlx/engine_core.py",
        "omlx/custom_kernels/quant_matmul/fast.py",
        "omlx/custom_kernels/quant_matmul/csrc/bindings.cpp",
        "omlx/custom_kernels/quant_matmul/csrc/MLX_LICENSE.txt",
        "setup.py",
    ],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def workspace(output):
    output.mkdir(parents=True, exist_ok=False)
    diff = git("diff", "--binary")
    (output / "preexisting.patch").write_bytes(diff)
    status = git("status", "--porcelain=v1", "-uall").decode()
    (output / "status.txt").write_text(status)
    historical = ROOT / "local/competitor-comparison/runs/comparison-20261101"
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "branch": git("branch", "--show-current").decode().strip(),
        "status": status,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))
        },
        "historical_sha256": {
            str(p.relative_to(historical)): sha(p)
            for p in sorted(historical.rglob("*"))
            if p.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"workspace captured: {output}", flush=True)


def sources(directory, fetch):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name, repo, revision, license_name in PINS:
        clone = directory / name
        if not (clone / ".git").is_dir():
            clone.mkdir(exist_ok=True)
            git("init", cwd=clone)
        if fetch:
            subprocess.run(
                ["git", "fetch", "--depth=1", f"https://github.com/{repo}.git", revision],
                cwd=clone,
                check=True,
            )
        # Both commands must succeed; never resolve against an enclosing repo.
        git("cat-file", "-e", f"{revision}^{{commit}}", cwd=clone)
        verified = (
            git("rev-parse", "--verify", f"{revision}^{{commit}}", cwd=clone).decode().strip()
        )
        if verified != revision:
            raise ValueError("source object identity mismatch")
        tree = git("ls-tree", "-r", "--name-only", revision, cwd=clone).decode().splitlines()
        (clone / "tree.txt").write_text("\n".join(tree) + "\n")
        record = {
            "name": name,
            "repository": f"https://github.com/{repo}",
            "revision": verified,
            "license": license_name,
            "files": [],
            "missing": [],
        }
        if name == "omlx":
            record["license_notes"] = (
                "Root Apache-2.0; quant_matmul/csrc includes Apple MLX MIT license. "
                "No oMLX code copied into MLX-One."
            )
        for path in PATHS["mlx" if name == "mlx_comparison" else name]:
            if path not in tree:
                record["missing"].append(path)
                continue
            raw = git("show", f"{revision}:{path}", cwd=clone)
            destination = clone / "captured" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            file_record = {
                "path": path,
                "sha256": sha(destination),
                "local_capture": str(destination.relative_to(ROOT)),
            }
            if path.endswith("MLX_LICENSE.txt"):
                file_record["license"] = "MIT"
            record["files"].append(file_record)
        records.append(record)
        print(name, verified, "missing:", record["missing"], flush=True)
    manifest = {
        "schema_version": 1,
        "upstream_repositories": records,
        "adapted_components": [],
        "scope": "Source study only; no newly copied or adapted upstream runtime component.",
    }
    (directory / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["workspace", "sources"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.operation == "workspace":
        workspace(args.output)
    else:
        sources(args.output, args.fetch)


if __name__ == "__main__":
    main()
