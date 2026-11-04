#!/usr/bin/env python3
"""Sequential bounded DETR/depth real-image matrices using shared comparators."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from compare_tensors import compare_archives

from mlx_one.inference_baseline import TOLERANCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", choices=("detr", "depth"), default="detr")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    rows = []
    images = (
        sorted(args.images.glob("*.jpg"))
        if args.family == "detr"
        else sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    )
    for image in images:
        folder = args.output / image.stem
        folder.mkdir()
        commands = []
        for backend, python in [
            ("reference", root / "local/native_tasks/reference_venv/bin/python"),
            ("native", Path(sys.executable)),
        ]:
            command = [
                str(python),
                str(root / f"benchmarks/native_tasks/export_{args.family}.py"),
                backend,
                str(
                    root
                    / "local/native_tasks/checkpoints"
                    / ("detr" if args.family == "detr" else "depth-anything")
                ),
                str(image),
                str(folder / (backend + ".npz")),
            ]
            commands.append(command)
            completed = subprocess.run(
                command,
                cwd=root,
                env={**os.environ, "MLX_ENABLE_TF32": "0"},
                capture_output=True,
                text=True,
                timeout=120,
            )
            (folder / (backend + ".log")).write_text(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{backend} export failed for {image}")
        report = compare_archives(
            folder / "reference.npz",
            folder / "native.npz",
            atol=TOLERANCES["float32"][0],
            rtol=TOLERANCES["float32"][1],
        )
        (folder / "comparison.json").write_text(json.dumps(report, indent=2))
        row = {
            "image": str(image),
            "passed": report["passed"],
            "first_divergent": report["first_divergent"],
            "commands": commands,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (args.output / "summary.json").open("x") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
