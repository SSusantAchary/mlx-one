#!/usr/bin/env python3
"""Bounded real-image CLIP matrix using the existing exporters and comparator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from compare_tensors import compare_archives

from mlx_one.inference_baseline import TOLERANCES

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "local/native_tasks/runs/m2-20261103/clip"
REF = ROOT / "local/native_tasks/reference_venv/bin/python"
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/clip-converted"
CASES = {
    "astronaut.png": "astronaut",
    "coffee.png": "coffee",
    "chelsea.png": "cat",
    "rocket.jpg": "rocket",
    "camera.png": "photographer",
    "clock_motion.png": "clock",
}
TEN = [
    "astronaut",
    "coffee",
    "cat",
    "rocket",
    "photographer",
    "clock",
    "dog",
    "car",
    "flower",
    "bird",
]
vit = json.loads((ROOT / "local/native_tasks/checkpoints/vit/config.json").read_text())["id2label"]
HUNDRED = list(TEN)
for name in vit.values():
    if name not in HUNDRED:
        HUNDRED.append(name)
    if len(HUNDRED) == 100:
        break
results = []
for image, label in CASES.items():
    for count in (2, 10, 100):
        labels = (
            [label, "dog" if label == "cat" else "cat"]
            if count == 2
            else TEN
            if count == 10
            else HUNDRED
        )
        folder = RUN / "matrix" / f"{Path(image).stem}-{count}"
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "labels.json").open("x") as out:
            json.dump(labels, out)
        command = [
            str(ROOT / "benchmarks/native_tasks/export_clip.py"),
            None,
            str(CHECKPOINT),
            str(RUN / "real-images" / image),
            str(folder / "labels.json"),
            None,
        ]
        for backend, python in [("reference", REF), ("native", Path(sys.executable))]:
            arguments = command.copy()
            arguments[1] = backend
            arguments[-1] = str(folder / (backend + ".npz"))
            env = {**os.environ, "MLX_ENABLE_TF32": "0"}
            completed = subprocess.run(
                [str(python), *arguments],
                env=env,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
            )
            with (folder / (backend + ".log")).open("x") as out:
                out.write(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"{backend} export failed: {folder}")
        report = compare_archives(
            folder / "reference.npz",
            folder / "native.npz",
            atol=TOLERANCES["float32"][0],
            rtol=TOLERANCES["float32"][1],
        )
        with (folder / "comparison.json").open("x") as out:
            json.dump(report, out, indent=2, allow_nan=False)
        with (
            np.load(folder / "reference.npz", allow_pickle=False) as ref,
            np.load(folder / "native.npz", allow_pickle=False) as native,
        ):
            expected = labels[int(np.argmax(ref["probabilities"][0]))]
            actual = labels[int(np.argmax(native["probabilities"][0]))]
        row = {
            "image": image,
            "expected_label": label,
            "candidates": count,
            "hf_top1": expected,
            "native_top1": actual,
            "correct": actual == label,
            "top1_agreement": actual == expected,
            "source_parity_pass": report["passed"],
            "first_divergent": report["first_divergent"],
        }
        results.append(row)
        print(row, flush=True)
with (RUN / "matrix-summary.json").open("x") as out:
    json.dump(results, out, indent=2)
