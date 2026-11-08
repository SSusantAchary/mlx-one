"""Real SAM component comparisons with pinned Torch reference and natural video."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


def qualify(entry, model, fixtures, report):
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    work = Path(report["fixtures_directory"]).parent
    source = work / "reference-sources/transformers-6133195dcb027feb7502379acd13750e167eb6c5/src"
    tool = root / "benchmarks/sam3_qualification.py"
    task = dict(
        sam3="concept",
        sam3_tracker="interactive",
        sam3_video="video",
        sam3_tracker_video="tracking",
    )[entry["model_type"]]
    video = task in {"video", "tracking"}
    natural = work / "fixtures/sam-natural/frames"
    image = natural / "001.png" if video else Path(fixtures["segmentation"]["path"])
    results = []
    if not report.get("reference_python"):
        raise ValueError("SAM qualification requires an isolated pinned reference interpreter")
    for streaming in [False, True] if video else [False]:
        paths = {}
        for backend, python in [
            ("native", sys.executable),
            ("reference", report["reference_python"]),
        ]:
            path = model.parent / f"{task}-{backend}-{streaming}.npz"
            command = [
                str(python),
                str(tool),
                "--model",
                str(model),
                "--task",
                task,
                "--backend",
                backend,
                "--image",
                str(image),
                "--text",
                "person" if video else "truck",
                "--output",
                str(path),
                "--frames",
                "32",
            ]
            if video:
                command += ["--video-frames", str(natural)]
            if streaming:
                command += ["--streaming"]
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
            env.pop("PYTHONPATH", None)
            if backend == "reference":
                env["PYTHONPATH"] = str(source) + os.pathsep + str(root / "src")
            log = (
                Path(report["evidence_directory"])
                / f"{entry['model_type']}-{backend}-{streaming}.log"
            )
            with log.open("w") as stream:
                completed = subprocess.run(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1500
                )
            report.setdefault("reference_runs", []).append(
                dict(
                    backend=backend,
                    device="cpu" if backend == "reference" else "mlx-metal",
                    streaming=streaming,
                    exit_code=completed.returncode,
                    log=log.name,
                    output_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                    if path.exists()
                    else None,
                )
            )
            if completed.returncode:
                raise ValueError(f"SAM {task} {backend} failed; see {log.name}")
            paths[backend] = path
        with np.load(paths["native"]) as native, np.load(paths["reference"]) as reference:
            if video:
                sys.path.insert(0, str(root / "benchmarks"))
                from sam3_compare import compare_video

                comparison = compare_video(native, reference, frames=32)
                positive = all(
                    np.asarray(native[f"frame_{i}_masks"]).any()
                    and np.asarray(reference[f"frame_{i}_masks"]).any()
                    for i in range(32)
                )
                comparison["foreground_in_every_frame"] = bool(positive)
                comparison["passed"] = comparison["passed"] and positive
            else:
                a, b = native["masks"].astype(bool), reference["masks"].astype(bool)
                if a.shape != b.shape:
                    raise ValueError("SAM image mask inventory differs from the reference")
                union = np.logical_or(a, b).sum(axis=(-2, -1))
                iou = np.logical_and(a, b).sum(axis=(-2, -1)) / np.maximum(union, 1)
                key = "scores" if task == "concept" else "predicted_iou"
                score_delta = float(np.max(np.abs(native[key] - reference[key])))
                comparison = dict(
                    mask_count=len(a),
                    mask_iou=iou.tolist(),
                    score_delta=score_delta,
                    passed=bool(
                        a.any() and b.any() and np.min(iou) >= 0.98 and score_delta <= 0.02
                    ),
                )
                if task == "interactive":
                    comparison["intermediate_max_absolute_errors"] = {
                        key: float(np.max(np.abs(native[key] - reference[key])))
                        for key in native.files
                        if key in reference.files and key not in {"masks", "boxes"}
                    }
                if task == "concept":
                    comparison["max_box_delta_px"] = float(
                        np.max(np.abs(native["boxes"] - reference["boxes"]))
                    )
                    comparison["passed"] &= comparison["max_box_delta_px"] <= 2
            results.append(dict(streaming=streaming, **comparison))
        report["sam_reference_comparison"] = dict(
            source_revision="6133195dcb027feb7502379acd13750e167eb6c5",
            task=task,
            helper_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),
            fixture="first 32 natural bedroom video frames" if video else "pinned HF truck image",
            modes=results,
        )
        assert comparison["passed"], "SAM mask/score/box/identity reference gate failed"
    report["quality_status"] = (
        "PASS_PINNED_NATURAL_VIDEO_REFERENCE" if video else "PASS_PINNED_IMAGE_MASK_REFERENCE"
    )
