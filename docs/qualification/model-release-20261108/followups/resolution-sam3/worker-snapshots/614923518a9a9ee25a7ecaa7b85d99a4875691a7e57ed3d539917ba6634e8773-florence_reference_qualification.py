"""Publisher comparison, retained as diagnosis alongside the original page gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path


def diagnose(model, image, page_sample, report):
    import numpy as np
    from rapidfuzz.distance import Levenshtein

    from mlx_one.ocr.postprocessing import visible_text

    helper = Path(__file__).with_name("florence_reference_probe.py")
    work = Path(report["fixtures_directory"]).parent
    outputs = {}
    for backend, python in [("native", sys.executable), ("reference", report["reference_python"])]:
        output = model.parent / f"florence-{backend}.npz"
        log = Path(report["evidence_directory"]) / f"florence-{backend}.log"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"}
        env.pop("PYTHONPATH", None)
        if backend == "reference":
            env["PYTHONPATH"] = str(
                work
                / "reference-sources/florence-publisher-f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"
            )
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    str(python),
                    str(helper),
                    "--backend",
                    backend,
                    "--model",
                    str(model),
                    "--image",
                    str(image),
                    "--output",
                    str(output),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        report.setdefault("florence_reference_runs", []).append(
            dict(backend=backend, exit_code=result.returncode, log=log.name)
        )
        if result.returncode:
            raise ValueError(f"Florence {backend} diagnostic failed; see {log.name}")
        outputs[backend] = output
    metrics = {}
    with np.load(outputs["native"]) as native, np.load(outputs["reference"]) as reference:
        for key in native.files:
            a, b = native[key], reference[key]
            metrics[key] = dict(native_shape=list(a.shape), reference_shape=list(b.shape))
            if a.shape == b.shape:
                metrics[key].update(
                    max_absolute_error=float(np.max(np.abs(a - b))),
                    mean_absolute_error=float(np.mean(np.abs(a - b))),
                )
    output = json.loads(outputs["reference"].with_suffix(".json").read_text())
    normalize = lambda value: " ".join(unicodedata.normalize("NFKC", visible_text(value)).split())
    expected, actual = (
        normalize(page_sample["fixture"]["reference"]),
        normalize(output["text"].replace("<s>", "").replace("</s>", "")),
    )
    report["florence_reference_diagnostic"] = dict(
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        source_revision="f6c1a25888ffc1d945ee8a1a77ac833c7303d46e",
        metrics=metrics,
        reference_text=output["text"],
        reference_page_cer=Levenshtein.distance(expected, actual) / len(expected),
        scope="Exact pinned publisher code on CPU FP32; original native page gate remains mandatory",
    )
