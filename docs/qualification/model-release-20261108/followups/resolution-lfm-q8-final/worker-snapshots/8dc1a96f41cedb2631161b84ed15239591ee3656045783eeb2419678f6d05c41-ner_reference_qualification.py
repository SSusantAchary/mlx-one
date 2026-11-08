"""Compare frozen CoNLL cases with independently executed publisher BERT outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def compare(model, rows, reference_python, candidate_python):
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    tool = root / "benchmarks/native_tasks/export_ner.py"
    cases = model.parent / "ner-cases.json"
    cases.write_text(json.dumps([" ".join(row["tokens"]) for row in rows]))
    logs = {}
    for backend, python in [("native", candidate_python), ("reference", reference_python)]:
        completed = subprocess.run(
            [
                str(python),
                str(tool),
                backend,
                str(model),
                str(cases),
                str(model.parent / (backend + "-ner.npz")),
                str(model.parent / (backend + "-ner-tasks.json")),
                "--batch-size",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        logs[backend] = dict(
            exit_code=completed.returncode, log=(completed.stdout + completed.stderr)[-4000:]
        )
        if completed.returncode:
            return dict(passed=False, subprocesses=logs)
    differences = {}
    tensor_keys_equal = True
    with (
        np.load(model.parent / "native-ner.npz") as native,
        np.load(model.parent / "reference-ner.npz") as reference,
    ):
        tensor_keys_equal = set(native.files) == set(reference.files)
        for key in reference.files:
            a, b = native[key], reference[key]
            group = key.split("_", 2)[-1]
            if a.shape != b.shape:
                differences.setdefault(group, []).append(
                    dict(shape_a=list(a.shape), shape_b=list(b.shape))
                )
            elif np.issubdtype(a.dtype, np.floating):
                differences.setdefault(group, []).append(dict(max_abs=float(np.abs(a - b).max())))
            else:
                differences.setdefault(group, []).append(dict(exact=bool(np.array_equal(a, b))))
    outputs = {
        backend: json.loads((model.parent / (backend + "-ner-tasks.json")).read_text())
        for backend in ("native", "reference")
    }
    numeric = max(v["max_abs"] for values in differences.values() for v in values if "max_abs" in v)
    inputs_equal = all(v.get("exact", True) for values in differences.values() for v in values)
    shapes_equal = all("shape_a" not in v for values in differences.values() for v in values)
    return dict(
        passed=numeric <= 1e-3 and inputs_equal and shapes_equal and tensor_keys_equal,
        shapes_equal=shapes_equal,
        tensor_keys_equal=tensor_keys_equal,
        maximum_error=numeric,
        inputs_equal=inputs_equal,
        differences=differences,
        outputs=outputs,
        subprocesses=logs,
        reference="transformers==4.57.1; pinned source weights; CPU FP32",
    )
