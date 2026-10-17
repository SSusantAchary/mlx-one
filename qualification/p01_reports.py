"""Build truthful P0.1 coverage matrices from supervised, checksummed local evidence."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from mlx_one.inference_baseline import source_provenance
from mlx_one.inference_qualification import write_integrity
from mlx_one.p0_engine_qualification import atomic_json, csv_write
from qualification.p01_comparison import read_worker
from qualification.p01_experiments import validate_result
from qualification.p01_math import WIDTHS

MODELS = ("mlx-community/Qwen3-0.6B-8bit", "mlx-community/Qwen2.5-0.5B-Instruct-8bit",
          "mlx-community/LFM2-350M-4bit", "mlx-community/Qwen3.5-0.8B-4bit")
OPERATORS = (("reference", "reference"), ("mlx_sdpa", "reference"),
             ("reference", "mlx_qmm"), ("mlx_sdpa", "mlx_qmm"))


def coverage_catalog():
    return [{"checkpoint": model, "prompt_tokens": width, "chunk": chunk,
        "attention": attention, "projection": projection, "logits": "last_only",
        "native_parity": "NOT_RUN", "performance": "PENDING", "release": "NOT_RUN",
        "release_suites": {suite: "NOT_RUN" for suite in
                           ("correctness", "streaming", "state", "lifecycle")},
        "artifacts": [], "reason": "required native evidence not executed"}
        for model, width, chunk, (attention, projection) in product(
            MODELS, WIDTHS, (None, 256, 512, 1024, 2048), OPERATORS)]


def record_failure(row, task):
    """Keep failed streaming/timing gates separate from mathematical parity."""
    status, workload = task.get("status"), task.get("workload")
    if status not in {"FAIL", "BLOCKED", "NOT_SUPPORTED"}:
        return
    row.setdefault("failures", []).append({"workload": workload, "status": status,
        "reason": task.get("reason"), "artifact": task.get("artifact")})
    if workload == "parity":
        row["native_parity"] = status
    elif workload == "release":
        row["release"] = status
        for suite in task.get("suites", []):
            row["release_suites"][suite] = status
    elif workload in {"engine", "controlled"}:
        row["performance"] = status
    row["reason"] = task.get("reason") or f"{workload} {status.lower()}"


def collect(root, source):
    rows = coverage_catalog()
    index = {(r["checkpoint"], r["prompt_tokens"], r["chunk"], r["attention"], r["projection"]): r
             for r in rows}
    historical, rejected, running = [], [], []
    for path in sorted(Path(root).rglob("manifest.json")):
        if "data" in path.parts:
            continue
        try:
            cohort = json.loads(path.read_text())
            if not isinstance(cohort.get("tasks"), list):
                continue
            if cohort.get("source", {}).get("source_sha256") != source["source_sha256"]:
                historical.append(str(path))
                continue
            if cohort.get("status") != "completed":
                running.append({"artifact": str(path), "status": cohort.get("status")})
            for task in cohort["tasks"]:
                operator = task.get("operators", {})
                key = (task["checkpoint"], task.get("prompt_tokens"), task.get("chunk"),
                       operator.get("attention"), operator.get("projection"))
                row = index.get(key)
                if row is None or task.get("route") != "optimized":
                    continue
                if task.get("status") != "PASS":
                    record_failure(row, task)
                    continue
                artifact = Path(task["artifact"]) / "data/experiment.json"
                result = read_worker(artifact)
                manifest = json.loads((artifact.parent.parent / "manifest.json").read_text())
                if manifest["source"]["source_sha256"] != source["source_sha256"]:
                    raise ValueError("cohort and supervised worker source mismatch")
                if any(manifest["task"].get(key) != task.get(key) for key in
                       ("checkpoint", "prompt_tokens", "chunk", "operators", "route", "workload")):
                    raise ValueError("cohort and supervised worker execution mismatch")
                validate_result(result, task, manifest["native"]["metadata"], manifest["source"])
                row["artifacts"].append(str(artifact))
                if task["workload"] == "parity" and row["native_parity"] not in {"FAIL", "BLOCKED"}:
                    row.update(native_parity="PASS",
                               reason="native numerical evidence; no promotion")
                if task["workload"] == "release":
                    for suite in task["suites"]:
                        if row["release_suites"][suite] not in {"FAIL", "BLOCKED"}:
                            row["release_suites"][suite] = "PASS"
                    if all(status == "PASS" for status in row["release_suites"].values()):
                        row["release"] = "CORPUS_PASS"
                # Timing worker success is only a measurement, never a passed
                # materiality, protected-matrix or competitor gate.
                if task["workload"] in {"engine", "controlled"}:
                    if row["performance"] not in {"FAIL", "BLOCKED"}:
                        row["performance"] = "MEASURED_UNQUALIFIED"
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append({"artifact": str(path), "reason": str(error)})
    return {"schema_version": "native-math-coverage-v1", "source": source, "rows": rows,
        "historical_cohorts": historical, "running_or_blocked_cohorts": running,
        "rejected_artifacts": rejected, "qualification_promoted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("qualification/results/p01_math"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect(args.root, source_provenance(Path.cwd()))
    atomic_json(args.output / "matrix.json", report)
    csv_write(args.output / "matrix.csv", report["rows"])
    table = ["| Checkpoint | Prompt | Chunk | Attention | Projection |"
             " Native parity | Performance |",
             "|---|---:|---:|---|---|---|---|"]
    table.extend("| " + " | ".join(str(row[key]) for key in ("checkpoint", "prompt_tokens", "chunk",
        "attention", "projection", "native_parity", "performance")) + " |"
        for row in report["rows"])
    (args.output / "matrix.md").write_text("\n".join(table) + "\n")
    write_integrity(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
