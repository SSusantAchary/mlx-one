#!/usr/bin/env python3
"""Thin offline receipt wrapper around pytest and the existing native task runner.

No registry promotion, package build or model download. Each attempt has its own
directory; rerun an unfinished task without overwriting earlier failure proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def snapshot(root=ROOT):
    """Bind active Python, tests, scripts and packaged assets, including local code."""
    paths = set()
    for name in ("src/mlx_one", "tests", "scripts", "examples", "benchmarks", "qualification"):
        for directory, folders, files in os.walk(root / name):
            folders[:] = [f for f in folders if f not in {
                "__pycache__", "results", "node_modules", ".venv", ".git"}]
            for filename in files:
                path = Path(directory) / filename
                if filename.endswith((".pyc", ".pyo", ".safetensors", ".bin", ".npy")):
                    continue
                # Private path manifests affect runs, not distributable identity.
                if filename.endswith(".local.json"):
                    continue
                paths.add(path)
    for name in ("pyproject.toml", "README.md", "LICENSE", "CITATION.bib"):
        paths.add(root / name)
    rows = {str(p.relative_to(root)): digest(p) for p in sorted(paths) if p.is_file()}
    sha = hashlib.sha256("".join(f"{p}\0{sha}\n" for p, sha in rows.items()).encode()).hexdigest()
    return {"schema_version": "alpha-validation-source-v1", "source_sha256": sha,
            "files": rows}


def verify_task_receipt(directory, expected_profile_sha256):
    """Re-use integrity and exact-scope validators; pending is never success."""
    from mlx_one.inference_qualification import validate_integrity
    from mlx_one.registry import InferenceScope
    from mlx_one.schemas import ArtifactRef
    from mlx_one.task_qualification import TASK_CORRECTNESS_CHECKS, validate_task_evidence

    directory = Path(directory)
    validate_integrity(directory, required=("manifest.json", "task-evidence.json",
                                          "task-gates.json", "task-runtime.json"))
    if (directory / "failure.json").exists():
        raise ValueError("native task retains a failure or blocked outcome")
    manifest = json.loads((directory / "manifest.json").read_text())
    gates = json.loads((directory / "task-gates.json").read_text())
    if manifest.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("task profile changed or receipt belongs to a different profile")
    if any(gates.get("checks", {}).get(key) is not True for key in TASK_CORRECTNESS_CHECKS):
        raise ValueError("native task has failed, pending or missing semantic gates")
    result = json.loads((directory / "task-evidence.json").read_text())
    scope = InferenceScope(**result["scope"])
    if not scope.complete or result.get("status") not in {"passed", "pending"}:
        raise ValueError("native task has incomplete scope or non-PASS evidence")
    evidence = [ArtifactRef(uri=str(directory / "task-evidence.json"), kind="native-task")]
    for artifact in result.get("artifacts", []):
        relative = Path(artifact["uri"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("task backing artifact must be contained in its receipt")
        path = directory / relative
        if path.is_symlink() or digest(path) != artifact.get("sha256"):
            raise ValueError("task backing artifact checksum mismatch")
        evidence.append(ArtifactRef(uri=str(path), kind=artifact["kind"],
                                    sha256=artifact["sha256"]))
    validate_task_evidence(result, scope, tuple(evidence))
    return {"correctness_checks": sorted(TASK_CORRECTNESS_CHECKS),
            "aggregate_status": gates.get("status"),
            "scope": "task correctness only; no performance or alpha support promotion"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("backend-free", "task"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--allow-metal", action="store_true",
                        help="Explicitly authorize this bounded native task invocation")
    args = parser.parse_args(argv)
    artifact = args.artifact_root.expanduser().resolve()
    if artifact.is_relative_to(ROOT):
        ignored = subprocess.run(["git", "check-ignore", "-q", str(artifact / "receipt")],
                                 cwd=ROOT, check=False)
        if ignored.returncode:
            parser.error("artifact root inside the checkout must be Git-ignored")
    output = artifact / f"{args.stage}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    source = snapshot()
    (output / "source.private.json").write_text(json.dumps(source, indent=2) + "\n")
    result = {"stage": args.stage, "status": "NOT_RUN", "qualification_promoted": False,
              "validation_source_sha256": source["source_sha256"]}
    code = 1
    try:
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "MLX_ONE_OFFLINE": "1", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}"}
        if args.stage == "backend-free":
            for key in list(environment):
                if key.startswith("MLX_ONE_RUN_"):
                    environment[key] = "0"
            command = [sys.executable, "-m", "pytest", "--no-cov", "-q", "tests",
                       f"--junitxml={output / 'pytest.junit.xml'}"]
        else:
            if not args.allow_metal:
                raise PermissionError("HARDWARE_PENDING: explicit --allow-metal is required")
            if args.profile is None:
                raise ValueError("native task requires --profile")
            from mlx_one.task_qualification import read_inputs, read_profile

            profile = read_profile(args.profile)
            cases, _ = read_inputs(profile)
            if (profile.slots != 1 or profile.repetitions != 5
                    or profile.timeout_seconds > 600 or len(cases["cases"]) > 8):
                raise ValueError("alpha task requires B1, five repetitions, <=600s and <=8 cases")
            result["profile_sha256"] = profile.digest
            command = [sys.executable, str(ROOT / "benchmarks/native_tasks.py"), "run",
                       "--profile", str(args.profile.resolve()), "--output", str(output / "native")]
        result["command"] = command
        with (output / "stdout.log").open("wb") as stdout, (
            output / "stderr.log"
        ).open("wb") as stderr:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout,
                                       stderr=stderr, check=False)
        result["exit_code"] = completed.returncode
        if completed.returncode:
            raise ValueError("validation subprocess failed; inspect retained stdout/stderr")
        if snapshot()["source_sha256"] != source["source_sha256"]:
            raise ValueError("validation source changed during this invocation")
        if args.stage == "task":
            result["task_readout"] = verify_task_receipt(
                output / "native", result["profile_sha256"])
        result["status"] = ("BACKEND_FREE_PASS" if args.stage == "backend-free"
                            else "TASK_CORRECTNESS_GATES_PASS")
        code = 0
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="HARDWARE_PENDING" if isinstance(error, PermissionError) else "FAIL",
                      reason=str(error), error_type=type(error).__name__)
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("stage", "status",
                                    "validation_source_sha256", "qualification_promoted"))
            writer.writeheader()
            writer.writerow({key: result[key] for key in writer.fieldnames})
        receipts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
                    if p.is_file() and p.name != "sha256-manifest.json"}
        (output / "sha256-manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
