"""Fail closed on incomplete, changed or unapproved release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import tomllib

REQUIRED_GATES = {
    "mathematics",
    "cache_semantics",
    "source_and_primary_profiles",
    "framework_quality",
    "interfaces",
    "resources",
    "clean_wheel",
    "licensing_and_dependencies",
    "claims_match_evidence",
    "regressions",
    "frozen_source",
    "lint_disposition",
    "proof_complete",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static(root):
    metadata = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    version = metadata["version"]
    if f'__version__ = "{version}"' not in (root / "src/mlx_one/__init__.py").read_text():
        raise ValueError("release validation failed")
    scope = json.loads((root / "src/mlx_one/data/compatibility/alpha_release.json").read_text())
    if not scope["proposed_version"] == version:
        raise ValueError("release validation failed")
    if not len(scope["checkpoints"]) == 13:
        raise ValueError("release validation failed")
    if not len({r["revision"] for r in scope["checkpoints"]}) == 13:
        raise ValueError("release validation failed")
    if any(r["alpha_supported"] for r in scope["checkpoints"]):
        if not scope["release_approval"] == "APPROVED":
            raise ValueError("release validation failed")
        if scope["admission_policy"] != "functional-architecture-checkpoint-v1":
            raise ValueError("release validation failed")
    if "mlx==0.32.2" not in metadata["dependencies"]:
        raise ValueError("release validation failed")
    return {
        "status": "PASS",
        "version": version,
        "checkpoint_count": 13,
        "release_status": "UNAPPROVED",
    }


def evidence(root, cohort, *, publication=False):
    index = json.loads((cohort / "evidence-index.json").read_text())
    for relative, expected in index["files"].items():
        path = (cohort / relative).resolve()
        if not path.is_relative_to(cohort.resolve()):
            raise ValueError(f"unsafe evidence path: {relative}")
        if not (path.is_file() and sha(path) == expected):
            raise ValueError(f"changed evidence: {relative}")
    gates = json.loads((cohort / "release-gate-summary.json").read_text())
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", gates["source_commit"], "HEAD"], cwd=root, check=True
    )
    package_sources = gates.get("package_source_sha256", gates["production_source_sha256"])
    for relative, expected in package_sources.items():
        path = (root / relative).resolve()
        if not (path.is_relative_to(root.resolve()) and sha(path) == expected):
            raise ValueError(relative)
    if publication:
        if not gates["decision"] == "GO — Restricted alpha":
            raise ValueError("release decision is not GO")
        if not gates["release_owner_approval"] == "APPROVED":
            raise ValueError("release approval missing")
        if {r["id"] for r in gates["mandatory_gates"]} != REQUIRED_GATES:
            raise ValueError("mandatory gate coverage is incomplete")
        scope = json.loads((root / "src/mlx_one/data/compatibility/alpha_release.json").read_text())
        supported = [r for r in scope["checkpoints"] if r["alpha_supported"]]
        if len(supported) != gates["alpha_supported_count"]:
            raise ValueError("package support matrix disagrees with release gates")
        if any(r["id"] == "M05" for r in supported):
            raise ValueError("Qwen3.5 is excluded from this release scope")
        if gates["source_frozen_commit"] is not True:
            raise ValueError("tested source is an uncommitted working tree")
        if not gates.get("package_source_sha256"):
            raise ValueError("complete package source manifest is missing")
        for relative, expected in package_sources.items():
            committed = subprocess.check_output(
                ["git", "show", gates["source_commit"] + ":" + relative], cwd=root
            )
            if not hashlib.sha256(committed).hexdigest() == expected:
                raise ValueError(relative)
        if not gates["alpha_supported_count"] >= gates["approved_minimum_count"]:
            raise ValueError("release validation failed")
        if not all(row["status"] == "PASS" for row in gates["mandatory_gates"]):
            raise ValueError("release validation failed")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        ):
            raise ValueError("release validation failed")
    return {
        "status": "PASS",
        "verified_files": len(index["files"]),
        "publication_checked": publication,
        "decision": gates["decision"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    result = static(args.root)
    if args.evidence:
        result["evidence"] = evidence(args.root, args.evidence, publication=args.publication)
    elif args.publication:
        parser.error("publication requires an evidence bundle")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
