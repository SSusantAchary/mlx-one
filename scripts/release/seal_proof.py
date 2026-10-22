"""Check and index an executed alpha proof; this never grants release approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_REPORTS = (
    "release-scope.md environment.json source-provenance.json reference-import-audit.json "
    "checkpoint-manifest.json mathematical-fidelity.md cache-semantics.md "
    "numerical-execution-shapes.md numerical-policy-proposal.md "
    "upstream-vs-mlx-one-cache-tests.md model-quality.md precision-qualification.md "
    "api-regression.md hardware-qualification.md inherited-failure-review.md "
    "security-and-licenses.md package-audit.md wheel-installation.md "
    "model-support-matrix.md known-limitations.md release-gate-summary.json release-decision.md"
).split()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(root, values):
    for relative, expected in values.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), "unsafe path: " + relative)
        require(path.is_file() and sha(path) == expected, "changed file: " + relative)
    return len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    artifact_run = run / read(run / "artifact-selection.json")["directory"]
    frozen = read(artifact_run / "frozen-candidate.json")
    gates = read(run / "release-gate-summary.json")
    records = read(run / "model-records.json")
    assets = read(run / "checkpoint-manifest.json")
    historical = verify_hashes(ROOT, read(run / "historical-evidence.json"))
    references = verify_hashes(ROOT, read(run / "reference-snapshot.json"))
    models = {
        p: h
        for p, h in read(run / "entry-source.json").items()
        if p.startswith("src/mlx_one/models/")
    }
    model_files = verify_hashes(ROOT, models)
    original_gate = "src/mlx_one/inference_baseline.py"
    verify_hashes(ROOT, {original_gate: read(run / "entry-source.json")[original_gate]})
    original_files = verify_hashes(ROOT, frozen["package_source_sha256"])
    candidate_files = verify_hashes(Path(frozen["path"]), frozen["package_source_sha256"])
    require(gates["source_commit"] == frozen["commit"], "source identity disagrees")
    require(len(records) == 10, "model schema: expected ten records")
    require({r["id"] for r in records} == {r["id"] for r in assets}, "model IDs disagree")
    required_fields = {
        "id",
        "model_id",
        "checkpoint_revision",
        "source_commit",
        "profiles",
        "status",
        "alpha_supported",
        "qualified_maximum_context",
        "api_source",
        "api_primary",
        "api_raw",
    }
    allowed_statuses = {
        "ALPHA_SUPPORTED",
        "EXPERIMENTAL_NOT_QUALIFIED",
        "BLOCKED_ASSET",
        "BLOCKED_ENVIRONMENT",
        "FAIL_NUMERICAL",
        "FAIL_STATE",
        "FAIL_API",
        "NOT_RUN",
        "UNSUPPORTED",
    }
    for row in records:
        require(required_fields <= row.keys(), "model schema: missing fields")
        require(row["status"] in allowed_statuses, "model schema: invalid status")
        require(isinstance(row["alpha_supported"], bool), "model schema: support must be bool")
        require(set(row["profiles"]) == {"source", "primary"}, "profile roles missing")
        asset = next(a for a in assets if a["id"] == row["id"])
        require(row["checkpoint_revision"] == asset["revision"], "checkpoint revision disagrees")
        for profile in row["profiles"].values():
            require((run / profile["raw"]).is_file(), "missing raw profile evidence")
    require(
        sum(r["alpha_supported"] for r in records) == gates["alpha_supported_count"],
        "support count disagrees",
    )
    require(records == gates["models"], "model records disagree with release gates")
    from validate import REQUIRED_GATES

    require(
        {g["id"] for g in gates["mandatory_gates"]} == REQUIRED_GATES, "gate coverage incomplete"
    )
    require(
        gates["decision"] == "NO-GO" and not gates["publication_authorized"],
        "this proof has no release approval",
    )
    require(read(artifact_run / "package-audit.json")["status"] == "PASS", "package audit failed")
    verify_hashes(artifact_run / "dist", read(artifact_run / "package-complete.json")["artifacts"])
    packaged = {
        p.removeprefix("src/"): h for p, h in frozen["package_source_sha256"].items()
        if Path(p).name != "README.md"
    }
    for artifact in (artifact_run / "dist").iterdir():
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                payload = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
        elif artifact.name.endswith(".tar.gz"):
            with tarfile.open(artifact) as archive:
                payload = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
        else:
            continue
        actual = {
            n[n.index("mlx_one/"):]: hashlib.sha256(data).hexdigest()
            for n, data in payload.items() if "mlx_one/" in n
        }
        require(all(actual.get(n) == h for n, h in packaged.items()),
                "packaged runtime/resource/notice mismatch: " + artifact.name)
    for name in REQUIRED_REPORTS:
        require((run.parent / name).is_file(), "required report missing: " + name)
    broken = []
    for path in (run / "reports").glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#")[0]).resolve()
            if destination in {run / "proof-integrity.json", run / "evidence-index.json"}:
                continue  # Both are written below by this same successful invocation.
            if not destination.exists():
                broken.append({"file": str(path.relative_to(run)), "target": target})
    require(not broken, "broken cohort report links: " + json.dumps(broken))
    proof = {
        "schema_version": "alpha-proof-integrity-v1",
        "status": "PASS",
        "historical_files_unchanged": historical,
        "reference_files_unchanged": references,
        "model_files_unchanged_since_entry": model_files,
        "frozen_gate_source_unchanged": True,
        "original_package_files_match_frozen": original_files,
        "candidate_package_files_match_frozen": candidate_files,
        "runtime_resource_notice_files_match_both_distributions": len(packaged),
        "required_reports_present": len(REQUIRED_REPORTS),
        "model_schema_validation": "PASS: explicit required-field/type/status/revision checks",
        "model_records": len(records),
        "mandatory_gate_coverage": len(REQUIRED_GATES),
        "source_commit": frozen["commit"],
        "decision": gates["decision"],
        "scope": "Evidence integrity and consistency; numerical failures and approvals unchanged",
    }
    (run / "proof-integrity.json").write_text(json.dumps(proof, indent=2) + "\n")
    files = {
        str(p.relative_to(run)): sha(p)
        for p in sorted(run.rglob("*"))
        if p.is_file() and p.name != "evidence-index.json" and "__pycache__" not in p.parts
    }
    index = {
        "schema_version": "alpha-evidence-index-v1",
        "source_commit": frozen["commit"],
        "hash_algorithm": "sha256",
        "exclusions": ["evidence-index.json (self-reference)", "__pycache__ (not evidence)"],
        "files": files,
    }
    (run / "evidence-index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({**proof, "indexed_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
