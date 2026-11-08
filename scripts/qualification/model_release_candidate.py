"""Review exact candidate tuples, independently of package publication approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from model_release_report import summarize

SCOPES = {
    "text": (
        "Bounded native completion and five-excerpt WikiText sanity; "
        "recorded chat template where applicable."
    ),
    "vlm": "Single-image cat/car prompts, bounded generation and recorded stream lifecycle.",
    "audio": (
        "Recorded English speech clips and exact task bounds; "
        "no general multilingual or conversational-quality claim."
    ),
    "vad": (
        "Speech presence, silence and state controls; "
        "human segment-boundary accuracy is unqualified."
    ),
    "retrieval": (
        "Recorded STS-B or relevant/unrelated controls; "
        "reference checks where recorded, no full MTEB claim."
    ),
    "ocr": (
        "Recorded line/page visible text and diagnostic CER; "
        "layout, tables, formulas and handwriting are separate scopes."
    ),
    "segmentation": (
        "Recorded image prompts or natural-video offline/streaming workflows "
        "and applicable CLI; reference comparisons where recorded."
    ),
    "classification": "Recorded SST-2, CoNLL or CLIP fixture scope; no broad accuracy guarantee.",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("candidate evidence must stay within its cohort")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", default="checkpoint-plan-release.json")
    parser.add_argument("--selections", default="release-retests.json")
    parser.add_argument("--reviews", default="release-evidence-reviews.json")
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    root = Path(__file__).resolve().parents[2]
    plan = read(evidence, args.plan)
    expected = set(json.loads((root / "scripts/release/expected_model_types.json").read_text()))
    if len(plan) != 40 or {row["model_type"] for row in plan} != expected:
        raise ValueError("candidate scope must enumerate all 40 unique native model types")
    catalog = json.loads(
        (root / "src/mlx_one/data/model_support_release_20261108.json").read_text()
    )
    supported = {row["model_type"]: row for row in catalog["models"]}
    if len(catalog["models"]) != 40 or set(supported) != expected:
        raise ValueError("packaged support catalog must match the canonical model inventory")
    decision_path = root / "docs/releases/0.1.1.json"
    approval = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    if approval and approval["support_catalog_sha256"] != sha(
        root / "src/mlx_one/data/model_support_release_20261108.json"
    ):
        raise ValueError("maintainer release decision differs from the support catalog")
    rows = summarize(
        plan,
        evidence,
        selections=read(evidence, args.selections),
        quality_reviews=read(evidence, args.reviews),
    )
    regression = read(evidence, "regression-resolution-audit.json")
    packaging = read(evidence, "package-audit-resolution-final.json")
    build = read(evidence, "candidate-build-resolution-final.json")
    cleanup = read(evidence, "cleanup-resolution-audit.json")
    installed = read(evidence, "clean-wheel-resolution-final.json")
    lineages = [
        read(evidence, name)
        for name in ["candidate-lineage.json", "candidate-lineage-resolution.json"]
    ]
    identities = {
        key
        for item in lineages
        for key, value in item.get("probe_identity_matches", {}).items()
        if value.get("build")
    }
    source_matches = all(
        (root / name).is_file() and sha(root / name) == digest
        for name, digest in build["packaged_files"].items()
    )
    common = dict(
        native_regression=(
            regression["status"] == "PASS"
            and regression["mlx_tests_enabled"] is True
            and sha(evidence / regression["xml"]) == regression["xml_sha256"]
        ),
        distribution=(
            packaging["status"] == "PASS"
            and build["exact_packaged_source_match"] is True
            and source_matches
        ),
        clean_wheel=(
            installed["status"] == "PASS"
            and installed["all_support_claims_bound_to_passing_results"] is True
            and any(
                row["build"] == build["build"]
                and row["installed_record_sha256"] == installed["installed_record_sha256"]
                for row in lineages[-1]["builds"]
            )
        ),
        cleanup=(cleanup["status"] == "PASS"),
    )
    profiles = []
    for entry, row in zip(plan, rows, strict=True):
        result = read(evidence, row["result_file"])
        claim = supported[row["model_type"]]

        def interface_pass(value):
            status = value.get("status", "") if isinstance(value, dict) else value
            return isinstance(status, str) and status.startswith(
                ("PASS", "NOT_ADVERTISED", "NOT_APPLICABLE")
            )

        assets = [
            read(evidence, str(p.relative_to(evidence)))
            for p in (evidence / row["result_file"]).parent.glob("*-assets.json")
        ]
        asset_match = any(
            a.get("checkpoint") == entry["repo"]
            and a.get("revision") == entry["revision"]
            and a.get("files")
            for a in assets
        )
        gates = dict(
            identity=bool(re.fullmatch(r"[0-9a-f]{40,64}", entry["revision"]))
            and bool(asset_match),
            task_execution=(
                row["status"] == "PASS_FUNCTIONAL" and row["functional_status"] == "PASS"
            ),
            interfaces=bool(row["interfaces"])
            and all(interface_pass(value) for value in row["interfaces"].values()),
            packaged_support_claim=(
                claim["status"] == "SUPPORTED"
                and claim["checkpoint"] == row["checkpoint"]
                and claim["revision"] == row["revision"]
                and claim["evidence"] == str((evidence / row["result_file"]).relative_to(root))
                and claim["evidence_sha256"] == sha(evidence / row["result_file"])
            ),
            installed_candidate=result.get("reference_packages_absent") is True
            and result.get("installed_record_sha256") in identities,
            results_before_weight_cleanup=row["cleanup_complete"],
            **common,
        )
        profiles.append(
            dict(
                model_type=row["model_type"],
                checkpoint=row["checkpoint"],
                revision=row["revision"],
                storage_profile=row["profile"],
                execution_dtype=row["execution_dtype"],
                decision="SUPPORTED" if all(gates.values()) else "BLOCKED",
                gates=gates,
                allowed_scope=SCOPES[row["kind"]],
                quality=row["quality_status"],
                quality_review=row["quality_review"],
                result=row["result_file"],
                result_sha256=sha(evidence / row["result_file"]),
                original_failed_profile=entry.get("original_profile"),
            )
        )
    count = sum(row["decision"] == "SUPPORTED" for row in profiles)
    document = dict(
        schema_version="mlx-one-functional-release-candidate-v2",
        support_standard="mlx-lm-aligned-functional-v1",
        qualified_profiles=count,
        expected_profiles=40,
        decision="READY_FOR_RELEASE" if count == 40 else "BLOCKED",
        review_kind="model implementation, real task, interface and distribution checks",
        profiles=profiles,
        publication=dict(
            performed=False,
            owner_approval=approval.get("release_approval", "PENDING"),
            approval_date=approval.get("approval_date"),
            source_frozen_commit=False,
        ),
        boundary=(
            "Support means the model implementation and recorded real workflows pass "
            "engineering checks, following MLX-LM admission practice. "
            "Accuracy/reference benchmarks remain recorded diagnostics rather than "
            "universal support gates; no automatic experimental label is applied."
        ),
    )
    (evidence / "release-candidate.json").write_text(json.dumps(document, indent=2) + "\n")
    lines = [
        "# Model support and release readiness",
        "",
        f"Technical checks: **{count}/40**. Decision: **{document['decision']}**.",
        "",
        f"Maintainer release approval: **{document['publication']['owner_approval']}**"
        + (f" on {approval['approval_date']}." if approval else ".")
        + " Publication has not been performed. The source changes are local; "
        "frozen-source/publication controls still apply when publishing.",
        "",
        "Supported models pass implementation, real task and interface checks under "
        "the user's MLX-LM-aligned standard. Historical benchmark failures remain "
        "visible and are not rewritten as benchmark passes.",
        "",
        f"Final native suite: **{regression['passed']:,} passed**, "
        f"**{regression['counts']['skipped']} skipped**, "
        f"**{regression['subtests_passed']} subtests passed**. "
        "Wheel/sdist payload, clean installed-wheel and downloaded-weight cleanup checks pass. "
        "Protected GPT-OSS model hashes are unchanged.",
        "",
        "| Model type | Exact profile | Technical decision | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in profiles:
        lines.append(
            f"| `{row['model_type']}` | `{row['checkpoint']}` at `{row['revision'][:12]}`; "
            f"{row['storage_profile']} | {row['decision']} | [Result]({row['result']}) |"
        )
    lines += [
        "",
        "See [machine-readable gates and scope](release-candidate.json), "
        "[original cohort](README.md), and [repair decisions](RESOLUTIONS.md).",
        "",
        "Final audits: [regression](regression-resolution-audit.json), "
        "[distribution payload](package-audit-resolution-final.json), "
        "[clean wheel](clean-wheel-resolution-final.json), "
        "[source/build identity](candidate-build-resolution-final.json), "
        "[installed lineage](candidate-lineage-resolution.json), "
        "[weight cleanup](cleanup-resolution-audit.json).",
        "",
        "The existing publication workflow targets the earlier restricted-alpha proof "
        "cohort. This report records current model support; it does not claim that "
        "the legacy publication job has been executed for this cohort.",
        "",
    ]
    (evidence / "RELEASE_CANDIDATE.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in document.items() if k != "profiles"}))


if __name__ == "__main__":
    main()
