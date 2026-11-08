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
        "Recorded line/page visible text and CER gate; "
        "layout, tables, formulas and handwriting are separate scopes."
    ),
    "segmentation": (
        "Recorded image prompt or first 32 natural-video frames, "
        "pinned reference and applicable CLI; other prompts/videos unqualified."
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
    if len(plan) != 40 or len({row["model_type"] for row in plan}) != 40:
        raise ValueError("candidate scope must enumerate all 40 unique native model types")
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
        cleanup=(cleanup["status"] == "PASS"),
    )
    profiles = []
    for entry, row in zip(plan, rows, strict=True):
        result = read(evidence, row["result_file"])
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
            task_and_sample_quality=row["scoped_review_candidate"],
            interfaces=bool(row["interfaces"]) and "remaining" not in row["interfaces"],
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
                decision="PASS_SCOPED_EXPERIMENTAL" if all(gates.values()) else "BLOCKED",
                gates=gates,
                allowed_scope=SCOPES[row["kind"]],
                quality=row["quality_status"],
                quality_review=row["quality_review"],
                result=row["result_file"],
                result_sha256=sha(evidence / row["result_file"]),
                original_failed_profile=entry.get("original_profile"),
            )
        )
    count = sum(row["decision"] == "PASS_SCOPED_EXPERIMENTAL" for row in profiles)
    document = dict(
        schema_version="mlx-one-scoped-release-candidate-v1",
        qualified_profiles=count,
        expected_profiles=40,
        decision="PASS_SCOPED_RELEASE_CANDIDATE" if count == 40 else "BLOCKED",
        review_kind="automated technical evidence review",
        profiles=profiles,
        publication=dict(performed=False, owner_approval="PENDING", source_frozen_commit=False),
        boundary=(
            "Exact checkpoint/task/precision/fixture/interface/host tuples only. "
            "Historical wheel identities are explicit; affected profiles are replayed. "
            "This does not grant full-family or production certification."
        ),
    )
    (evidence / "release-candidate.json").write_text(json.dumps(document, indent=2) + "\n")
    lines = [
        "# Scoped model release candidate",
        "",
        f"Technical checks: **{count}/40**. Decision: **{document['decision']}**.",
        "",
        "Publication has not been performed. Owner approval and the existing "
        "frozen-source/publication controls remain pending.",
        "",
        "Each row qualifies only the recorded experimental scope. Failed original "
        "profiles and earlier attempts remain in the baseline report and retest histories.",
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
    ]
    (evidence / "RELEASE_CANDIDATE.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in document.items() if k != "profiles"}))


if __name__ == "__main__":
    main()
