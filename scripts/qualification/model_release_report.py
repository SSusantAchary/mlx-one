"""Summarize saved checkpoint probes without turning smoke passes into release approval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(plan, evidence, *, selections=None, quality_reviews=None):
    rows = []
    retests_path = evidence / "retests.json"
    retests = (
        (json.loads(retests_path.read_text()) if retests_path.exists() else {})
        if selections is None
        else selections
    )
    reviews_path = evidence / "evidence-reviews.json"
    reviews = (
        (json.loads(reviews_path.read_text()) if reviews_path.exists() else {})
        if quality_reviews is None
        else quality_reviews
    )
    for entry in plan:
        path = evidence / (entry["model_type"] + ".json")
        if entry["model_type"] in retests:
            path = (evidence / retests[entry["model_type"]]["result"]).resolve()
            if not path.is_relative_to(evidence.resolve()):
                raise ValueError("retest evidence must stay within the campaign")
        result = json.loads(path.read_text()) if path.exists() else {}
        review = {}
        if entry["model_type"] in reviews:
            review_path = (evidence / reviews[entry["model_type"]]).resolve()
            if not review_path.is_relative_to(evidence.resolve()):
                raise ValueError("quality review must stay within the campaign")
            review = json.loads(review_path.read_text())
            if (
                (evidence / review["source_result"]).resolve() != path.resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != review["source_sha256"]
                or review.get("checkpoint") != entry["repo"]
                or review.get("revision") != entry["revision"]
            ):
                raise ValueError("quality review does not match the selected immutable result")
            if not review.get("status", "").startswith("PASS"):
                raise ValueError("a selected positive quality review must have a passing decision")
        if result and (
            result.get("checkpoint") != entry["repo"] or result.get("revision") != entry["revision"]
        ):
            raise ValueError("result identity differs from the pinned checkpoint")
        cleanups = [
            json.loads(candidate.read_text()) for candidate in path.parent.glob("*-cleanup.json")
        ]
        matching = [row for row in cleanups if entry["model_type"] in row.get("model_types", [])]
        if len(matching) > 1:
            raise ValueError("ambiguous checkpoint cleanup records")
        cleanup = matching[0] if matching else {}
        rows.append(
            dict(
                model_type=entry["model_type"],
                kind=entry["kind"],
                checkpoint=entry["repo"],
                revision=entry["revision"],
                profile=entry.get("profile", "stored publisher precision"),
                status=result.get("status", "PENDING"),
                functional_status=result.get(
                    "functional_status",
                    "PASS" if result.get("status") == "PASS_FUNCTIONAL" else "PENDING",
                ),
                release_qualified=result.get("release_qualified", False),
                quality_status=review.get("status", result.get("quality_status", "PENDING")),
                original_quality_status=result.get("quality_status", "PENDING"),
                quality_review=reviews.get(entry["model_type"]),
                execution_dtype=result.get("execution_dtype"),
                interfaces=result.get("interfaces", {}),
                error=result.get("error", ""),
                cleanup_complete=cleanup.get("cleanup_complete", False),
                result_file=str(path.relative_to(evidence)) if path.exists() else None,
                initial_result_file=(entry["model_type"] + ".json")
                if entry["model_type"] in retests
                else None,
            )
        )
        row = rows[-1]
        row["scoped_review_candidate"] = (
            row["status"] == "PASS_FUNCTIONAL"
            and row["quality_status"].startswith("PASS")
            and "remaining" not in row["interfaces"]
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    plan = json.loads((evidence / "checkpoint-plan.json").read_text())
    rows = summarize(plan, evidence)
    counts = Counter(r["status"] for r in rows)
    modalities = {}
    for row in rows:
        count = modalities.setdefault(row["kind"], dict(total=0, functional_passes=0))
        count["total"] += 1
        count["functional_passes"] += row["status"] == "PASS_FUNCTIONAL"
    report = dict(
        schema_version="model-release-readiness-v1",
        expected_model_types=len(plan),
        attempts_recorded=sum(r["status"] != "PENDING" for r in rows),
        status_counts=dict(counts),
        release_qualified_count=sum(r["release_qualified"] for r in rows),
        scoped_review_candidates=sum(r["scoped_review_candidate"] for r in rows),
        modalities=modalities,
        gate="GO" if all(r["release_qualified"] for r in rows) else "BLOCKED",
        models=rows,
    )
    (evidence / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    pending = counts["PENDING"]
    lines = [
        "# Historical extended qualification campaign",
        "",
        "Current support decisions are in [Model support and release readiness]"
        "(RELEASE_CANDIDATE.md). This stricter original-protocol rollup retains historical "
        "quality gates and experimental-review counts; it is not the current model-support "
        "decision. Original failed measurements remain preserved.",
        "",
        f"Release gate: **{report['gate']}**. "
        f"Recorded attempts: **{report['attempts_recorded']}/{len(plan)}**. "
        f"Functional probes passed: **{counts['PASS_FUNCTIONAL']}**. Pending: **{pending}**. "
        f"Release-qualified: **{report['release_qualified_count']}**.",
        "",
        "A functional pass applies to the exact checkpoint revision, precision, "
        "fixture and operations recorded. "
        "It does not certify every checkpoint in a family, every modality/task variant, "
        "corpus quality, long context or other hardware. "
        "Publication approval and existing package support decisions are unchanged.",
        "The campaign tests one representative per registered type. "
        "Passing one representative does not execute every checkpoint/task/precision "
        "profile in the guide.",
        "",
        "Tests execute on the Apple M4 32-GiB host using the built candidate wheel "
        "outside the checkout. "
        "The native worker rejects installed/imported reference frameworks. "
        "Hugging Face fixtures are pinned and identified in [datasets.json](datasets.json). "
        "Raw sample inputs stay in the temporary work directory.",
        "",
        "## Results by modality",
        "",
        "| Modality | Types | Functional passes |",
        "| --- | ---: | ---: |",
    ]
    for kind, count in modalities.items():
        lines.append(f"| {kind} | {count['total']} | {count['functional_passes']} |")
    lines += [
        "",
        f"**{report['scoped_review_candidates']}** records are candidates for a "
        "bounded experimental-support review: their recorded quality sanity checks "
        "and applicable interfaces pass. This is separate from release approval. "
        "Any remaining quality or interface gaps are listed in the per-type "
        "records and next steps.",
        "",
        "## All 40 model types",
        "",
        "| Model type | Probe | Sample quality | Interfaces | Weights removed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        name = row["model_type"]
        probe = f"[{row['status']}]({row['result_file']})" if row["result_file"] else row["status"]
        if row["initial_result_file"]:
            probe += f" ([initial]({row['initial_result_file']}))"
        api = []
        for key, value in row["interfaces"].items():
            status = (
                value.get("status", value.get("exit_code")) if isinstance(value, dict) else value
            )
            api.append(f"{key}: {status}")
        lines.append(
            f"| `{name}` | {probe} | {row['quality_status']} | "
            f"{'; '.join(api) or 'Pending'} | "
            f"{'Yes' if row['cleanup_complete'] else 'Pending'} |"
        )
    failures = [r for r in rows if r["error"]]
    if failures:
        lines += ["", "## Recorded blockers", ""]
        for row in failures:
            error = row["error"].replace("\n", " ").replace("`", "'")
            lines.append(f"- `{row['model_type']}`: {error[:700]}")
    lines += [
        "",
        "## Evidence and cleanup",
        "",
        "- [Initial candidate identity](candidate-build.json), "
        "[final candidate identity](candidate-build-final.json) and "
        "[candidate lineage](candidate-lineage.json). "
        "The campaign includes explicit retests across successive candidate builds, "
        "identified by each worker's installed RECORD digest.",
        "- [Exact checkpoint inventory](checkpoint-plan.json)",
        "- [All explicit retest histories and final selections](retests.json)",
        "- [Remaining release work](NEXT_STEPS.md) and [cleanup audit](cleanup-audit.json)",
        "- [Final native regression log](regression-final.log); "
        "earlier failures and corrections are retained separately.",
        "- [Final package audit](package-audit-final2.json)",
        "- [Cleanup safety tests](campaign-safety.log)",
        "- Each `*-assets.json` records file sizes and SHA-256 hashes before inference.",
        "- Each `*-cleanup.json` records that durable results preceded removal "
        "of campaign-owned weights.",
        "- Existing GPT-OSS directories remain protected. "
        "Reused existing weights are verified and retained.",
        "- Failed and timed-out attempts retain their evidence; "
        "a failure does not become a pass after cleanup.",
        "",
        "See [PROTOCOL.md](PROTOCOL.md) for scope, thresholds and replay commands.",
        "",
    ]
    (evidence / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}))


if __name__ == "__main__":
    main()
