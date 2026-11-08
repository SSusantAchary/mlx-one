"""Evidence reviews cannot hide failed probes or bind to changed results."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/qualification/model_release_report.py"
SPEC = importlib.util.spec_from_file_location("release_report", PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def evidence(root, *, status="PASS_FUNCTIONAL"):
    entry = dict(model_type="sam3_tracker", kind="segmentation", repo="test/sam", revision="a" * 40)
    path = root / "sam3_tracker.json"
    path.write_text(
        json.dumps(
            dict(
                checkpoint=entry["repo"],
                revision=entry["revision"],
                status=status,
                quality_status="MEASURED_REVIEW_REQUIRED",
                interfaces={"python": "PASS"},
            )
        )
    )
    (root / "shared-component-cleanup.json").write_text(
        json.dumps(
            dict(
                model_types=["sam3", "sam3_tracker"],
                cleanup_complete=True,
            )
        )
    )
    review = dict(
        source_result=path.name,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint=entry["repo"],
        revision=entry["revision"],
        status="PASS_SCOPED_REFERENCE",
    )
    (root / "review.json").write_text(json.dumps(review))
    (root / "evidence-reviews.json").write_text(json.dumps({entry["model_type"]: "review.json"}))
    return entry, path


def test_review_requires_immutable_identity_and_keeps_original_measurement(tmp_path):
    entry, path = evidence(tmp_path)
    row = REPORT.summarize([entry], tmp_path)[0]
    assert row["quality_status"] == "PASS_SCOPED_REFERENCE"
    assert row["original_quality_status"] == "MEASURED_REVIEW_REQUIRED"
    assert row["scoped_review_candidate"] and row["cleanup_complete"]
    assert not row["release_qualified"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="immutable result"):
        REPORT.summarize([entry], tmp_path)


def test_positive_quality_review_cannot_override_a_failed_probe(tmp_path):
    entry, _ = evidence(tmp_path, status="FAIL")
    row = REPORT.summarize([entry], tmp_path)[0]
    assert row["status"] == "FAIL" and not row["scoped_review_candidate"]
