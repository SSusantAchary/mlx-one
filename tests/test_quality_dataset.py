"""Held-out scoring and evidence eligibility without a native backend."""
import hashlib
import json

import numpy as np
import pytest

from mlx_one.quality_dataset import (
    DEPTH_BANDS,
    QUALITY_SCHEMA,
    SCORER_VERSION,
    load_quality_manifest,
    paired_logits_metrics,
    quality_coverage,
    retrieval_score,
    summarize_quality_documents,
)


def manifest(tmp_path):
    data = {"schema_version": QUALITY_SCHEMA, "dataset_id": "fixture-v1", "split": "test",
            "provenance": {"source": "CPU test fixture"}, "license": "CC0",
            "settings": {"scorer": SCORER_VERSION, "context_lengths": [2048],
                         "min_oracle_retrieval_accuracy": .9}}
    for kind, rows in (("documents", [{"id": "a", "text": "Held out text."}]),
                       ("retrieval", [{"id": "a", "context_tokens": 2048,
                           "depth_band": "start", "context": "Fact A", "question": "Fact?",
                           "accepted_answers": ["A"]}])):
        path = tmp_path / f"{kind}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        data[kind] = {"path": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    return path


def test_manifest_requires_offline_integrity_and_held_out_provenance(tmp_path):
    path = manifest(tmp_path)
    loaded = load_quality_manifest(path)
    assert loaded["identity"]["split"] == "test"
    (tmp_path / "documents.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_quality_manifest(path)
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data["split"] = "train"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="held-out"):
        load_quality_manifest(path)


def test_paired_nll_is_shift_invariant_and_token_weighted():
    logits = np.array([[2., 0.], [0., 2.]])
    row = paired_logits_metrics(logits, logits + 1000, (0, 1))
    assert row["oracle_nll_sum"] == pytest.approx(2 * np.log1p(np.exp(-2)))
    assert row["oracle_nll_sum"] == pytest.approx(row["target_nll_sum"])
    other = paired_logits_metrics([[0., 0.]], [[0., 2.]], (0,))
    result = summarize_quality_documents([row, other])
    assert result["scored_tokens"] == 3
    assert result["held_out_nll_mean_delta"] == pytest.approx(
        (other["target_nll_sum"] - other["oracle_nll_sum"]) / 3)
    with pytest.raises(ValueError, match="mapping"):
        paired_logits_metrics(logits, logits, (0,))
    with pytest.raises(ValueError):
        paired_logits_metrics([[np.nan]], [[0.]], (0,))


def test_retrieval_exact_normalization_and_coverage_are_required():
    assert retrieval_score("  ＡＮＳＷＥＲ\n ONE ", ["answer one"])
    assert not retrieval_score("the answer one is somewhere here", ["answer one"])
    documents = {"document_count": 100, "scored_tokens": 100_000}
    rows = [{"depth_band": band, "context_tokens": 2048, "rendered_tokens": 2000}
            for band in DEPTH_BANDS for _ in range(20)]
    assert quality_coverage(documents, {"fixtures": rows}, 2048)["eligible"]
    assert not quality_coverage(documents, {"fixtures": rows[:-1]}, 2048)["eligible"]
    assert not quality_coverage(documents, {"fixtures": rows}, 8192)["eligible"]
