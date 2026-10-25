"""Publication rejects incomplete, tampered or incompatible corpus measurements."""

import copy
import json

import pytest

from benchmarks.ocr import corpus_report
from benchmarks.ocr.corpus import accuracy, sha, summarize

pytest.importorskip("rapidfuzz", reason="install the ocr-bench extra for corpus scoring")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_report, "EVIDENCE", tmp_path)
    fixtures = [{"id": str(i), "sha256": "a" * 64, "reference": "Hello world"} for i in range(100)]
    manifest = {"fixtures": fixtures}
    (tmp_path / "sample.json").write_text(json.dumps(manifest))
    entry = {"id": "O04", "checkpoint": "test/fixture", "revision": "b" * 40, "precision": "FP16"}
    samples = [
        {
            "id": f["id"],
            "image_sha256": f["sha256"],
            "status": "OK",
            "latency_seconds": 2.0,
            "transcript": "Hello world",
            "accuracy": accuracy(f["reference"], "Hello world"),
            "output_tokens": 2,
            "peak_mlx_memory_gib": 0.5,
        }
        for f in fixtures
    ]
    record = {
        **entry,
        "schema_version": "mlx-one-ocr-omni100-v1",
        "status": "MEASURED",
        "dataset": corpus_report.DATASET,
        "dataset_revision": corpus_report.REVISION,
        "sample_sha256": sha(tmp_path / "sample.json"),
        "hardware": {"chip": "TEST ONLY", "memory_bytes": 2**30},
        "software": {"mlx": "test"},
        "measurement_type": "direct-model",
        "implementation_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "checkpoint_files": [{"path": "model.safetensors", "sha256": "e" * 64, "bytes": 10}],
        "precision": {"mode": "floating", "dtypes": ["float16"]},
        "protocol": {
            "batch_size": 1,
            "warmups": 1,
            "warmup_max_tokens": 16,
            "repetitions_per_image": 1,
            "images": 100,
            "max_tokens": 1023,
            "temperature": 0,
            "backend": "native MLX/Metal",
        },
        "samples": samples,
        "metrics": summarize(samples),
    }
    return record, entry, manifest


def test_corpus_validator_checks_all_images_and_scores(evidence):
    record, entry, manifest = evidence
    corpus_report.validate(record, entry, manifest)
    missing = copy.deepcopy(record)
    missing["samples"].pop()
    with pytest.raises(ValueError, match="all 100"):
        corpus_report.validate(missing, entry, manifest)
    altered = copy.deepcopy(record)
    altered["samples"][0]["transcript"] = "wrong prediction"
    with pytest.raises(ValueError, match="accuracy"):
        corpus_report.validate(altered, entry, manifest)


@pytest.mark.parametrize("change", ["mock", "hardware", "precision", "protocol", "summary"])
def test_corpus_publication_rejects_invented_context(evidence, change):
    record, entry, manifest = evidence
    if change == "mock":
        record["mock"] = True
    elif change == "hardware":
        record["hardware"] = {}
    elif change == "precision":
        record["precision"]["dtypes"] = ["float32"]
    elif change == "protocol":
        record["protocol"]["max_tokens"] = 2048
    else:
        record["metrics"]["completed_pages_per_second"] = 999
    with pytest.raises(ValueError):
        corpus_report.validate(record, entry, manifest)


def test_failed_images_count_as_empty_predictions():
    samples = [
        {
            "status": "OK",
            "latency_seconds": 1,
            "accuracy": accuracy("abcdef", "abcdef"),
            "peak_mlx_memory_gib": 1,
            "output_tokens": 3,
        },
        {"status": "FAILED", "latency_seconds": 3, "accuracy": accuracy("abc", "")},
    ]
    metrics = summarize(samples)
    assert metrics["completed_pages_per_second"] == 0.25
    assert metrics["corpus_cer"] == 3 / 9
    assert metrics["failed"] == 1
    assert metrics["actual_output_tokens"] == 3


def test_nfkc_scoring_preserves_case_and_punctuation():
    assert accuracy("Ａ  cat\n", "A cat")["character_edits"] == 0
    assert accuracy("A cat.", "a cat")["character_edits"] == 2


def test_handwriting_has_its_own_count_budget_and_dataset(evidence, tmp_path, monkeypatch):
    from benchmarks.ocr import handwriting

    record, entry, manifest = evidence
    monkeypatch.setattr(handwriting, "HAND_EVIDENCE", tmp_path)
    manifest["fixtures"] = manifest["fixtures"][:10]
    (tmp_path / "sample.json").write_text(json.dumps(manifest))
    record.update(
        schema_version="mlx-one-ocr-hand10-v1",
        dataset=handwriting.DATASET,
        dataset_revision=handwriting.REVISION,
        sample_sha256=sha(tmp_path / "sample.json"),
    )
    record["samples"] = record["samples"][:10]
    record["metrics"] = summarize(record["samples"])
    record["protocol"].update(images=10, max_tokens=128)
    corpus_report.validate(record, entry, manifest, hand=True)
    with pytest.raises(ValueError, match="schema"):
        corpus_report.validate(record, entry, manifest)
