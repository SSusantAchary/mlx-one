"""Qualification cannot be inferred from smoke, mismatched fixtures or missing metrics."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

RUNNER = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/audio_qualification.py"))


def artifacts(tmp_path: Path, count: int = 100):
    predictions = [
        {
            "id": str(index),
            "anchor": index < 16,
            "reference": "one two three",
            "prediction": "one two three",
        }
        for index in range(count)
    ]
    common = {
        "model": "model",
        "revision": "pin",
        "manifest_sha256": "checksum",
        "predictions": predictions,
        "peak_process_bytes": 1024,
        "cleanup_growth_bytes": 0,
    }
    reference = tmp_path / "reference.json"
    native = tmp_path / "native.json"
    reference.write_text(json.dumps({"backend": "reference", **common}))
    native.write_text(json.dumps({"backend": "native", **common}))
    return SimpleNamespace(reference=reference, native=native)


def test_complete_transcript_fixture_matrix_passes_only_scoped_gates(tmp_path) -> None:
    result = RUNNER["compare"](artifacts(tmp_path))
    assert result["status"] == "PASS"
    assert "timestamps pending" in result["scope"]


def test_smoke_counts_do_not_qualify(tmp_path) -> None:
    result = RUNNER["compare"](artifacts(tmp_path, count=16))
    assert result["status"] == "FAIL"
    assert result["gates"]["100_clips"] is False


@pytest.mark.parametrize("field", ["model", "revision", "manifest_sha256"])
def test_different_reference_inputs_are_rejected(tmp_path, field) -> None:
    args = artifacts(tmp_path)
    payload = json.loads(args.native.read_text())
    payload[field] = "different"
    args.native.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inputs differ"):
        RUNNER["compare"](args)


def test_punctuation_anchor_mismatch_cannot_be_normalized_away(tmp_path) -> None:
    args = artifacts(tmp_path)
    payload = json.loads(args.native.read_text())
    payload["predictions"][0]["prediction"] = "one two three."
    args.native.write_text(json.dumps(payload))
    result = RUNNER["compare"](args)
    assert result["status"] == "FAIL"
    assert result["anchor_mismatches"] == ["0"]
    assert result["gates"]["wer_degradation"] is True


def test_missing_cleanup_measurement_cannot_qualify(tmp_path) -> None:
    args = artifacts(tmp_path)
    payload = json.loads(args.native.read_text())
    del payload["cleanup_growth_bytes"]
    args.native.write_text(json.dumps(payload))
    assert RUNNER["compare"](args)["gates"]["cleanup_growth"] is False


def test_corpus_error_is_weighted_without_changing_legacy_macro_profiles() -> None:
    records = [
        {"reference": "one", "prediction": "wrong"},
        {"reference": "one two three", "prediction": "one two three"},
    ]
    assert RUNNER["corpus_error"](records, "prediction") == pytest.approx(0.25)
