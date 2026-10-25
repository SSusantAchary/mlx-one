"""Benchmark evidence cannot silently invent metrics or change unrelated README content."""

import copy
import json
from pathlib import Path

import pytest

from benchmarks.audio.readme_generator import END, START, render, update_readme
from benchmarks.audio.result_schema import METRICS, PROTOCOL, SCHEMA, validate


def record():
    # Validator unit fixture only; never persisted as benchmark evidence.
    row = dict.fromkeys(METRICS)
    row.update(
        processing_seconds=2,
        audio_seconds=10,
        real_time_factor=0.2,
        audio_seconds_per_second=5,
        peak_mlx_gib=1,
        peak_rss_gib=2,
    )
    return {
        "schema": SCHEMA,
        "status": "MEASURED",
        "name": "fixture",
        "checkpoint": "org/test",
        "revision": "a" * 40,
        "precision": "FP32",
        "task": "transcribe",
        "workload": "ASR-S1",
        "protocol": PROTOCOL,
        "runs": [dict(row) for _ in range(5)],
        "metrics": dict(row),
        "hardware": {"chip": "test-only", "memory_bytes": 32 * 1024**3},
        "software": {"mlx": "test"},
        "backend": "native-mlx-metal",
        "measurement_type": "direct-model-prepared-pcm",
        "implementation_sha256": "b" * 64,
        "checkpoint_assets": {"config.json": "c" * 64},
        "input": {"audio_seconds": 10, "audio_sha256": "d" * 64},
    }


def test_pending_catalog_is_all_null_and_never_published_as_measured():
    publication = json.loads(Path("docs/benchmarks/audio/publication.json").read_text())
    for path in publication["records"]:
        result = json.loads(Path(path).read_text())
        validate(result)
        if result["status"] == "MEASURED":
            continue
        bad = copy.deepcopy(result)
        bad["metrics"]["processing_seconds"] = 1
        with pytest.raises(ValueError, match="invented"):
            validate(bad)


@pytest.mark.parametrize("mutation", ["rtf", "median", "nan", "revision", "hardware", "count"])
def test_validator_rejects_invalid_provenance_or_measurements(mutation):
    value = record()
    validate(value)
    if mutation == "rtf":
        for row in value["runs"]:
            row["real_time_factor"] = 1
        value["metrics"]["real_time_factor"] = 1
    elif mutation == "median":
        value["metrics"]["processing_seconds"] = 4
    elif mutation == "nan":
        value["runs"][0]["processing_seconds"] = float("nan")
    elif mutation == "revision":
        value["revision"] = "main"
    elif mutation == "hardware":
        value["hardware"] = {}
    else:
        value["runs"] = value["runs"][:4]
    with pytest.raises(ValueError):
        validate(value)


def test_generator_only_changes_marked_region(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("Original intro\n" + START + "\nold\n" + END + "\nOriginal images\n")
    update_readme(path, START + "\nnew\n" + END + "\n")
    assert path.read_text() == "Original intro\n" + START + "\nnew\n" + END + "\nOriginal images\n"


def test_generator_rejects_precision_mismatch():
    value = record()
    catalog = {
        "entries": [
            {
                "name": "fixture",
                "checkpoint": "org/test",
                "revision": "a" * 40,
                "task": "transcribe",
                "workload": "ASR-S1",
                "precision": "Q4",
            }
        ]
    }
    with pytest.raises(ValueError, match="precision"):
        render(catalog, [value])


def test_generator_keeps_hardware_cohorts_separate():
    first = record()
    second = copy.deepcopy(first)
    second["checkpoint"] = "org/other"
    second["hardware"]["chip"] = "different chip"
    catalog = {
        "entries": [
            {k: r[k] for k in ("name", "checkpoint", "revision", "task", "workload", "precision")}
            for r in (first, second)
        ]
    }
    with pytest.raises(ValueError, match="incompatible"):
        render(catalog, [first, second])


def test_model_language_hints_do_not_change_fixture_identity():
    first = record()
    second = copy.deepcopy(first)
    second["checkpoint"] = "org/other"
    first["input"]["language"] = second["input"]["language"] = "English"
    first["decoding"] = {"language_hint": "en"}
    second["decoding"] = {"language_hint": "English"}
    catalog = {
        "entries": [
            {k: r[k] for k in ("name", "checkpoint", "revision", "task", "workload", "precision")}
            for r in (first, second)
        ]
    }
    assert render(catalog, [first, second]).count("**5.000**") == 2
