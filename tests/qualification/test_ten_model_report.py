"""Evidence compilation cannot replace failed gates with registration success."""

import importlib.util
import json
from pathlib import Path

import pytest


def reporter(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/qualification"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ten_report", scripts / "report_ten_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return scripts, module


def test_failed_profile_and_absent_profiles_remain_unqualified(tmp_path, monkeypatch):
    _, module = reporter(monkeypatch)
    run = tmp_path / "run"
    profile = run / "native"
    profile.mkdir(parents=True)
    (profile / "M01.json").write_text(
        json.dumps(
            {
                "repo": "openai-community/gpt2",
                "revision": "a" * 40,
                "status": "FAIL_NUMERICAL",
                "checks": {"weights": {"status": "PASS"}},
            }
        )
    )
    documents, hardware = module.compile_results(run, tmp_path / "reports")
    assert len(documents) == 10 and hardware == []
    assert all(d["qualification"] == "NOT_QUALIFIED" for d in documents)
    assert documents[0]["profiles"][0]["status"] == "FAIL_NUMERICAL"
    assert documents[1]["profiles"] == []
    with pytest.raises(ValueError, match="output exists"):
        module.compile_results(run, tmp_path / "reports")


def test_schema_rejects_missing_fields_and_invalid_status(monkeypatch):
    scripts, module = reporter(monkeypatch)
    schema = json.loads((scripts / "ten_model_result.schema.json").read_text())
    with pytest.raises(ValueError, match="required"):
        module.validate({}, schema)
    with pytest.raises(ValueError, match="enum"):
        module.validate(
            "SKIPPED", schema["properties"]["profiles"]["items"]["properties"]["status"]
        )


def test_hardware_coverage_never_promotes_other_precisions_batches_or_contexts(monkeypatch):
    _, module = reporter(monkeypatch)
    result = module.coverage(
        [{"checkpoint_id": "M01", "checkpoint": {"context_tokens": 1024}}],
        [
            {
                "id": "M01",
                "source_precision": "stored",
                "weight_dtypes": ["float32"],
                "input_tokens": 16,
                "output_budget": 16,
                "status": "PASS",
                "evidence": "native/M01.json",
            }
        ],
    )
    cells = {
        (r["precision"], r["input_tokens"], r["output_budget"], r["batch_size"]): r
        for r in result["cells"]
    }
    assert result["counts"]["PASS"] == 1
    assert cells["float32", 16, 16, 1]["status"] == "PASS"
    assert cells["float32", 16, 16, 2]["status"] == "NOT_RUN"
    assert cells["affine_q4", 16, 16, 1]["status"] == "NOT_RUN"
    assert cells["float32", 1024, 1, 1]["status"] == "UNSUPPORTED"


def test_supervisor_timeout_overrides_last_running_snapshot_without_overwriting_it(
    tmp_path, monkeypatch
):
    _, module = reporter(monkeypatch)
    run = tmp_path / "run"
    profile = run / "bounded"
    profile.mkdir(parents=True)
    snapshot = {
        "id": "M01",
        "repo": "openai-community/gpt2",
        "revision": "a" * 40,
        "status": "NOT_RUN",
        "checks": {
            "hardware": {"status": "NOT_RUN", "reason": "campaign still running", "rows": []}
        },
    }
    original = json.dumps(snapshot)
    (profile / "M01.json").write_text(original)
    (profile / "summary.json").write_text(
        json.dumps(
            [
                {
                    **snapshot,
                    "status": "BLOCKED_ENVIRONMENT",
                    "supervision": {"status": "timeout"},
                }
            ]
        )
    )
    documents, _ = module.compile_results(run, tmp_path / "reports")
    result = documents[0]["profiles"][0]
    assert result["status"] == "BLOCKED_ENVIRONMENT"
    assert result["checks"]["hardware"]["status"] == "BLOCKED_ENVIRONMENT"
    assert "supervisor timeout" in result["checks"]["hardware"]["reason"]
    assert (profile / "M01.json").read_text() == original


def test_report_can_be_reexported_inside_run_without_ingesting_previous_aggregates(
    tmp_path, monkeypatch
):
    _, module = reporter(monkeypatch)
    run = tmp_path / "run"
    profile = run / "native"
    profile.mkdir(parents=True)
    (profile / "M01.json").write_text(
        json.dumps(
            {
                "repo": "openai-community/gpt2",
                "revision": "a" * 40,
                "status": "FAIL_NUMERICAL",
                "checks": {},
            }
        )
    )
    first, _ = module.compile_results(run, run / "reports-1")
    second, _ = module.compile_results(run, run / "reports-2")
    assert first == second
    assert len(second[0]["profiles"]) == 1
