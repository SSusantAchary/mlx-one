"""PR-00 inventory facts must not imply task or hardware qualification."""
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from mlx_one.core.registry import get_registration
from mlx_one.runtime_baseline import registry_inventory


def test_inventory_accounts_for_every_registration_without_promoting_declarations():
    report = registry_inventory()
    assert report["registration_count"] == 27
    rows = {r["model_type"]: r for r in report["registrations"]}
    assert {k for k, r in rows.items() if r["integration_kind"] == "architecture_only"} == {
        "lfm2_audio"}
    assert rows["sam3_tracker_video"]["integration_kind"] == "component_only"
    assert rows["sam3_tracker_video"]["parent_model_type"] == "sam3_video"
    assert rows["qwen3_5"]["task_status"]["vision-language-generation"] == "loader_declared"
    for row in rows.values():
        from mlx_one.engine.runtime_specs import BUILTIN_ADAPTERS

        assert row["shared_service_integrated"] == (row["model_type"] in BUILTIN_ADAPTERS)
        assert row["correctness_validation"] == row["performance_qualification"] == "not_run"
    assert report["qualification_promoted"] is False


def test_inventory_uses_registration_data_including_future_entries():
    entry = replace(get_registration("gpt2"), model_type="future_causal", loader_path=None)
    report = registry_inventory([entry])
    assert report["registration_count"] == 1
    row = report["registrations"][0]
    assert row["model_type"] == "future_causal" and row["integration_kind"] == "task_unavailable"
    assert row["task_status"] == {"text-generation": "unavailable"}
    assert "loader" in row["reason"]


def test_inventory_imports_neither_mlx_nor_model_math_in_a_fresh_process():
    result = subprocess.run([sys.executable, "-c", """
import json,sys
from mlx_one.runtime_baseline import registry_inventory
r=registry_inventory()
assert len(r['registrations']) == 27
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
assert not any(n.startswith('mlx_one.models.language.') and n.endswith('.model')
               for n in sys.modules)
print(json.dumps({'count':r['registration_count']}))
"""], env={**os.environ, "PYTHONPATH": "src"}, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"count": 27}


def test_inventory_cli_persists_checksummed_facts_without_native_preflight(tmp_path, monkeypatch):
    from mlx_one import runtime_baseline as baseline
    from mlx_one.inference_qualification import validate_integrity
    from mlx_one.native_qualification import main

    monkeypatch.setattr(baseline, "checkpoint_inventory", lambda: {
        "qwen3-small": {"status": "missing", "reason": "fixture assets absent"}})
    monkeypatch.setattr("mlx_one.native_qualification.preflight", lambda *args:
                        pytest.fail("inventory must not initialize native execution"))
    assert main(["inventory", "--output", str(tmp_path)]) == 0
    report = next(tmp_path.rglob("inventory.json"))
    validate_integrity(report.parent, required={"inventory.json", "manifest.json", "assets.json"})
    snapshot = json.loads(report.read_text())
    assert snapshot["native_preflight"] == "not_run"
    assert snapshot["source"]["git_revision"]
    assert snapshot["dependencies"]["python"]
    assert snapshot["registry"]["registration_count"] == 27


def test_checkpoint_inventory_is_offline_and_checksums_exact_assets(tmp_path, monkeypatch):
    import hashlib

    from mlx_one import runtime_baseline as baseline
    from mlx_one.p1_profiles import CHECKPOINTS

    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    (tmp_path / "model.safetensors").write_bytes(b"fixture header only")
    (tmp_path / "tokenizer.json").write_text("{}")
    calls = []
    def resolve(model, *, revision, local_files_only):
        calls.append((model, revision, local_files_only))
        if model != CHECKPOINTS["qwen3-small"].model_id:
            raise FileNotFoundError("offline fixture miss")
        return str(tmp_path)
    monkeypatch.setattr("huggingface_hub.snapshot_download", resolve)
    report = baseline.checkpoint_inventory()
    assert len(report) == len(calls) == 6 and all(c[-1] for c in calls)
    small = report["qwen3-small"]
    assert small["files"]["tokenizer.json"] == hashlib.sha256(b"{}").hexdigest()
    assert small["execution_validated"] is False
    assert report["qwen3-large"]["status"] == "missing"


def test_missing_assets_never_start_a_model_worker(tmp_path, monkeypatch):
    from mlx_one import native_qualification as driver
    from mlx_one import runtime_baseline as baseline

    monkeypatch.setattr(baseline, "checkpoint_inventory", lambda: {
        "qwen3-small": {"status": "missing", "reason": "offline miss"}})
    monkeypatch.setattr(driver, "supervised_command", lambda *args, **kwargs:
                        pytest.fail("missing assets must not start a model worker"))
    assert baseline.run_baseline(tmp_path, native=True) == 1
    report = json.loads((tmp_path / "diagnostic.json").read_text())
    assert report["status"] == "NOT_RUN" and "assets" in report["reason"]


def test_failed_preflight_prevents_model_execution(tmp_path, monkeypatch):
    from mlx_one import native_qualification as driver
    from mlx_one import runtime_baseline as baseline

    monkeypatch.setattr(baseline, "checkpoint_inventory", lambda: {
        "qwen3-small": {"status": "local", "files": {}}})
    def failed(*args, **kwargs):
        raise RuntimeError("RSS unavailable")
    monkeypatch.setattr(driver, "preflight", failed)
    monkeypatch.setattr(driver, "supervised_command", lambda *args, **kwargs:
                        pytest.fail("preflight failure must stop execution"))
    assert baseline.run_baseline(tmp_path, native=True) == 1
    assert json.loads((tmp_path / "diagnostic.json").read_text())["status"] == "NOT_RUN"


@pytest.mark.parametrize("configured,expected", [(None, 24 * 1024**3), (1024**3, 1024**3)])
def test_configured_ceiling_bounds_preflight(tmp_path, monkeypatch, configured, expected):
    from mlx_one import native_qualification as driver

    monkeypatch.setattr(driver.subprocess, "check_output", lambda args, **kwargs:
                        "Apple M4" if "machdep.cpu.brand_string" in args else str(32 * 1024**3))
    seen = []
    def smoke(command, output, *, ceiling):
        seen.append(ceiling)
        output.mkdir()
        data = {"device": "Device(gpu, 0)", "metal": True, "result": 1240,
                "active_bytes": 1, "cached_bytes": 0, "peak_bytes": 1,
                "device_info": {"max_recommended_working_set_size": 20 * 1024**3}}
        (output / "worker.log").write_text(json.dumps(data))
        return {"status": "completed", "rss_samples": [{"rss_bytes": 100}]}
    monkeypatch.setattr(driver, "supervised_command", smoke)
    report = driver.preflight(tmp_path / "preflight", configured_ceiling=configured)
    assert seen == [expected]
    assert report["process_ceiling_bytes"] == min(expected, 20 * 1024**3)


@pytest.mark.parametrize("configured,expected", [(None, 18), (15, 15), (25, 18)])
def test_native_worker_ceiling_cannot_exceed_preflight_recommendation(configured, expected):
    from mlx_one.runtime_baseline import effective_ceiling

    assert effective_ceiling(18, configured) == expected
    with pytest.raises(ValueError):
        effective_ceiling(18, 0)


def test_reproduced_failure_is_not_qualification():
    from mlx_one.runtime_baseline import diagnostic_outcome

    failure = {"status": "failed", "failure_reproduced": True, "cleanup_passed": True}
    report = diagnostic_outcome({"status": "completed"}, failure)
    assert report["readiness"] == "reproduced" and report["qualification"] == "failed"
    assert report["qualification_promoted"] is False
    report = diagnostic_outcome({"status": "memory_watchdog_terminated"}, failure)
    assert report["status"] == "FAIL" and report["readiness"] != "reproduced"


def test_missing_worker_results_fail_instead_of_using_old_evidence():
    from mlx_one.runtime_baseline import diagnostic_outcome

    assert diagnostic_outcome({"status": "completed"}, None)["status"] == "FAIL"


@pytest.mark.parametrize("change", ["source", "revision", "dtype", "mode", "asset", "path"])
def test_prefix_manifest_rejects_substitution(tmp_path, monkeypatch, change):
    from mlx_one import runtime_baseline as baseline
    from mlx_one.p1_profiles import CHECKPOINTS

    monkeypatch.setattr(baseline, "source_provenance", lambda root: {"source_sha256": "source"})
    (tmp_path / "config.json").write_text("{}")
    checkpoint = CHECKPOINTS["qwen3-small"]
    manifest = {"source": {"source_sha256": "source"},
        "configuration": {"execution_dtype": "float16", "key_type": "f16", "value_type": "f16",
                          "mode": "experimental_dense_batch", "forced_steps": 32},
        "asset": {"model_id": checkpoint.model_id, "revision": checkpoint.revision,
            "path": str(tmp_path),
            "files": {"config.json": baseline.file_digest(tmp_path / "config.json")}}}
    baseline.validate_prefix_manifest(manifest)
    if change == "source":
        manifest["source"]["source_sha256"] = "different"
    elif change == "revision":
        manifest["asset"]["revision"] = "b" * 40
    elif change in {"dtype", "mode"}:
        manifest["configuration"]["execution_dtype" if change == "dtype" else "mode"] = "changed"
    elif change == "asset":
        (tmp_path / "config.json").write_text("changed")
    else:
        manifest["asset"]["files"] = {"../config.json": "wrong"}
    with pytest.raises(ValueError):
        baseline.validate_prefix_manifest(manifest)
