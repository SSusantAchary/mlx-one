"""Fail-closed wrapper tests; all documents here are synthetic, not native proof."""

import importlib.util
import json
from pathlib import Path

import pytest

from mlx_one.inference_qualification import write_integrity

PATH = Path(__file__).resolve().parents[2] / "scripts/qualification/alpha/run.py"
SPEC = importlib.util.spec_from_file_location("alpha_receipts", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_native_run_requires_explicit_opt_in_and_keeps_independent_attempts(tmp_path):
    for _ in range(2):
        assert runner.main(["task", "--artifact-root", str(tmp_path)]) == 1
    outputs = list(tmp_path.iterdir())
    assert len(outputs) == 2
    for output in outputs:
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "HARDWARE_PENDING"
        assert result["qualification_promoted"] is False
        hashes = json.loads((output / "sha256-manifest.json").read_text())
        assert hashes["result.json"] == runner.digest(output / "result.json")


@pytest.mark.parametrize("status", ["pending", "failed", "passed"])
def test_exit_zero_or_status_pass_cannot_hide_nonpass_semantic_gate(tmp_path, status):
    for name, data in {
        "manifest.json": {"profile_sha256": "a" * 64},
        "task-gates.json": {"status": status, "checks": {"quality": None}},
        "task-runtime.json": {}, "task-evidence.json": {},
    }.items():
        (tmp_path / name).write_text(json.dumps(data))
    write_integrity(tmp_path)
    with pytest.raises(ValueError, match="semantic gates"):
        runner.verify_task_receipt(tmp_path, "a" * 64)
    with pytest.raises(ValueError, match="different profile"):
        runner.verify_task_receipt(tmp_path, "b" * 64)


def test_source_identity_binds_scripts_tests_and_non_python_assets(tmp_path):
    (tmp_path / "src/mlx_one/data").mkdir(parents=True)
    asset = tmp_path / "src/mlx_one/data/compatibility.json"
    asset.write_text("{}")
    first = runner.snapshot(tmp_path)["source_sha256"]
    asset.write_text('{"changed": true}')
    assert runner.snapshot(tmp_path)["source_sha256"] != first
