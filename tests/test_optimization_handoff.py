"""Offline handoff regression tests; all native receipts here are simulations."""
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mlx_one.inference_qualification import write_integrity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualification/run_inference_optimization.py"
spec = importlib.util.spec_from_file_location("optimization_handoff", SCRIPT)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


def test_no_model_stage_starts_without_explicit_local_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "preflight", lambda *a, **k: pytest.fail("started GPU work"))
    with pytest.raises(SystemExit) as result:
        handoff.main(["correctness", "--output", str(tmp_path)])
    assert result.value.code == 2
    paths = tmp_path / "paths.json"
    paths.write_text(json.dumps({"qwen3-small": str(tmp_path / "absent")}))
    with pytest.raises(SystemExit) as result:
        handoff.main(["correctness", "--checkpoints", "qwen3-small",
                      "--checkpoint-paths", str(paths)])
    assert result.value.code == 2


@pytest.mark.parametrize("change", (None, "status", "source", "scope", "test", "skip"))
def test_fused_operator_receipt_requires_current_exact_scope_and_complete_tests(tmp_path, change):
    names = sorted(handoff.fused_test_names([]))
    assert len(names) == 223
    source = {"source_sha256": "simulated"}
    scope = {"checkpoints": [], "hardware": {"chip": "simulated"},
             "attention_backend": "forced_fused_diagnostic"}
    report = {"schema_version": "native-fused-diagnostic-v2", "source": source,
              "scope": scope, "status": "PASS", "force_fused": True,
              "qualification_promoted": False,
              "cases": [{"test": name, "status": "PASS"} for name in names]}
    if change == "status":
        report["cases"][0]["status"] = "HARDWARE_PENDING"
    elif change == "source":
        report["source"] = {"source_sha256": "stale"}
    elif change == "scope":
        report["scope"] = {**scope, "attention_backend": "reference"}
    elif change == "test":
        report["cases"][0]["test"] = "forged_count_only_test"
    suite = ET.Element("testsuite")
    for name in names:
        node = ET.SubElement(suite, "testcase", name=name)
        if change == "skip":
            ET.SubElement(node, "skipped")
    ET.ElementTree(suite).write(tmp_path / "junit.xml")
    path = tmp_path / "fused-results.json"
    handoff.write(path, report)
    (tmp_path / "worker.log").write_text("simulated operator receipts\n")
    handoff.write(tmp_path / "command.json", {"source": source,
        "argv": handoff.fused_command(tmp_path, "python", kernel_only=True)})
    handoff.write(tmp_path / "supervision.json", {"status": "completed",
        "worker_log_sha256": hashlib.sha256((tmp_path / "worker.log").read_bytes()).hexdigest()})
    write_integrity(tmp_path)
    if change:
        with pytest.raises(ValueError):
            handoff.validate_fused_report(path, source=source, scope=scope, paths={})
    else:
        assert handoff.validate_fused_report(path, source=source, scope=scope, paths={}) == report


def test_pending_hardware_measurements_are_not_filled_with_numbers(tmp_path):
    handoff.collect_results(tmp_path)
    result = json.loads((tmp_path / "measurements.json").read_text())
    assert result == {"rows": [], "qualification_promoted": False}
    assert len((tmp_path / "measurements.csv").read_text().splitlines()) == 1


def test_native_exceptions_are_failures_even_before_a_receipt_row_exists():
    from functools import wraps
    from types import SimpleNamespace

    source = SCRIPT.parents[2] / "tests/optimization/test_fused_attention_native.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "guarded")
    namespace = {"wraps": wraps}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    receipt = {"cases": []}
    request = SimpleNamespace(node=SimpleNamespace(nodeid="simulated_native_failure"))

    @namespace["guarded"]
    def fail(*, receipt, request):
        raise ValueError("executed kernel failure")

    with pytest.raises(ValueError, match="executed kernel failure"):
        fail(receipt=receipt, request=request)
    assert receipt["cases"] == [{"test": request.node.nodeid, "status": "FAIL",
        "exception": {"type": "ValueError", "message": "executed kernel failure"}}]


def test_operator_and_checkpoint_commands_run_independently_without_fail_fast(tmp_path):
    operators = handoff.fused_command(tmp_path, "python", kernel_only=True)
    model = handoff.fused_command(tmp_path, "python", checkpoint_only=True)
    assert "-x" not in operators + model
    assert operators[-2:] == ["-k", "not checkpoint"]
    assert model[-2:] == ["-k", "checkpoint"]
    assert handoff.fused_test_names(["qwen3-small"], checkpoint_only=True) == {
        "test_fused_checkpoint_logits_state_and_continuation[qwen3-small]"}


@pytest.mark.parametrize("kind", ("p1", "fused"))
def test_evidence_node_names_match_actual_pytest_parameter_declarations(tmp_path, kind):
    """Collect identical decorators with empty bodies, without importing native MLX."""
    from mlx_one.native_correctness import test_nodes as nodes_for
    from mlx_one.p1_profiles import CHECKPOINTS

    repo = SCRIPT.parents[2]
    source = repo / ("tests/test_p1_checkpoint_parity.py" if kind == "p1" else
                     "tests/optimization/test_fused_attention_native.py")
    tests = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_") and (
                kind == "fused" or node.name == "test_native_p1_serial_dense_paged_full_state"):
            node.body = [ast.Pass()]
            node.decorator_list = [decorator for decorator in node.decorator_list
                                   if not isinstance(decorator, ast.Name)]
            tests.append(ast.unparse(node))
    copy = tmp_path / "test_parameter_contract.py"
    prefix = ("import json, os, pytest\n"
              f"CHECKPOINTS = {list(CHECKPOINTS)!r}\n"
              f"os.environ['MLX_ONE_FUSED_KEYS'] = {json.dumps(list(CHECKPOINTS))!r}\n")
    copy.write_text(prefix + "\n\n".join(tests))
    result = subprocess.run([sys.executable, "-m", "pytest", "--no-cov", "-q",
                             "--collect-only", str(copy)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = {line.split("::")[-1] for line in result.stdout.splitlines() if "::test_" in line}
    expected = (handoff.fused_test_names(list(CHECKPOINTS)) if kind == "fused" else {
        node.split("::")[-1] for checkpoint in CHECKPOINTS for paged in (False, True)
        for node in nodes_for(checkpoint, paged)})
    assert actual == expected
