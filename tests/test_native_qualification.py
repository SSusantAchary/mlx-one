"""Deterministic qualification orchestration; these fixtures are not native evidence."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from test_dense_batching import native as native

from mlx_one import native_qualification as driver
from mlx_one.inference_qualification import write_integrity


def test_full_legal_matrix_and_explicit_smol_rejections():
    manifest = driver.command_manifest()
    identities = {t["id"] for t in manifest["tasks"]}
    assert manifest["optimized_configurations"] == 2592
    assert len(manifest["rejected_literal_contexts"]) == 6
    for task in manifest["tasks"]:
        case = driver.NativeCase(**task["case"])
        assert case.prompt_tokens + case.output_tokens <= driver.CHECKPOINTS[
            case.checkpoint].context_tokens
        if case.prompt_class == "maximum_legal_8k":
            assert case.prompt_tokens + case.output_tokens == 8192
        assert all(dep in identities for dep in task["dependencies"])
    assert manifest["qualification_promoted"] is False


def test_comparison_traffic_is_independent_of_backend_and_slots():
    case = next(c for c in driver.cases(["qwen3-small"]) if c.slots == 4)
    assert driver.comparison_traffic(case) == driver.comparison_traffic(case.reference())
    assert driver.comparison_traffic(case) == driver.comparison_traffic(case.optimized_b1())
    assert len(driver.comparison_traffic(case)["requests"]) == 12


def test_failed_dependency_is_retained_and_blocks_optimized_run(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "source_provenance", lambda root: {"source_sha256": "source"})
    monkeypatch.setattr(driver, "export_matrix", lambda *args: None)
    manifest = driver.command_manifest(["qwen3-small"])
    target = next(t for t in manifest["tasks"] if t["case"]["mode"] != "reference_dense")
    ids = set(target["dependencies"]) | {target["id"]}
    manifest["tasks"] = [t for t in manifest["tasks"] if t["id"] in ids]
    calls = []
    def failed(item, path, **kwargs):
        calls.append(item)
        return {"status": "worker_failed"}
    outcomes = driver.execute_manifest(manifest, tmp_path, {}, runner=failed)
    assert outcomes[target["id"]]["status"] == "BLOCKED"
    assert all(p.inference_mode == "reference_dense" for p in calls)
    assert json.loads((tmp_path / "outcomes.json").read_text()) == outcomes


@pytest.mark.parametrize("status", ["memory_watchdog_unavailable", "memory_watchdog_terminated"])
def test_watchdog_failure_stops_all_native_expansion(tmp_path, monkeypatch, status):
    monkeypatch.setattr(driver, "source_provenance", lambda root: {"source_sha256": "source"})
    manifest = driver.command_manifest(["qwen3-small"])
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return {"status": status}
    with pytest.raises(RuntimeError, match="supervision failed"):
        driver.execute_manifest(manifest, tmp_path, {}, runner=failed)
    assert len(calls) == 1 and (tmp_path / "outcomes.json").exists()


def test_changed_source_stops_cohort(tmp_path, monkeypatch):
    count = iter(["before", "after"])
    monkeypatch.setattr(driver, "source_provenance", lambda root: {"source_sha256": next(count)})
    with pytest.raises(RuntimeError, match="source changed"):
        driver.execute_manifest(driver.command_manifest(["qwen3-small"]), tmp_path, {})


def test_unmeasured_checkpoint_rows_stay_not_run():
    rows = driver.correctness_matrix({"qwen3-small:False": {"status": "FAIL", "path": "failure"}},
                                     {"source_sha256": "source"})
    assert len(rows) == 12
    assert rows[0]["status"] == "FAIL" and rows[1]["status"] == "BLOCKED"
    assert all(r["status"] == "NOT_RUN" for r in rows[2:])
    assert not any(r["production_qualified"] for r in rows)


def test_correctness_cannot_accept_a_bare_pass(tmp_path):
    path = tmp_path / "correctness.json"
    driver.write(path, {"qwen3-small:False": {"status": "PASS"}})
    write_integrity(tmp_path)
    with pytest.raises(ValueError, match="schema"):
        driver.validate_correctness(path, ["qwen3-small"], source={"source_sha256": "s",
            "native_test_sha256": "t"}, hardware={})


@pytest.mark.parametrize("change", ["source", "hardware", "test", "log", "skip"])
def test_correctness_rejects_stale_or_incomplete_native_proof(tmp_path, change):
    import hashlib

    source = {"source_sha256": "source", "native_test_sha256": "test"}
    hardware = {"chip": "test-only"}
    outcomes = {}
    for paged in (False, True):
        directory = tmp_path / str(paged)
        directory.mkdir()
        log = b"6 passed\n"
        if change == "skip":
            log += b"1 skipped\n"
        (directory / "worker.log").write_bytes(log)
        driver.write(directory / "command.json", {})
        driver.write(directory / "supervision.json", {"status": "completed",
            "worker_log_sha256": hashlib.sha256(log).hexdigest()})
        write_integrity(directory)
        if change == "log":
            (directory / "worker.log").write_text("altered")
        outcomes[f"qwen3-small:{paged}"] = {"status": "PASS", "path": str(directory)}
    proof = {"schema_version": driver.CORRECTNESS_VERSION, "source": source,
             "hardware": hardware, "outcomes": outcomes}
    expected = deepcopy(source)
    if change == "source":
        expected["source_sha256"] = "different"
    if change == "test":
        expected["native_test_sha256"] = "different"
    path = tmp_path / "correctness.json"
    driver.write(path, proof)
    write_integrity(tmp_path)
    with pytest.raises(ValueError):
        driver.validate_correctness(path, ["qwen3-small"], source=expected,
                                     hardware={} if change == "hardware" else hardware)


def test_controlled_decode_excludes_prepared_history_and_first_token():
    from mlx_one.p1_decode import measure_decode

    clock = iter([100, 102])
    steps = []
    result = measure_decode(lambda i: steps.append(i) or 4, 31, clock=lambda: next(clock))
    assert steps == list(range(31))
    assert result["interval_tokens"] == 124 and result["tokens_per_second"] == 62
    with pytest.raises(ValueError):
        measure_decode(lambda i: 1, 0)


def test_missing_gpu_output_stays_blocked_with_durable_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.subprocess, "check_output", lambda args, **kwargs:
                        "Apple M4" if "machdep.cpu.brand_string" in args else str(32 * 1024**3))
    def failed(command, output, **kwargs):
        output.mkdir()
        (output / "worker.log").write_text("Metal could not initialize\n")
        return {"status": "worker_failed", "rss_samples": []}
    monkeypatch.setattr(driver, "supervised_command", failed)
    with pytest.raises(RuntimeError, match="preflight failed"):
        driver.preflight(tmp_path / "preflight")
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert report["status"] == "BLOCKED" and not report["gpu_smoke"]


def test_cli_exposes_comparison_and_queue_budgets(capsys):
    from mlx_one.p1_benchmark import main

    assert main(["profiles", "--prompt-tokens", "128", "--output-tokens", "32",
        "--slots", "4", "--reference-run", "/tmp/explicit-reference", "--queue-size", "32",
        "--max-scheduled-tokens", "4096", "--max-prefill-tokens", "2048"]) == 0
    item = json.loads(capsys.readouterr().out)["profiles"][0]
    assert item["reference_run"] == "/tmp/explicit-reference" and item["queue_size"] == 32
    assert item["max_scheduled_tokens"] == 4096 and item["max_prefill_tokens"] == 2048


def gate_fixture():
    from dataclasses import asdict

    from mlx_one.p1_decode import DECODE_VERSION
    from mlx_one.p1_profiles import profile

    item = profile("qwen3-small", mode="experimental_dense_batch", slots=1,
        ttft_budget_ms=2000, decode_gap_budget_ms=100,
        max_queue_wait_ms=5000, max_service_gap_ms=250)
    identity = {key: "fixture" for key in ("hardware", "source", "weight_precision",
        "cache_precision", "mlx_version", "quantization_digest", "floating_policy")}
    summary = {"identity": identity, "failed": 0, "cleanup_passed": True,
        "starved_requests": 0, "max_admission_wait_ms": 10,
        "max_runnable_service_gap_ms": 10, "model_token_gap_p95_ms": 10,
        "warm_client_ttft_p95_ms": 100,
        "controlled_decode": {"schema_version": DECODE_VERSION, "input_identity": "tokens",
                              "rates": [100] * 5},
        "comparison_inputs": {"traffic_identity": "traffic", "rendered_tokens_sha256": "tokens",
                              "generated_tokens_sha256": "output"}}
    serial = {"profile": {**asdict(item), "inference_mode": "reference_dense"},
              "identity": identity, "native_execution": True, "summary": deepcopy(summary)}
    return item, summary, serial


@pytest.mark.parametrize("field", ["traffic_identity", "max_prefill_tokens", "queue_size",
                                  "max_scheduled_tokens", "output_stall_timeout_s"])
def test_p1_gate_rejects_changed_workload_or_execution_budget(field):
    from mlx_one.p1_qualification import evaluate_p1

    item, summary, serial = gate_fixture()
    serial["profile"][field] = "changed"
    with pytest.raises(ValueError, match="reference mismatch"):
        evaluate_p1(item, summary, [], {"status": "passed"}, serial=serial)


def test_service_throughput_cannot_substitute_for_controlled_decode():
    from mlx_one.p1_qualification import evaluate_p1

    item, summary, serial = gate_fixture()
    del summary["controlled_decode"]
    summary["aggregate_decode_tokens_per_second_by_group"] = [100000] * 5
    result = evaluate_p1(item, summary, [], {"status": "passed"}, serial=serial)
    assert result["status"] == "pending" and "prepared-history" in result["reason"]


def test_changed_rendered_or_generated_tokens_fail_paired_comparison():
    from mlx_one.p1_qualification import evaluate_p1

    item, summary, serial = gate_fixture()
    serial["summary"]["comparison_inputs"]["generated_tokens_sha256"] = "different"
    result = evaluate_p1(item, summary, [{"phase": "decode", "batch_size": 1}],
                         {"status": "passed"}, serial=serial)
    assert not result["checks"]["matched_serial"] and result["status"] == "failed"


@pytest.mark.parametrize("policy", [None, "native-float-execution-v1"])
def test_older_precision_policy_cannot_be_inferred_from_envelope_defaults(policy):
    from mlx_one.engine.envelope import QualificationEnvelope, profile_envelope

    item, _, _ = gate_fixture()
    envelope = profile_envelope(item).to_dict()
    if policy is None:
        del envelope["execution_policy_version"]
    else:
        envelope["execution_policy_version"] = policy
    with pytest.raises(ValueError):
        QualificationEnvelope.from_mapping(envelope)


def test_smol_maximum_legal_8k_prompt_requires_prefix_benefit():
    from dataclasses import asdict

    from mlx_one.p1_qualification import evaluate_p1

    item, summary, serial = gate_fixture()
    item = replace(item, model_id=driver.CHECKPOINTS["smol-small"].model_id,
        workload="prefix", context_tokens=8064, output_tokens=128)
    serial["profile"] = {**asdict(item), "inference_mode": "reference_dense"}
    result = evaluate_p1(item, summary, [{"phase": "decode", "batch_size": 1}],
                         {"status": "passed"}, serial=serial)
    assert result["checks"]["8k_prefix"] is False


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_controlled_decode_fixture_has_matched_inputs_and_cleanup(native, monkeypatch, slots):
    import mlx.core as mx

    from mlx_one.inference_baseline import BaselineProfile
    from mlx_one.p1_decode import native_decode

    mx.synchronize = lambda: None
    monkeypatch.setattr("mlx_one.inference_baseline_worker.build_messages",
                        lambda bundle, target, suffix: ([], "abc" + suffix))
    item = BaselineProfile(native.model_id, native.revision, "experimental_dense_batch",
        context_tokens=128, output_tokens=4, slots=slots, repeats=5,
        cache_type="f16", execution_dtype="float16")
    result = native_decode(native, item)
    oracle = native_decode(native, replace(item, inference_mode="reference_dense", slots=1))
    assert result["input_identity"] == oracle["input_identity"]
    assert len(result["rates"]) == 5 and len(result["samples"]) == 6
    assert all(s["interval_tokens"] == 12 for s in result["samples"])
    assert max(c["batch_size"] for c in result["batch_calls"]) == slots
    assert result["cleanup"]["active_handles"] == result["cleanup"]["reserved_bytes"] == 0
