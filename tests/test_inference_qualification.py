"""Deterministic qualification gates: fixtures never assert real hardware qualification."""

import hashlib
import json
from dataclasses import replace

import pytest

from mlx_one.inference_baseline import CHECKPOINTS, BaselineError, BaselineProfile, run_profile
from mlx_one.inference_qualification import (
    evaluate_gate_d,
    evaluate_gate_e,
    evaluate_gate_f,
    evaluate_quality,
    export_matrix,
    load_reference,
    qualification_profiles,
    qualified_active_decoders,
    validate_integrity,
    workload_envelope,
    write_integrity,
)


def profile(**kwargs):
    return BaselineProfile(*CHECKPOINTS["qwen"], "paged", execution_dtype="float16",
                           cache_type="f16", **kwargs)


def test_profiles_cover_pages_chunks_batches_prefix_ratios_and_soak():
    items = list(qualification_profiles("paged", models=["qwen"], workload="baseline"))
    assert {p.page_tokens for p in items} == {16, 32, 64, 128}
    assert {p.chunk_tokens for p in items} == {512, 1024, 2048, 4096}
    assert {p.slots for p in items} == {1, 2, 4, 8}
    assert all(p.max_scheduled_tokens >= p.chunk_tokens * 2 for p in items)
    shared = list(qualification_profiles("paged", models=["qwen"], workload="multi_turn"))
    assert {p.shared_prefix_fraction for p in shared} == {0, .25, .5, .8}
    assert {p.admission_policy for p in shared} == {"deadline", "cache_aware"}
    soak = list(qualification_profiles("paged", models=["qwen15"], workload="soak"))
    assert {p.duration_seconds for p in soak} == {1800, 3600}
    assert {p.context_tokens for p in soak} == {8192, 126976}


def test_auto_and_missing_quality_limits_cannot_qualify():
    with pytest.raises(BaselineError, match="explicit resolved"):
        BaselineProfile(*CHECKPOINTS["qwen"], "auto")
    parity = {"max_error": .01, "held_out_nll_mean_delta": .001,
              "seeded_rng_isolation": True, "cancelled_request_isolated": True,
              "prefix_isolation_passed": True}
    retrieval = {"oracle_accuracy": 1, "target_accuracy": 1}
    assert evaluate_quality(parity, retrieval, None)["status"] == "diagnostic"
    limits = {"max_logit_error": .1, "max_mean_nll_delta": .01,
              "max_retrieval_accuracy_drop": .1}
    assert evaluate_quality(parity, retrieval, limits)["status"] == "diagnostic"
    from mlx_one.quality_dataset import QUALITY_SCHEMA

    parity.update(quality_kind=QUALITY_SCHEMA, coverage={"eligible": True})
    retrieval["min_oracle_accuracy"] = .8
    assert evaluate_quality(parity, retrieval, limits)["status"] == "passed"
    result = evaluate_quality(parity, {**retrieval, "target_accuracy": .5}, limits)
    assert result["status"] == "failed"
    assert evaluate_gate_d(profile(), {}, [], parity)["status"] == "diagnostic"
    assert evaluate_gate_d(profile(ttft_budget_ms=10, decode_gap_budget_ms=2), {}, [], parity)[
        "status"] == "pending"
    quantized = replace(profile(ttft_budget_ms=10, decode_gap_budget_ms=2),
                        kv_bits=4, cache_type="q4_0")
    assert evaluate_gate_d(quantized, {}, [], parity)["status"] == "diagnostic"


def test_unsupported_long_context_is_rejected_before_starting_worker(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("worker must not be started")
    monkeypatch.setattr("mlx_one.inference_baseline.subprocess.Popen", unexpected)
    item = profile(context_tokens=130944, workload="long")
    outcome = run_profile(item, tmp_path / "run",
                          hardware={"memory_bytes": 1000, "chip": "fixture"},
                          provenance={"source_sha256": "fixture"})
    assert outcome["status"] == "rejected_context"
    assert (tmp_path / "run" / "failure.json").is_file()


def test_service_decoder_count_uses_model_intervals_and_sweep_not_queue_overlap():
    item = profile(ttft_budget_ms=100, decode_gap_budget_ms=50, min_decode_tokens_per_second=10)
    def row(start, end):
        return {"temperature_phase": "warm", "status": "completed", "client_ttft_ms": 10,
                "started": 0, "ended": 100, "model_token_gaps_ms": [10],
                "metrics": {"model_generation_started_at": 0,
                            "model_token_times_ms": [start * 1000, end * 1000],
                            "decode_tokens_per_second": 100}}
    # The middle interval overlaps both peers, but at most two decoders coexist.
    rows = [row(1, 2), row(1.5, 3.5), row(3, 4)]
    assert qualified_active_decoders(rows, item) == 2
    assert qualified_active_decoders(rows, replace(item, min_decode_tokens_per_second=None)) is None
    assert qualified_active_decoders([row(1, 2), row(2, 3)], item) == 1
    rows[0]["metrics"].pop("model_generation_started_at")
    assert qualified_active_decoders(rows[:1], item) == 0


def test_integrity_and_matrix_fail_closed_on_changed_or_incomplete_evidence(tmp_path):
    run = tmp_path / "results" / "fixture" / "checkpoint" / "run"
    run.mkdir(parents=True)
    item = profile()
    files = {"manifest.json": {"schema_version": "native-baseline-v1", "profile": item.to_dict()},
             "summary.json": {"profile": item.to_dict(), "identity": {"hardware": "fixture"}},
             "runtime.json": {}, "gate-d.json": {"status": "passed"}}
    for name, payload in files.items():
        (run / name).write_text(json.dumps(payload))
    (run / "requests.jsonl").write_text("")
    write_integrity(run)
    assert load_reference(run)["profile"]["inference_mode"] == "paged"
    rows = export_matrix(tmp_path / "results", tmp_path / "matrix")
    assert rows[0]["gate_d"] == "pending" and rows[0]["gate_e"] == "pending"
    (run / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="stale"):
        load_reference(run)
    rows = export_matrix(tmp_path / "results", tmp_path / "matrix")
    assert rows[0]["gate_d"] == "pending"
    assert rows[0]["status"] == "stale_or_incomplete_evidence"
    assert (tmp_path / "matrix" / "matrix.csv").exists()


def test_integrity_rejects_artifact_path_traversal(tmp_path):
    path = tmp_path / "outside.json"
    path.write_text("{}")
    root = tmp_path / "run"
    root.mkdir()
    (root / "result-integrity.json").write_text(json.dumps({
        "../outside.json": hashlib.sha256(path.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="artifact path"):
        validate_integrity(root)


def test_matched_direct_gate_requires_direct_calls_and_actual_batch():
    item = profile(slots=2, ttft_budget_ms=100, decode_gap_budget_ms=50,
                   max_queue_wait_ms=100, max_service_gap_ms=50)
    identity = {"hardware": "fixture", "source": "fixture", "weight_precision": "float16",
                "cache_precision": "float16", "mlx_version": "fixture"}
    summary = {"identity": identity, "failed": 0, "starved_requests": 0,
               "observed_prompt_max_tokens": item.context_tokens,
               "fairness": {"status": "passed"},
               "warm_client_ttft_p95_ms": 10, "model_token_gap_p95_ms": 5,
               "aggregate_output_tokens_per_second_by_group": [2, 2, 2]}
    reference = {"profile": {**item.to_dict(), "inference_mode": "reference_dense"},
                 "identity": identity, "summary": {
                     "aggregate_output_tokens_per_second_by_group": [1, 1, 1], "failed": 0}}
    traces = [{"direct_attention_calls": 1, "history_materializations": 0,
               "batch_size": 2, "phase": "decode", "request_ids": ["a", "b"]},
              {"direct_attention_calls": 1, "history_materializations": 0,
               "batch_size": 1, "phase": "decode", "request_ids": ["b"]}]
    result = evaluate_gate_d(item, summary, traces, {"status": "passed"}, reference)
    assert result["status"] == "passed"
    traces[0]["batch_size"] = 1
    result = evaluate_gate_d(item, summary, traces, {"status": "passed"}, reference)
    assert result["status"] == "failed"
    reference["identity"] = {**identity, "cache_precision": "bfloat16"}
    with pytest.raises(ValueError, match="precision"):
        evaluate_gate_d(item, summary, traces, {"status": "passed"}, reference)
    assert set(workload_envelope(item)) >= {"page_tokens", "slots", "output_tokens", "cache_type"}


def test_hybrid_and_soak_gates_require_real_state_and_cleanup():
    assert evaluate_gate_e({"status": "passed"}, [])["status"] == "pending"
    item = profile(workload="soak", duration_seconds=1800,
                   ttft_budget_ms=100, decode_gap_budget_ms=50, leak_budget_bytes=0,
                   max_queue_wait_ms=100, max_service_gap_ms=50)
    summary = {"failed": 0, "soak_initial_active_bytes": 10, "soak_final_active_bytes": 10,
               "soak_elapsed_seconds": 1800, "warm_client_ttft_p95_ms": 10,
               "model_token_gap_p95_ms": 5, "recovery_output_matches": True,
               "prefix_isolation_passed": True, "pressure_recovery_passed": True}
    cleanup = {"active_handles": 0, "reserved_bytes": 0}
    assert evaluate_gate_f(item, summary, cleanup)["status"] == "pending"
    summary.update(memory_reconciliation={"status": "passed"},
        shutdown={"actor_alive": False, "server_alive": False}, fairness={"status": "passed"},
        stress_coverage={"mixed_lengths": True, "shared_prefixes": True, "arrivals": True,
            "cancellation": True, "http_stall_observed": True, "direct_stall_probe": True,
            "real_pressure_cycles": 2})
    assert evaluate_gate_f(item, summary, cleanup)["status"] == "passed"
    assert evaluate_gate_f(item, summary, {**cleanup, "reserved_bytes": 1})["status"] == "failed"
    result = evaluate_gate_f(item, {**summary, "memory_reconciliation": {"status": "pending"}},
                             cleanup)
    assert result["status"] == "pending"


def test_direct_gate_does_not_accept_static_batches_or_cache_size_as_peak_evidence():
    item = profile(slots=2, ttft_budget_ms=100, decode_gap_budget_ms=50,
                   max_queue_wait_ms=100, max_service_gap_ms=50)
    identity = {"hardware": "fixture", "source": "fixture", "weight_precision": "float16",
                "cache_precision": "float16", "mlx_version": "fixture"}
    summary = {"identity": identity, "failed": 0, "fairness": {"status": "passed"},
               "observed_prompt_max_tokens": item.context_tokens,
               "warm_client_ttft_p95_ms": 10, "model_token_gap_p95_ms": 5,
               "aggregate_output_tokens_per_second_by_group": [1, 1, 1], "peak_cache_bytes": 1}
    reference = {"profile": {**item.to_dict(), "inference_mode": "reference_dense"},
                 "identity": identity, "summary": {
                    "aggregate_output_tokens_per_second_by_group": [1, 1, 1],
                    "peak_cache_bytes": 100, "failed": 0}}
    traces = [{"direct_attention_calls": 1, "history_materializations": 0,
               "batch_size": 2, "phase": "decode", "request_ids": ["a", "b"]}] * 2
    gate = evaluate_gate_d(item, summary, traces, {"status": "passed"}, reference)
    assert gate["status"] == "failed"
    assert not gate["checks"]["dynamic_membership"]
    assert not gate["checks"]["measured_advantage"]
