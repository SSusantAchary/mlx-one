import copy
import json
import subprocess
import sys

import pytest

from qualification.p01_math import (
    abba,
    await_prerequisites,
    canonical_digest,
    compare,
    experiment_inventory,
    summarize,
    validate_pair,
    validate_worker_result,
    worker_status,
)


def identity():
    return {"checkpoint": "qwen3", "revision": "a" * 40, "assets_sha256": "b" * 64,
            "tokenizer_identity": "tokens", "prompt_ids": [1, 2], "output_fixture": [3, 4],
            "execution_dtype": "float16", "kv_dtype": "float16", "source_sha256": "c" * 64,
            "environment": {"mlx": "0.32.2"}, "context_limit": 8192, "seed": 0,
            "prefix_reuse": False, "batch": 1, "profile": False,
            "measurement_kind": "python_engine_model_token_intervals"}


def test_import_does_not_initialize_mlx():
    result = subprocess.run([sys.executable, "-c", "import sys; import qualification.p01_math; "
                             "assert 'mlx.core' not in sys.modules"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_balanced_independent_order():
    assert abba(2) == ["reference", "optimized", "optimized", "reference"] * 2
    with pytest.raises(ValueError):
        abba(True)


def test_native_gate_failure_is_not_confused_with_environment_or_watchdog_failure():
    assert worker_status({"status": "FAIL"}, {"status": "worker_failed"}) == "FAIL"
    assert worker_status({"status": "PASS"}, {"status": "memory_watchdog_terminated"}) == "BLOCKED"
    assert worker_status({}, {"status": "worker_failed"}) == "BLOCKED"
    assert worker_status({"status": "PASS"}, {"status": "completed"}) == "PASS"


def test_inventory_preserves_illegal_contexts_and_exact_widths():
    models = [{"model_id": "gpt2", "revision": "a", "configured_context": 1024},
              {"model_id": "qwen3", "revision": "b", "configured_context": 40960}]
    rows = experiment_inventory(models, routes=("reference", "optimized"))
    assert len(rows) == 40
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["prompt_tokens"] for r in rows} == {128, 512, 2048, 4096, 8192}
    assert all(r["status"] == "NOT_SUPPORTED" for r in rows
               if r["checkpoint"] == "gpt2" and r["prompt_tokens"] >= 2048)
    assert all(r["repetitions"] == 2 and r["warmups"] == 1 for r in rows)


@pytest.mark.parametrize("key", list(identity()))
def test_mismatches_never_form_comparison(key):
    a, b = identity(), identity()
    b[key] = "changed"
    with pytest.raises(ValueError, match="unmatched"):
        validate_pair(a, b)


def test_missing_identity_and_profiled_results_cannot_qualify():
    a = identity()
    with pytest.raises(ValueError, match="incomplete"):
        validate_pair(a, {})
    a["profile"] = True
    with pytest.raises(ValueError, match="instrumented"):
        validate_pair(a, a)


def test_missing_measurements_and_numerical_gate_stay_pending():
    a = {"identity": identity(), "correctness_passed": True, "measurements": [
        {"prefill_tps": 10, "decode_tps": 20, "peak_mlx_bytes": 100,
         "rss_peak_bytes": 200}] * 10}
    b = copy.deepcopy(a)
    assert compare(a, b)["status"] == "MEASURED"
    b["correctness_passed"] = False
    assert compare(a, b)["status"] == "PENDING"
    b["correctness_passed"] = True
    b["measurements"][0]["rss_peak_bytes"] = None
    assert compare(a, b)["status"] == "PENDING"
    assert compare(a, b)["qualification_promoted"] is False


def test_summary_never_invents_unmeasured_values():
    assert all(value is None for value in summarize([]).values())
    assert summarize([1, 2, 3, 4])["p95"] == 4
    assert summarize([1, 2, 3, 4])["p50"] == 2.5
    with pytest.raises(ValueError):
        summarize([float("nan")])


def test_waits_for_existing_native_work_without_using_elapsed_time_as_success(tmp_path):
    path = tmp_path / "driver.json"
    path.write_text(json.dumps({"status": "running"}))
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        path.write_text(json.dumps({"status": "PASS"}))
    await_prerequisites([path], clock=lambda: 0, sleep=sleep)
    assert sleeps == [5]
    path.write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(RuntimeError, match="prerequisite stopped"):
        await_prerequisites([path])


def test_stalled_prerequisite_times_out_without_native_initialization(tmp_path):
    path = tmp_path / "driver.json"
    path.write_text(json.dumps({"status": "QUEUED"}))
    ticks = iter([0, 2])
    with pytest.raises(TimeoutError):
        await_prerequisites([path], timeout=1, clock=lambda: next(ticks))


@pytest.mark.parametrize("change", ("source", "precision", "rows", "fixture", "reuse", "leak"))
def test_plausible_pass_with_incomplete_worker_proof_is_rejected(change):
    metadata = {"model_id": "qwen3", "revision": "a" * 40, "assets": {"config": "hash"}}
    source = {"source_sha256": "c" * 64}
    task = {"prompt_tokens": 2, "output_tokens": 128, "repetitions": 1}
    result = {"identity": identity(), "measurements": [{"generated_tokens": 2,
        "timing": {"token_ids": [3, 4], "actual_prefill_tokens": 2, "batch_sizes": [1, 1]}}],
        "ownership": {"active_handles": 0, "reserved_bytes": 0, "allocated_bytes": 0}}
    result["identity"]["assets_sha256"] = canonical_digest(metadata["assets"])
    validate_worker_result(result, task, metadata, source)
    if change == "source":
        result["identity"]["source_sha256"] = "old"
    elif change == "precision":
        result["identity"]["kv_dtype"] = "float32"
    elif change == "rows":
        result["measurements"] = []
    elif change == "fixture":
        result["measurements"][0]["timing"]["token_ids"] = [5, 6]
    elif change == "reuse":
        result["measurements"][0]["timing"]["actual_prefill_tokens"] = 1
    else:
        result["ownership"]["active_handles"] = 1
    with pytest.raises(ValueError):
        validate_worker_result(result, task, metadata, source)
