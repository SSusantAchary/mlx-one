import json
import subprocess
from dataclasses import asdict, replace

import pytest

from mlx_one.inference_baseline import (
    CHECKPOINTS,
    BaselineError,
    BaselineProfile,
    main,
    profiles,
    run_profile,
    summarize,
    supervise,
    validate_identity,
)


def profile():
    return BaselineProfile(*CHECKPOINTS["qwen"], "reference_dense")


def identity(item):
    return {"inference": {"engine_mode": item.inference_mode, "capabilities": {}, "scope": {
        "model_id": item.model_id, "revision": item.revision, "topology_digest": "topology",
        "weight_precision": "bfloat16", "cache_precision": "bfloat16", "mlx_version": "0.32.2",
        "hardware": "M4", "source_revision": "digest",
    }}, "scheduler": {"parallel_slots": item.slots}}


def test_pinned_profiles_are_cpu_listable_and_overrides_change_identity(capsys):
    items = list(profiles("reference_dense"))
    assert len(items) == 12
    assert {p.context_tokens for p in items} == {2048, 8192}
    assert {p.slots for p in items} == {1, 2, 4}
    assert all(p.output_tokens == 128 and p.repeats == 3 for p in items)
    assert replace(items[0], atol=.1, rtol=.1).profile_id != items[0].profile_id
    assert main(["--inference-mode", "reference_dense", "--list-profiles"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 12


@pytest.mark.parametrize("field,value", [("inference_mode", ""), ("revision", "main"),
                                         ("memory_fraction", .8), ("slots", 0), ("atol", .1)])
def test_invalid_profiles_fail_before_loading_model(field, value):
    with pytest.raises((BaselineError, ValueError)):
        BaselineProfile(**{**asdict(profile()), field: value})


def test_identity_rejects_fallback_dtype_and_missing_metadata():
    item = profile()
    runtime = identity(item)
    validate_identity(item, runtime)
    for field in ("mlx_version", "topology_digest", "source_revision", "hardware"):
        changed = identity(item)
        changed["inference"]["scope"][field] = None
        with pytest.raises(BaselineError, match="incomplete"):
            validate_identity(item, changed)
    runtime["inference"]["scope"]["cache_precision"] = "float16"
    with pytest.raises(BaselineError, match="precision"):
        validate_identity(item, runtime)
    runtime = identity(item)
    runtime["inference"]["engine_mode"] = "experimental_block"
    with pytest.raises(BaselineError, match="substitution"):
        validate_identity(item, runtime)


class Process:
    pid = 10
    stopped = False
    killed = False

    def poll(self):
        return 0 if self.stopped else None

    def terminate(self):
        self.stopped = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -15 if self.stopped else 0


@pytest.mark.parametrize("rss,expected", [(200, "memory_watchdog_terminated"),
                                        (None, "memory_watchdog_unavailable")])
def test_watchdog_stops_and_records_failure(rss, expected):
    process = Process()
    result = supervise(process, timeout=10, memory_limit=100, rss=lambda _: rss,
                       clock=lambda: 0, sleep=lambda _: None)
    assert result["status"] == expected and process.stopped


def test_timeout_kills_unresponsive_worker():
    class Stuck(Process):
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return -9

    ticks = iter([0, 1, 2, 3])
    process = Stuck()
    result = supervise(process, timeout=1, memory_limit=100, rss=lambda _: 50,
                       clock=lambda: next(ticks), sleep=lambda _: None)
    assert result["status"] == "timeout" and process.killed


def test_summary_separates_cold_warm_sse_and_model_timing():
    rows = [dict(status="completed", temperature_phase=phase, group_id=phase,
                 started=1, ended=3, generated_tokens=4, client_ttft_ms=ttft,
                 sse_delivery_gaps_ms=[20, 40], model_token_gaps_ms=[10, 15])
            for phase, ttft in [("cold", 100), ("warmup", 50), ("warm", 30), ("warm", 40)]]
    result = summarize(rows)
    assert result["cold_client_ttft_ms"] == [100]
    assert result["warm_client_ttft_p95_ms"] == 40
    assert result["sse_delivery_gap_p95_ms"] == 40
    assert result["model_token_gap_p95_ms"] == 15
    assert result["aggregate_output_tokens_per_second_by_group"] == [4]
    assert not result["qualification_promoted"]


def test_failed_worker_writes_durable_artifacts(tmp_path, monkeypatch):
    import mlx_one.inference_baseline as baseline

    class Failed(Process):
        def poll(self):
            return 2

        def wait(self, timeout=None):
            return 2

    monkeypatch.setattr(baseline.subprocess, "Popen", lambda *args, **kwargs: Failed())
    output = tmp_path / "run"
    outcome = run_profile(profile(), output, hardware={"chip": "M4", "memory_bytes": 1024},
                          provenance={"source_sha256": "source"})
    assert outcome["status"] == "worker_failed"
    assert json.loads((output / "failure.json").read_text())["exit_code"] == 2
    assert json.loads((output / "manifest.json").read_text())["profile"]["offline"]


def test_sse_metrics_and_terminal_failures_without_network():
    import asyncio

    import httpx

    from mlx_one.inference_baseline_worker import stream_request

    metrics = {"model_token_times_ms": [10, 20, 40],
               "execution_observer": "synchronized-forward-v1", "generated_tokens": 3}
    events = [
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}], "mlx": metrics},
    ]
    text = "".join("data: " + json.dumps(event) + "\n\n" for event in events)

    async def run(content):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=content))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await stream_request(client, profile(), [{"role": "user", "content": "hi"}],
                                        phase="warm", group="warm-0", index=0)

    result = asyncio.run(run(text + "data: [DONE]\n\n"))
    assert result["status"] == "completed"
    assert result["model_token_gaps_ms"] == [10, 20]
    assert len(result["sse_delivery_gaps_ms"]) == 1
    failed = asyncio.run(run(text))
    assert failed["status"] == "failed" and "marker" in failed["error"]


def test_http_prompt_sizing_is_deterministic_and_matches_rendered_limit():
    from types import SimpleNamespace

    from mlx_one.inference_baseline_worker import build_messages

    class Tokenizer:
        def encode(self, value):
            return list(value.encode())

        def decode(self, values):
            return bytes(values).decode()

    bundle = SimpleNamespace(tokenizer=Tokenizer(), chat_template=SimpleNamespace(
        render=lambda messages, **kwargs: "<user>" + messages[0]["content"] + "<assistant>"))
    a = build_messages(bundle, 512, suffix="\nAnswer A.")
    b = build_messages(bundle, 512, suffix="\nAnswer A.")
    assert a == b
    assert len(bundle.tokenizer.encode(a[1])) <= 512
    assert a[0][0]["content"].endswith("\nAnswer A.")


def test_batch_matrix_and_latency_budget_validation():
    items = list(profiles("experimental_dense_batch", workload="batching"))
    assert len(items) == 32
    assert {p.slots for p in items} == {1, 2, 4, 8}
    assert {p.chunk_tokens for p in items} == {512, 1024, 2048, 4096}
    assert all(p.execution_dtype == "float16" and p.cache_type == "f16" for p in items)
    with pytest.raises(BaselineError, match="both latency"):
        replace(items[0], ttft_budget_ms=100)
    with pytest.raises(BaselineError, match="finite"):
        replace(items[0], ttft_budget_ms=float("inf"), decode_gap_budget_ms=100)


def test_gate_c_requires_budgets_matching_identity_and_all_checks():
    from mlx_one.engine.qualification import evaluate_gate_c

    item = next(profiles("experimental_dense_batch", workload="batching"))
    assert evaluate_gate_c(item, {}, [], {})["status"] == "diagnostic"
    item = replace(item, ttft_budget_ms=500, decode_gap_budget_ms=100)
    assert evaluate_gate_c(item, {}, [], {})["status"] == "pending"
    proof = {key: key for key in
             ("hardware", "source", "weight_precision", "cache_precision", "mlx_version")}
    summary = {"aggregate_output_tokens_per_second_by_group": [20, 21, 22],
               "warm_client_ttft_p95_ms": 100, "model_token_gap_p95_ms": 50,
               "failed": 0, "starved_requests": 0, "identity": proof}
    reference = {"profile": replace(item, inference_mode="reference_dense").to_dict(),
                 "identity": proof,
                 "summary": {"aggregate_output_tokens_per_second_by_group": [10, 11, 12]}}
    traces = [{"phase": "decode", "request_ids": ids, "batch_size": len(ids)}
              for ids in [("a", "b"), ("b", "c")]]
    gate = evaluate_gate_c(item, summary, traces, {"status": "passed"}, reference)
    assert gate["status"] == "passed"
    slower = {**summary, "model_token_gap_p95_ms": 101}
    gate = evaluate_gate_c(item, slower, traces, {"status": "passed"}, reference)
    assert gate["status"] == "failed"
    changed = {**reference, "identity": {**proof, "hardware": "other"}}
    with pytest.raises(ValueError, match="hardware"):
        evaluate_gate_c(item, summary, traces, {"status": "passed"}, changed)
