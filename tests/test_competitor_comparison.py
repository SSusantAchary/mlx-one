"""Reject contaminated/incomplete benchmarks and count usage, never SSE events."""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location(
    "competitor_comparison", Path(__file__).resolve().parents[1] / "scripts/compare_competators.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def read(events, *, done=True, expected=128):
    payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if done:
        payload += "data: [DONE]\n\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=payload, headers={"Content-Type": "text/event-stream"}
        )
    )
    with httpx.Client(transport=transport) as client:
        return benchmark.read_stream(client, "http://test/v1/completions", {}, expected)


def events(*, prompt_tokens=128, completion_tokens=12, cached_tokens=0):
    return [
        {"choices": [{"text": "hello ", "finish_reason": None}]},
        {"choices": [{"text": "world", "finish_reason": "length"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        },
    ]


def test_usage_tokens_are_independent_of_sse_chunk_count():
    sample = read(events())
    assert len(sample["events"]) == 3
    assert sample["completion_tokens"] == 12
    assert sample["text"] == "hello world"
    assert sample["output_tokens_per_second"] == pytest.approx(12 / sample["latency_seconds"])


@pytest.mark.parametrize("prompt_tokens", [None, 127, 129])
def test_different_prompt_token_counts_are_rejected(prompt_tokens):
    with pytest.raises(ValueError, match="prompt token mismatch"):
        read(events(prompt_tokens=prompt_tokens))


def test_prompt_cache_hits_are_rejected():
    with pytest.raises(ValueError, match="caching contaminated"):
        read(events(cached_tokens=64))


def test_incomplete_sse_is_rejected_even_if_usage_is_present():
    with pytest.raises(ValueError, match="incomplete stream"):
        read(events(), done=False)


@pytest.mark.parametrize("count", [None, 0, -1, 12.5])
def test_absent_or_invalid_usage_is_not_estimated_from_text(count):
    with pytest.raises(ValueError, match="authoritative completion-token"):
        read(events(completion_tokens=count))


def test_mlx_one_terminal_metrics_are_supported_and_cache_hits_rejected():
    stream = events()[:2]
    stream[-1]["mlx"] = {"prompt_tokens": 128, "generated_tokens": 12, "reused_prompt_tokens": 0}
    assert read(stream)["completion_tokens"] == 12
    stream[-1]["mlx"]["reused_prompt_tokens"] = 64
    with pytest.raises(ValueError, match="caching contaminated"):
        read(stream)


def test_server_errors_are_never_treated_as_results():
    with pytest.raises(RuntimeError, match="SSE error"):
        read([{"error": {"message": "inference failed"}}])


def test_full_length_view_uses_the_same_subset_for_every_runtime():
    receipt = {
        "protocol": {"output_budget": 128, "rounds": 1, "repetitions_per_round": 2},
        "models": [{"key": "example", "repo": "test/example"}],
    }
    grouped = defaultdict(list)
    for backend in benchmark.BACKENDS:
        grouped[("example", backend)] = [
            {
                "prompt_index": i,
                "completion_tokens": 128,
                "finish_reason": "length",
                "ttft_ms": 1,
                "latency_seconds": 1,
                "output_tokens_per_second": 128,
                "text": "test",
            }
            for i in range(2)
        ]
    grouped[("example", "omlx")][0].update(completion_tokens=44, finish_reason="stop")
    rows, analysis = benchmark.comparison_analysis(receipt, grouped)
    assert analysis["example"]["full_length_prompt_indices"] == [1]
    assert analysis["example"]["common_prompt_indices"] == [0, 1]
    assert all(len(values) == 2 for values in grouped.values())
    for backend in benchmark.BACKENDS:
        assert any(f"| test/example | {backend} | 1 |" in row for row in rows)


def test_missing_runtime_cannot_produce_a_comparison_winner():
    receipt = {
        "protocol": {"output_budget": 128, "rounds": 1, "repetitions_per_round": 1},
        "models": [{"key": "example", "repo": "test/example"}],
    }
    grouped = defaultdict(list)
    grouped[("example", "mlx-one")] = [
        {
            "prompt_index": 0,
            "completion_tokens": 128,
            "finish_reason": "length",
            "ttft_ms": 1,
            "latency_seconds": 1,
            "output_tokens_per_second": 128,
            "text": "test",
        }
    ]
    rows, analysis = benchmark.comparison_analysis(receipt, grouped)
    assert analysis["example"]["full_length_prompt_indices"] == []
    assert not any("had the highest" in row for row in rows)
