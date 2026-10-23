#!/usr/bin/env python3
"""Opt-in local GPT-OSS HTTP checks and measured tool-call timing.

Never downloads weights or executes model-requested tools. The synthetic tool result
only confirms receipt; actual coding-agent execution is a separate Kilo acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

ALIAS = "gpt-oss-20b-mxfp4-128k"
TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a text file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}
METRICS = (
    "queue_ms",
    "decode_tokens_per_second",
    "prompt_tokens",
    "generated_tokens",
    "generation_latency_ms",
    "prefill_ms",
    "decode_ms",
    "prompt_tokens_per_second",
    "tokens_per_second",
    "server_text_ttft_ms",
    "time_to_first_token_ms",
    "memory_used_bytes",
    "memory_peak_bytes",
)


def stream(client, payload):
    started = time.perf_counter()
    first = first_tool = complete = None
    calls, content, usage, metrics = {}, "", None, {}
    finish, done = None, False
    with client.stream(
        "POST",
        "/chat/completions",
        json={**payload, "stream": True, "stream_options": {"include_usage": True}},
    ) as response:
        response.raise_for_status()
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if "error" in event:
                raise AssertionError(event["error"])
            if event.get("usage"):
                assert event["choices"] == []
                usage = event["usage"]
            metrics = event.get("mlx", metrics)
            for choice in event["choices"]:
                delta = choice["delta"]
                if (delta.get("content") or delta.get("tool_calls")) and first is None:
                    first = time.perf_counter() - started
                content += delta.get("content", "")
                for item in delta.get("tool_calls", []):
                    if first_tool is None:
                        first_tool = time.perf_counter() - started
                    call = calls.setdefault(
                        item["index"],
                        {
                            "id": item.get("id"),
                            "type": "function",
                            "function": {"name": item["function"].get("name"), "arguments": ""},
                        },
                    )
                    call["function"]["arguments"] += item["function"].get("arguments", "")
                if choice["finish_reason"] is not None:
                    complete = time.perf_counter() - started
                    finish = choice["finish_reason"]
    assert done and usage is not None
    for call in calls.values():
        assert isinstance(json.loads(call["function"]["arguments"]), dict)
    return {
        "role": "assistant",
        "content": content or None,
        **({"tool_calls": list(calls.values())} if calls else {}),
    }, {
        "status": "PASS",
        "finish_reason": finish,
        "ttft_s": first,
        "first_tool_delta_s": first_tool,
        "complete_s": complete,
        "http_total_s": time.perf_counter() - started,
        "usage": usage,
        "native_metrics": {key: metrics.get(key) for key in METRICS},
    }


def check_robustness(client, base):
    checks = {}
    payloads = {
        "unknown_model": ({**base, "model": "missing-model"}, 404),
        "required_choice": ({**base, "tools": [TOOL], "tool_choice": "required"}, 400),
        "unknown_function": (
            {
                **base,
                "tools": [TOOL],
                "tool_choice": {"type": "function", "function": {"name": "absent"}},
            },
            422,
        ),
        "context_overflow": (
            {
                **base,
                "tools": [TOOL],
                "max_tokens": 16384,
                "messages": [{"role": "user", "content": " token" * 130000}],
            },
            400,
        ),
    }
    for name, (payload, expected) in payloads.items():
        response = client.post("chat/completions", json=payload)
        assert response.status_code == expected, (name, response.status_code, response.text[:300])
        checks[name] = {"status": "PASS", "http_status": response.status_code}
    if client.headers.get("Authorization"):
        response = httpx.get(str(client.base_url) + "models", timeout=30)
        assert response.status_code == 401
        checks["no_auth"] = {"status": "PASS", "http_status": 401}
        response = client.get("models", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401
        checks["wrong_auth"] = {"status": "PASS", "http_status": 401}
    return checks


def check_cancellation(client, base):
    """Disconnect while prefill has emitted a role delta but no model output."""
    started = time.perf_counter()
    payload = {
        **base,
        "tools": [TOOL],
        "stream": True,
        "max_tokens": 256,
        "messages": [{"role": "user", "content": " token" * 10000}],
    }
    with client.stream("POST", "chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.startswith("data: "):
                first = json.loads(line[6:])
                assert first["choices"][0]["delta"].get("role") == "assistant"
                break
    disconnected = time.perf_counter()
    while time.perf_counter() - disconnected < 30:
        runtime = client.get("runtime").json()
        if runtime.get("cache", {}).get("active_handles") == 0:
            break
        time.sleep(0.1)
    else:
        raise AssertionError("disconnect did not release native KV handles")
    cleanup_s = time.perf_counter() - disconnected
    followup = client.post("chat/completions", json={**base, "max_tokens": 256})
    followup.raise_for_status()
    assert followup.json()["choices"][0]["message"]["content"]
    return {
        "status": "PASS",
        "active_handles_after": 0,
        "cleanup_s": cleanup_s,
        "http_total_s": time.perf_counter() - started,
        "followup_chat": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8084/v1/")
    parser.add_argument("--model", default=ALIAS)
    parser.add_argument("--api-key", default=os.environ.get("MLX_ONE_API_KEY"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "checkpoint": "local GPT-OSS",
        "model": args.model,
        "checks": {},
        "limitations": [
            "128K inference not tested",
            "cold load time not measured",
            "HTTP check never executes tools; Kilo tested separately",
        ],
    }
    headers = {"Authorization": "Bearer " + args.api_key} if args.api_key else {}
    try:
        with httpx.Client(base_url=args.url, headers=headers, timeout=1200) as client:
            models = client.get("models")
            models.raise_for_status()
            assert args.model in [item["id"] for item in models.json()["data"]]
            report["checks"]["discovery"] = "PASS"
            base = {
                "model": args.model,
                "max_tokens": 512,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with exactly: Hello."}],
            }
            plain = client.post("chat/completions", json=base)
            plain.raise_for_status()
            assert plain.json()["choices"][0]["message"]["content"]
            report["checks"]["plain_chat"] = "PASS"
            message, result = stream(client, base)
            assert message["content"] and "tool_calls" not in message
            report["checks"]["plain_sse"] = result
            tool_base = {
                **base,
                "tools": [TOOL],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use write_file now to create hello.py with the exact text "
                        "print('Hello'). "
                        "You must call the provided function rather than describe the file.",
                    }
                ],
            }
            single = client.post("chat/completions", json=tool_base)
            single.raise_for_status()
            data = single.json()
            assert data["choices"][0]["finish_reason"] == "tool_calls", data["choices"]
            call = data["choices"][0]["message"]["tool_calls"][0]
            assert call["function"]["name"] == "write_file"
            assert isinstance(json.loads(call["function"]["arguments"]), dict)
            report["checks"]["nonstream_tool"] = "PASS"
            measurements = []
            for index in range(args.repetitions + 1):
                message, result = stream(client, tool_base)
                assert result["finish_reason"] == "tool_calls", message
                if index:
                    measurements.append(result)
                else:
                    report["checks"]["tool_warmup"] = result
                print(f"Tool repetition {index}: PASS ({result['http_total_s']:.3f}s)", flush=True)
            report["checks"]["stream_tools"] = measurements
            history = tool_base["messages"] + [message]
            for call in message["tool_calls"]:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": "Success: hello.py was written with the requested content.",
                    }
                )
            continuation_started = time.perf_counter()
            continuation, result = stream(client, {**tool_base, "messages": history})
            assert continuation["content"] and result["finish_reason"] == "stop", continuation
            report["checks"]["tool_continuation"] = result
            report["checks"]["http_round_trip_s"] = (
                measurements[-1]["http_total_s"] + time.perf_counter() - continuation_started
            )
            fixture = json.loads(
                (
                    Path(__file__).resolve().parents[1] / "tests/fixtures/kilo/request.json"
                ).read_text()
            )
            fixture.update(
                model=args.model,
                temperature=0,
                max_tokens=512,
                messages=[
                    {"role": "user", "content": "Use coding_tool_00 with path hello.py now."}
                ],
            )
            message, result = stream(client, fixture)
            assert result["finish_reason"] == "tool_calls", message
            assert message["tool_calls"][0]["function"]["name"] == "coding_tool_00"
            report["checks"]["catalog_29_tools"] = result
            report["checks"]["robustness"] = check_robustness(client, base)
            report["checks"]["cancellation"] = check_cancellation(client, base)
            report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["failure"] = str(exc)[:2000]
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
