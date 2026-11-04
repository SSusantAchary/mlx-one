"""Real loopback API, cancellation/recovery and 100-request B1 checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import httpx
    from evidence import snapshot

    spec = importlib.util.spec_from_file_location(
        "competitors", ROOT / "scripts/compare_competators.py"
    )
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    model = json.loads((ROOT / "local/competitor-comparison/models.json").read_text())["models"][0]
    snapshot(output, model=model, flags={"fast_decode": True})
    python = ROOT / "local/competitor-comparison/venvs/mlx-one/bin/python"
    port = b.free_port()
    base = f"http://127.0.0.1:{port}"
    command = b.server_command("mlx-one", python, model, output, port)
    command[command.index("reference_dense")] = "mlxboost_b1"
    command = [
        str(python),
        str(ROOT / "benchmarks/mlxboost/local_server.py"),
        *command[1:],
        "--mlxboost-fast-decode",
        "--api-key",
        "integrity-key",
    ]
    headers = {"Authorization": "Bearer integrity-key"}
    body = {
        "model": model["key"],
        "prompt": model["prompts"][1]["text"],
        "max_tokens": 8,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
    }
    result = {"command": command, "checks": {}, "requests": []}
    with b.launch(command, output, 180, headers, base) as launch_state:
        with httpx.Client(timeout=60, trust_env=False, headers=headers) as client:
            unauthorized = httpx.post(base + "/v1/completions", json=body, trust_env=False)
            result["checks"]["authentication"] = unauthorized.status_code in {401, 403}
            streamed = b.read_stream(client, base + "/v1/completions", body, 128)
            plain = client.post(base + "/v1/completions", json={**body, "stream": False})
            plain.raise_for_status()
            payload = plain.json()
            result["checks"]["stream_nonstream_exact_bytes"] = (
                streamed["text"] == payload["choices"][0]["text"]
            )
            result["checks"]["stream_nonstream_usage"] = (
                streamed["completion_tokens"] == payload["usage"]["completion_tokens"] == 8
            )
            result["checks"]["invalid_input"] = (
                client.post(base + "/v1/completions", json={**body, "max_tokens": -1}).status_code
                == 422
            )
            first = streamed["text"]
            stop = first[1:5] if len(first) >= 5 else first
            stopped = b.read_stream(client, base + "/v1/completions", {**body, "stop": [stop]}, 128)
            result["checks"]["subtoken_stop"] = stop not in stopped["text"]
            # Close a live stream after the first visible fragment, then require
            # an independent request to recover with exact baseline output.
            with client.stream(
                "POST", base + "/v1/completions", json={**body, "max_tokens": 128}
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        event = json.loads(line[6:])
                        if any(choice.get("text") for choice in event.get("choices", [])):
                            break
            recovered = b.read_stream(client, base + "/v1/completions", body, 128)
            result["checks"]["decode_disconnect_recovery"] = recovered["text"] == first
            for index in range(100):
                sample = b.read_stream(client, base + "/v1/completions", body, 128)
                sample["request_index"] = index
                result["requests"].append(sample)
                if index % 20 == 0:
                    print("requests", index, flush=True)
            result["checks"]["100_request_exact_output_usage"] = all(
                r["text"] == first and r["completion_tokens"] == 8 and r["cached_tokens"] == 0
                for r in result["requests"]
            )
            active = [r["server_metrics"]["memory_used_bytes"] for r in result["requests"]]
            result["active_bytes_range"] = [min(active), max(active)]
            result["memory_scope"] = (
                "per-request MLX active at terminal metrics, not RSS or a long soak"
            )
            result["prefill_cancel"] = (
                "NOT_MEASURED: short 128-token prefill completes before probe"
            )
            result["ended_at_unix"] = time.time()
        result["peak_rss_bytes"] = launch_state[1].peak
    result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    snapshot(output, model=model, status=result["status"], flags={"fast_decode": True})
    if result["status"] != "PASS":
        raise SystemExit("API integrity failed")


if __name__ == "__main__":
    main()
