#!/usr/bin/env python3
"""Measure CLI process or HTTP serving costs separately from direct-model scores."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface", choices=["cli", "http"], required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--cli", default="mlx-one")
    p.add_argument("--base-url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="tq3")
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.repetitions < 1 or a.max_tokens < 1:
        p.error("positive repetitions and token budget required")
    prompt = a.prompt_file.read_text()
    samples = []
    for _ in range(a.repetitions):
        start = time.perf_counter()
        if a.interface == "cli":
            if not a.checkpoint or not a.checkpoint.is_dir():
                p.error("CLI requires an existing local checkpoint")
            result = subprocess.run(
                [
                    a.cli,
                    "generate",
                    str(a.checkpoint),
                    prompt,
                    "--offline",
                    "--max-tokens",
                    str(a.max_tokens),
                    "--json-output",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
        else:
            import httpx

            headers = {"Authorization": "Bearer " + os.environ.get("MLX_ONE_API_KEY", "local-test")}
            with httpx.Client(base_url=a.base_url, headers=headers, timeout=600) as client:
                response = client.post(
                    "/v1/completions",
                    json=dict(
                        model=a.model,
                        prompt=prompt,
                        max_tokens=a.max_tokens,
                        temperature=0,
                        stream=False,
                    ),
                )
                response.raise_for_status()
                payload = response.json()
        samples.append(dict(wall_seconds=time.perf_counter() - start, response=payload))
    report = dict(
        schema_version="tq3-interfaces-v1",
        interface=a.interface,
        measurement="fresh CLI process including load"
        if a.interface == "cli"
        else "HTTP request including transport; model already resident",
        direct_model_comparison=False,
        requested_tokens=a.max_tokens,
        actual_counts=[
            x["response"].get(
                "generation_tokens", x["response"].get("usage", {}).get("completion_tokens")
            )
            for x in samples
        ],
        wall_median_seconds=statistics.median(x["wall_seconds"] for x in samples),
        samples=samples,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
