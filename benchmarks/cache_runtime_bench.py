#!/usr/bin/env python3
"""HTTP qualification harness for the mlx-one Cache Runtime V1.

Start an mlx-one server for one model/backend, then point this script at it.  The
server is intentionally external: each backend needs a fresh process and this
keeps measured work on the actual scheduler and HTTP route.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from copy import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from mlx_one.inference_baseline import deterministic_prompt


@dataclass(frozen=True)
class RunSpec:
    model: str
    context_tokens: int
    parallel: int
    workload: str
    run: int
    client_concurrency: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model currently loaded by the server")
    model_group.add_argument(
        "--models", help="Comma-separated models, paired with comma-separated --base-urls"
    )
    parser.add_argument(
        "--base-urls", help="Comma-separated server URLs for --models batch execution"
    )
    parser.add_argument(
        "--tokenizer-source", help="Local path or Hugging Face model for prompt sizing"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Resolve tokenizer assets locally only"
    )
    parser.add_argument("--contexts", default="2048,8192")
    parser.add_argument("--allow-long-context", action="store_true")
    parser.add_argument("--api-key-env", help="Environment variable containing the bearer key")
    parser.add_argument("--client-concurrency", type=int, default=8)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inference-mode", required=True, choices=(
        "reference_dense", "experimental_block", "experimental_dense_batch",
        "paged", "experimental_hybrid_batch"))
    parser.add_argument("--parallel", default="1,2,4,8")
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.90)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def parse_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit(f"{name} must be a comma-separated list of integers") from exc
    if not values or any(item < 1 for item in values):
        raise SystemExit(f"{name} must contain positive integers")
    return values


def parse_strings(value: str, *, name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise SystemExit(f"{name} must contain at least one value")
    return values


def load_tokenizer(source: str, *, offline: bool) -> Any:
    """Load tokenizer assets without constructing an MLX model."""

    from mlx_one.text.loading import _load_tokenizer, _resolve, resolve_text_model_type

    root = _resolve(source, revision=None, offline=offline, cache_dir=None)
    return _load_tokenizer(root, resolve_text_model_type(root, offline=True))


async def runtime(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/v1/runtime")
    response.raise_for_status()
    return response.json()


def authentication_headers(variable):
    if variable is None:
        return {}
    value = os.environ.get(variable)
    if not value:
        raise SystemExit("API key environment variable is unset or empty")
    return {"Authorization": "Bearer " + value}


def request_messages(prompt):
    return [{"role": "user", "content": prompt
             + "\nReturn only the final PROJECT ID, OWNER, and CHECKSUM."}]


def exact_prefix_metadata(tokenizer, template, prompts):
    import hashlib

    arrays = [tuple(tokenizer.encode(template.render(request_messages(p)))) for p in prompts]
    shared = min(map(len, arrays), default=0)
    for index in range(shared):
        if len({ids[index] for ids in arrays}) > 1:
            shared = index
            break
    return {"verified_common_prefix_tokens": shared,
            "actual_rendered_lengths": [len(ids) for ids in arrays],
            "rendered_token_sha256": [hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                                      for ids in arrays]}


async def request(client, *, model, prompt, max_tokens):
    started = time.perf_counter()
    arrivals, parts, metrics = [], [], {}
    done = finished = False
    try:
        async with client.stream("POST", "/v1/chat/completions", json={
            "model": model, "messages": request_messages(prompt), "temperature": 0,
            "max_tokens": max_tokens, "stream": True,
        }) as response:
            if response.status_code >= 400:
                return {"status": "unsupported" if response.status_code == 400 else "failed",
                        "http_status": response.status_code}
            async for line in response.aiter_lines():
                if line == "data: [DONE]":
                    done = True
                    break
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    return {"status": "failed", "reason": "server streaming error"}
                choice = event.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                text = delta.get("content", "") + delta.get("reasoning_content", "")
                if text:
                    arrivals.append(time.perf_counter())
                    parts.append(text)
                if event.get("mlx") is not None:
                    metrics = event["mlx"]
                finished |= choice.get("finish_reason") is not None
        return {"status": "completed" if done and finished and metrics else "failed",
                "client_latency_ms": (time.perf_counter() - started) * 1000,
                "client_ttft_ms": (arrivals[0] - started) * 1000 if arrivals else None,
                "sse_delivery_gaps_ms": [(b - a) * 1000 for a, b in
                                         zip(arrivals, arrivals[1:], strict=False)],
                "response": "".join(parts), "metrics": metrics}
    except (httpx.HTTPError, ValueError):
        return {"status": "failed", "reason": "HTTP transport or stream parsing failure"}


async def execute_group(
    client: httpx.AsyncClient,
    *,
    spec: RunSpec,
    prompt: str,
    max_tokens: int,
    tokenizer=None,
    template=None,
) -> dict[str, Any]:
    prompts = ([prompt + "\nREQUEST-SEED"] if spec.workload == "common_prefix" else [])
    prompts += [prompt + f"\nREQUEST-{index:02d}" for index in range(spec.client_concurrency)]
    token_metadata = exact_prefix_metadata(tokenizer, template, prompts) if tokenizer else {}
    if spec.workload == "common_prefix":
        seed = await request(
            client,
            model=spec.model,
            prompt=prompt + "\nREQUEST-SEED",
            max_tokens=max_tokens,
        )
        followers = await asyncio.gather(
            *(
                request(
                    client,
                    model=spec.model,
                    prompt=prompt + f"\nREQUEST-{index:02d}",
                    max_tokens=max_tokens,
                )
                for index in range(spec.client_concurrency)
            )
        )
        payloads = [seed, *followers]
    else:
        payloads = await asyncio.gather(
            *(
                request(
                    client,
                    model=spec.model,
                    prompt=prompt + f"\nREQUEST-{index:02d}",
                    max_tokens=max_tokens,
                )
                for index in range(spec.client_concurrency)
            )
        )
    completed = [item for item in payloads if item["status"] == "completed"]
    latencies = [item["client_latency_ms"] for item in completed]
    ttft = [item["metrics"].get("time_to_first_token_ms") for item in completed]
    return {
        **asdict(spec),
        "tokenization": token_metadata,
        "requests": payloads,
        "summary": {
            "completed": len(completed),
            "actual_prefill_tokens": sum(item["metrics"].get("actual_prefill_tokens", 0)
                                         or 0 for item in completed),
            "client_ttft_p95_ms": percentile([item["client_ttft_ms"] for item in completed
                                             if item.get("client_ttft_ms") is not None], .95),
            "rejected_or_failed": len(payloads) - len(completed),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "ttft_p50_ms": percentile([item for item in ttft if item is not None], 0.50),
            "ttft_p95_ms": percentile([item for item in ttft if item is not None], 0.95),
            "prefix_tokens_reused": sum(
                item["metrics"].get("reused_prompt_tokens", 0) for item in completed
            ),
            "prefill_tokens_avoided": sum(
                item["metrics"].get("avoided_prefill_tokens", 0) for item in completed
            ),
        },
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = max(round(fraction * 100) - 1, 0)
    return statistics.quantiles(values, n=100, method="inclusive")[index]


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.tokenizer_source or args.model, offline=args.offline)
    from mlx_one.text.chat import ChatTemplate
    from mlx_one.text.loading import _resolve, resolve_text_model_type

    root = _resolve(args.tokenizer_source or args.model, revision=args.revision,
                    offline=args.offline, cache_dir=None)
    template = ChatTemplate.from_directory(root, resolve_text_model_type(root, offline=True),
                                           tokenizer)
    contexts = parse_ints(args.contexts, name="--contexts")
    if any(v > 8192 for v in contexts) and not args.allow_long_context:
        raise SystemExit("long contexts require --allow-long-context and safe server admission")
    if args.runs < 3 or args.client_concurrency < 1:
        raise SystemExit("require three repetitions and positive client concurrency")
    parallel_values = parse_ints(args.parallel, name="--parallel")
    if not 0 < args.shared_prefix_ratio < 1:
        raise SystemExit("--shared-prefix-ratio must be between zero and one")
    limits = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=limits,
                                 headers=authentication_headers(args.api_key_env)) as client:
        before = await runtime(client)
        inference = before.get("inference", {})
        scope = inference.get("scope", {})
        if (inference.get("engine_mode") != args.inference_mode
                or scope.get("revision") != args.revision):
            raise SystemExit("server execution mode or checkpoint revision mismatch")
        configured_parallel = int(before.get("scheduler", {}).get("parallel_slots", 1))
        if any(value != configured_parallel for value in parallel_values):
            raise SystemExit(
                "the server is configured for parallel="
                f"{configured_parallel}; run one server process per --parallel value"
            )
        results = []
        for context in contexts:
            shared = deterministic_prompt(tokenizer, int(context * args.shared_prefix_ratio))
            for parallel in parallel_values:
                for run in range(args.runs):
                    independent = deterministic_prompt(tokenizer, context, suffix=f"\nRUN-{run}")
                    results.append(
                        await execute_group(
                            client,
                            spec=RunSpec(args.model, context, parallel, "independent", run,
                                         args.client_concurrency),
                            prompt=independent,
                            max_tokens=args.max_new_tokens, tokenizer=tokenizer, template=template,
                        )
                    )
                    results.append(
                        await execute_group(
                            client,
                            spec=RunSpec(args.model, context, parallel, "common_prefix", run,
                                         args.client_concurrency),
                            prompt=shared,
                            max_tokens=args.max_new_tokens, tokenizer=tokenizer, template=template,
                        )
                    )
        after = await runtime(client)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": hardware_metadata(),
        "server_runtime_before": before,
        "server_runtime_after": after,
        "arguments": vars(args) | {"output": str(args.output) if args.output else None},
        "results": results,
    }


def hardware_metadata() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "mlx_one_commit": git_revision(),
    }


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.models is None:
        payload: dict[str, Any] = asyncio.run(main_async(args))
    else:
        models = parse_strings(args.models, name="--models")
        if args.base_urls is None:
            raise SystemExit("--models requires one --base-urls entry per model")
        base_urls = parse_strings(args.base_urls, name="--base-urls")
        if len(models) != len(base_urls):
            raise SystemExit("--models and --base-urls must have the same number of entries")
        matrices = []
        for model, base_url in zip(models, base_urls, strict=True):
            matrix_args = copy(args)
            matrix_args.model = model
            matrix_args.base_url = base_url
            matrices.append(asyncio.run(main_async(matrix_args)))
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "models": models,
            "matrices": matrices,
        }
    destination = args.output or (
        Path("benchmarks/results") / f"cache_runtime_{int(time.time())}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
