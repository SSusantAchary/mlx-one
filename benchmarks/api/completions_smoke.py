"""Bounded offline checkpoint smoke: Python, raw HTTP/SSE, discovery and cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def run(checkpoint, dtype):
    from fastapi.testclient import TestClient

    from mlx_one import Engine
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text import TextGenerationOptions

    manager = ModelManager()
    bundle = manager.load(checkpoint, offline=True)
    scope = {"model_id": bundle.model_id, "revision": bundle.revision,
             "architecture": bundle.architecture, "quantization": bundle.quantization,
             "execution_dtype": dtype, "mode": "reference_dense", "batch_size": 1}
    prompt = "The capital of France is"
    options = TextGenerationOptions(max_tokens=16, temperature=0)
    with Engine(bundle, inference_mode="reference_dense", execution_dtype=dtype,
                cache_config=CacheConfig(prefix_cache_enabled=False)) as python:
        expected = python.generate(prompt, options=options)
    engine = GenerationEngine(manager, inference_mode="reference_dense", execution_dtype=dtype)
    settings = ServerConfig(warmup=False, cache_prompt=False, inference_mode="reference_dense",
                            execution_dtype=dtype)
    checks = {}
    app = create_app(manager, engine, config=settings)
    with TestClient(app) as client:
        checks["health"] = client.get("/health").status_code == 200
        checks["discovery"] = bundle.model_id in [
            row["id"] for row in client.get("/v1/models").json()["data"]]
        checks["runtime"] = client.get("/v1/runtime").status_code == 200
        payload = {"model": bundle.model_id, "prompt": prompt,
                   "temperature": 0, "max_tokens": 16}
        started = time.perf_counter()
        response = client.post("/v1/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        result = response.json()
        checks["python_http_text"] = result["choices"][0]["text"] == expected.text
        checks["finish_reason"] = result["choices"][0]["finish_reason"] == expected.finish_reason
        checks["token_usage"] = result["usage"] == {
            "prompt_tokens": expected.prompt_tokens,
            "completion_tokens": expected.generation_tokens,
            "total_tokens": expected.prompt_tokens + expected.generation_tokens}
        checks["cleanup_after_completion"] = engine.cache_stats()["active_handles"] == 0
        repeated = client.post("/v1/completions", json=payload).json()
        checks["repeat_greedy"] = repeated["choices"] == result["choices"]
        streamed = client.post("/v1/completions", json={**payload, "stream": True})
        streamed.raise_for_status()
        events = [json.loads(line[6:]) for line in streamed.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        checks["sse_text"] = "".join(e["choices"][0]["text"] for e in events) == expected.text
        checks["sse_finish"] = events[-1]["choices"][0]["finish_reason"] == expected.finish_reason
        checks["sse_done"] = streamed.text.endswith("data: [DONE]\n\n")
        checks["cleanup_after_sse"] = engine.cache_stats()["active_handles"] == 0
        checks["unknown_model"] = client.post("/v1/completions", json={
            **payload, "model": "unknown"}).status_code == 404
        checks["context_overflow"] = client.post("/v1/completions", json={
            **payload, "prompt": " a" * bundle.context_length}).status_code == 400
        checks["invalid_output_budget"] = client.post("/v1/completions", json={
            **payload, "max_tokens": 0}).status_code == 422
    try:
        manager.current_model()
        checks["unload_after_shutdown"] = False
    except RuntimeError:
        checks["unload_after_shutdown"] = True
    return {"schema_version": "revamp-completion-smoke-v1", "scope": scope,
            "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "prompt": prompt, "generated_text": expected.text,
            "usage": result["usage"], "http_generation_latency_ms": latency_ms,
            "metrics": result.get("mlx"), "qualification": "SMOKE_ONLY",
            "notes": "In-process ASGI HTTP, not a socket/load benchmark; chat and cancellation "
                     "are separate tests. No checkpoint numerical or quality qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new evidence path")
    try:
        report = run(args.checkpoint, args.dtype)
    except Exception as exc:
        report = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}",
                  "qualification": "NOT_QUALIFIED"}
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "src/mlx_one").rglob("*.py"))}
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
