"""Controlled forward throughput after all histories and first tokens are prepared.

This is a forced-token model measurement, separate from HTTP service throughput.
The same four requests execute in serial or batches; prefill is outside the timer.
"""
from __future__ import annotations

import hashlib
import json
import time

DECODE_VERSION = "native-p1-prepared-decode-v1"


def measure_decode(forward, steps, *, clock=time.perf_counter, synchronize=lambda: None):
    if steps < 1:
        raise ValueError("decode measurement requires an interval after the first token")
    synchronize()
    start = clock()
    tokens = 0
    for step in range(steps):
        tokens += forward(step)
    synchronize()
    elapsed = clock() - start
    return {"interval_tokens": tokens, "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else None}


def native_decode(bundle, profile):
    import mlx.core as mx

    from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.hybrid import HybridRunner
    from mlx_one.engine.paged import PagedRunner
    from mlx_one.inference_baseline_worker import build_exact_messages, build_messages

    backend = {"reference_dense": "dense", "experimental_dense_batch": "dense",
               "experimental_hybrid_batch": "hybrid", "paged": "paged"}[profile.inference_mode]
    runtime = CacheManager(bundle, CacheConfig(backend=backend,
        block_size_tokens=profile.page_tokens, key_type="f16", value_type="f16",
        prefix_cache_enabled=False))
    runtime.harness_only, runtime.mode = True, profile.inference_mode
    runner = (HybridRunner() if runtime.hybrid_state else PagedRunner()
              if backend == "paged" else BatchedDenseRunner())
    serial = profile.inference_mode == "reference_dense"
    batch = 1 if serial else profile.slots
    prompts = []
    for index in range(4):
        builder = build_exact_messages if profile.exact_prompt_tokens else build_messages
        _, rendered = builder(bundle, profile.context_tokens, suffix=f"\nRequest {index}.")
        prompts.append(tuple(bundle.tokenizer.encode(rendered)))
    fixture = bundle.tokenizer.encode(" A deterministic continuation.")
    forced = [fixture[i % len(fixture)] for i in range(profile.output_tokens - 1)]
    encoded = json.dumps({"prompts": prompts, "forced": forced}).encode()
    identity = hashlib.sha256(encoded).hexdigest()
    samples, traces = [], []
    handles = []
    def forward(group):
        if serial:
            request = group[0]
            result = bundle.model(mx.array([request.token_ids]), cache=request.cache)
            mx.eval(result.logits)
        else:
            runner.execute(tuple(group))
        return len(group)
    try:
        for repetition in range(profile.repeats + 1):
            handles = [runtime.create_handle(f"decode-{repetition}-{i}",
                       key_type="f16", value_type="f16") for i in range(4)]
            # Evaluate every history before starting the controlled decode interval.
            for ids, handle in zip(prompts, handles, strict=True):
                for start in range(0, len(ids), profile.chunk_tokens):
                    request = ForwardRequest(bundle, ids[start:start + profile.chunk_tokens],
                        handle.bundle.entries, "prefill", ids, runtime=runtime, handle=handle,
                        request_id=handle.request_id)
                    forward([request])
            def step(index, handles=handles, repetition=repetition):
                count = 0
                for start in range(0, 4, batch):
                    group = [ForwardRequest(bundle, (forced[index],), handle.bundle.entries,
                        "decode", ids, runtime=runtime, handle=handle,
                        request_id=handle.request_id) for ids, handle in zip(
                            prompts[start:start + batch], handles[start:start + batch],
                            strict=True)]
                    count += forward(group)
                    if index == 0:
                        traces.append({"repetition": repetition, "batch_size": len(group),
                                       "request_ids": [r.request_id for r in group]})
                return count
            sample = measure_decode(step, len(forced), synchronize=mx.synchronize)
            sample["phase"] = "warmup" if repetition == 0 else "measured"
            samples.append(sample)
            for handle in handles:
                runtime.release(handle)
            handles = []
        return {"schema_version": DECODE_VERSION, "workload": "forced-token-prepared-histories",
                "input_identity": identity, "rendered_token_ids": prompts,
                "forced_token_ids": forced, "requests": 4, "samples": samples,
                "batch_calls": traces, "rates": [s["tokens_per_second"] for s in samples[1:]],
                "excludes": ["prefill", "first_output_token", "sampling", "HTTP", "SSE"],
                "cleanup": runtime.stats().to_dict(), "qualification_promoted": False}
    finally:
        for handle in handles:
            runtime.release(handle)
        runtime.close()
