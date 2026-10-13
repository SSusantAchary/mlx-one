"""Metal-only measurements used by isolated native qualification workers."""

from __future__ import annotations

import hashlib
import json
import time


async def pressure_recovery(client, bundle, engine, scheduler, profile, output):
    """Exercise live pressure using a declared ceiling change, without allocating host RAM."""
    import asyncio
    from dataclasses import replace

    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.inference_baseline import percentile
    from mlx_one.inference_baseline_worker import build_messages, stream_request, write_json

    controller, runtime = engine._pressure, engine._cache_runtime
    if controller is None or not controller.config.enabled:
        result = {"status": "pending", "reason": "live pressure control is disabled"}
        write_json(output / "pressure-recovery.json", result)
        return result
    messages, _ = build_messages(bundle, profile.context_tokens,
                                 suffix="\nStable pressure recovery.")
    baseline = await stream_request(client, profile, messages,
                                    phase="pressure", group="before", index=0)
    original = controller.config
    task = None
    states = []
    def sample():
        controller.last_sample = -float("inf")
        state = controller.poll()
        states.append(controller.stats())
        return state
    try:
        sample()
        observed = controller.working_set()
        if observed <= 0:
            result = {"status": "pending", "reason": "live working-set measurement unavailable"}
        else:
            controller.config = replace(original, process_ceiling_bytes=int(observed / .95))
            warning = sample()
            task = asyncio.create_task(stream_request(client, profile, messages,
                                       phase="pressure", group="queued", index=0))
            await asyncio.sleep(1.1)
            queued = not task.done() and controller.queued_admissions > 0
            controller.config = original
            sample()
            await asyncio.sleep(1.1)
            sample()
            recovered = await asyncio.wait_for(task, timeout=profile.timeout_seconds)
            controller.config = replace(original, process_ceiling_bytes=max(observed - 1, 1))
            critical = sample()
            rejected = False
            def reservation_guard():
                nonlocal rejected
                try:
                    with runtime.transient_reservation(observed + 1):
                        pass
                except CacheCapacityError as exc:
                    rejected = not exc.retryable
            scheduler.execute(reservation_guard)
            controller.config = original
            sample()
            await asyncio.sleep(1.1)
            sample()
            checks = {"warning": warning == "warning", "admission_queued": queued,
                      "critical": critical == "critical", "typed_reservation_rejection": rejected,
                      "recovered": controller.state == "normal",
                      "output_parity": baseline.get("output_sha256") is not None
                      and baseline.get("output_sha256") == recovered.get("output_sha256")
                      and recovered["status"] == "completed",
                      "reservations_released": runtime.stats().reserved_bytes == 0}
            gap = percentile(recovered.get("model_token_gaps_ms", []), .95)
            ttft = recovered.get("client_ttft_ms")
            checks["service_limits"] = (None if profile.ttft_budget_ms is None else
                ttft is not None and gap is not None and ttft <= profile.ttft_budget_ms
                and gap <= profile.decode_gap_budget_ms)
            status = ("diagnostic" if checks["service_limits"] is None else
                      "passed" if all(checks.values()) else "failed")
            result = {"status": status, "checks": checks,
                      "ceiling_control": "configured-process-ceiling", "states": states,
                      "baseline": baseline, "recovery": recovered}
    finally:
        controller.config = original
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    write_json(output / "pressure-recovery.json", result)
    return result


def prefix_messages(bundle, target, ratio, group, index):
    from mlx_one.inference_baseline import deterministic_prompt

    shared = deterministic_prompt(bundle.tokenizer, max(int(target * ratio) - 64, 0))
    unique = f"REQUEST-{group}-{index}.\n" + deterministic_prompt(
        bundle.tokenizer, max(target - len(bundle.tokenizer.encode(shared)) - 96, 64))
    while True:
        messages = [{"role": "user", "content": shared + unique
                     + "\nReturn the request identifier."}]
        prompt = bundle.chat_template.render(messages, enable_thinking=False)
        if len(bundle.tokenizer.encode(prompt)) <= target:
            return messages, prompt
        ids = bundle.tokenizer.encode(unique)
        if not ids:
            raise ValueError("prefix fixture cannot fit its chat template")
        unique = bundle.tokenizer.decode(ids[:-1])


def attention_microbench(bundle, profile):
    import mlx.core as mx

    from mlx_one.engine.block_cache import BlockPool
    from mlx_one.engine.paged import PagedKVCache, dequantize_page
    from mlx_one.engine.paged_attention import paged_attention
    from mlx_one.inference_baseline import TOLERANCES

    config = bundle.model.config
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    dim = config.hidden_size // heads
    dtype = mx.bfloat16 if profile.cache_type == "bf16" else mx.float16
    pool = BlockPool(2**63 - 1, block_tokens=profile.page_tokens)
    rows, reference, measurements = [], [], []
    try:
        for row in range(profile.slots):
            length = max(profile.context_tokens - row * profile.page_tokens - 1, 1)
            shape = (1, kv_heads, length, dim)
            keys = mx.random.normal(shape, key=mx.random.key(profile.seed + row * 2)).astype(dtype)
            values = mx.random.normal(shape,
                                      key=mx.random.key(profile.seed + row * 2 + 1)).astype(dtype)
            cache = PagedKVCache(pool, dtype="bfloat16" if dtype == mx.bfloat16 else "float16",
                                 bits=profile.kv_bits)
            rows.append(cache)
            cache.append(keys, values)
            if profile.kv_bits:
                materialized = []
                for ref in cache._blocks:
                    block = ref.block
                    pair = ((dequantize_page(block.state[:3], profile.kv_bits, dim),
                             dequantize_page(block.state[3:], profile.kv_bits, dim))
                            if block.layout.quantization_bits else block.state)
                    materialized.append(tuple(t[:, :, :block.token_count].astype(mx.float32)
                                              for t in pair))
                keys, values = tuple(mx.concatenate([p[i] for p in materialized], axis=2)
                                     for i in (0, 1))
            reference.append((keys, values))
        query = mx.random.normal((profile.slots, heads, 1, dim),
                                  key=mx.random.key(profile.seed + 100)).astype(dtype)
        positions = tuple((c.offset - 1,) for c in rows)
        result = None
        for phase in ("cold", "warmup", *(["warm"] * profile.repeats)):
            started = time.perf_counter()
            result = paged_attention(query, rows, positions, (1,) * profile.slots, scale=dim**-.5)
            mx.eval(result)
            measurements.append({"phase": phase, "synchronized_dispatch_ms": (
                time.perf_counter() - started) * 1000})
        maximum, within = 0.0, True
        dense_timings = []
        atol, rtol = (TOLERANCES["bfloat16" if profile.cache_type == "bf16" else "float16"]
                      if profile.atol is None else (profile.atol, profile.rtol))
        for phase in ("cold", "warmup", *(["warm"] * profile.repeats)):
            started = time.perf_counter()
            expected = tuple(mx.fast.scaled_dot_product_attention(
                query[row:row + 1].astype(keys.dtype), keys, values, scale=dim**-.5)
                for row, (keys, values) in enumerate(reference))
            mx.eval(*expected)
            dense_timings.append({"phase": phase, "synchronized_dispatch_ms": (
                time.perf_counter() - started) * 1000})
        for row, expected_row in enumerate(expected):
            error = mx.abs(result[row:row + 1] - expected_row)
            maximum = max(maximum, float(mx.max(error)))
            within &= bool(mx.all(error <= atol + rtol * mx.abs(expected_row)).item())
        return {"status": "diagnostic" if profile.kv_bits else "completed",
                "measurements": measurements, "attention_max_error": maximum,
                "dense_reference_measurements": dense_timings,
                "dense_reference_path": ("independent-dequantized-page-oracle" if profile.kv_bits
                                         else "independent-B1-fused-attention"),
                "oracle_history_materializations": profile.slots if profile.kv_bits else 0,
                "within_tolerance": within, "atol": atol, "rtol": rtol,
                "physical_page_bytes": pool.nbytes, "history_materializations": 0,
                "actual_query_dtype": str(query.dtype),
                "gpu_timestamp_latency_ms": None, "timing": "host-synchronized-kernel-dispatch",
                "qualification_promoted": False}
    finally:
        for row in rows:
            row.reset()
        pool.clear()


def retrieval_quality(bundle, profile):
    from mlx_one import BatchingConfig, CacheConfig, Engine, MemoryPressureConfig
    from mlx_one.inference_baseline import deterministic_prompt

    results = []
    for placement in ("start", "middle", "end"):
        marker = "REFERENCE-" + hashlib.sha256(placement.encode()).hexdigest()[:8].upper()
        filler = deterministic_prompt(bundle.tokenizer, max(profile.context_tokens - 256, 64))
        sentence = f"The secret reference is {marker}.\n"
        index = {"start": 0, "middle": len(filler) // 2, "end": len(filler)}[placement]
        prompt = (filler[:index] + sentence + filler[index:]
                  + "\nReturn only the secret reference, including its REFERENCE- prefix.\n")
        outputs = {}
        for name, mode, cache_type in (("oracle", "reference_dense", "f16"),
                                       ("target", profile.inference_mode, profile.cache_type)):
            from mlx_one.engine.modes import resolve_backend

            with Engine(bundle, inference_mode=mode,
                        cache_config=CacheConfig(backend=resolve_backend(mode, None),
                                                 prefix_cache_enabled=False,
                                                 key_type=cache_type, value_type=cache_type,
                                                 block_size_tokens=profile.page_tokens),
                        batching_config=BatchingConfig(prefill_chunk_size=profile.chunk_tokens),
                        memory_pressure_config=MemoryPressureConfig(enabled=False)) as engine:
                engine._cache_manager.harness_only = True
                result = engine.generate(prompt, max_tokens=profile.output_tokens, temperature=0,
                                         seed=profile.seed)
                outputs[name] = {"text": result.text, "retrieved": marker in result.text}
        results.append({"placement": placement, "expected": marker,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "rendered_tokens": len(bundle.tokenizer.encode(prompt)), **outputs})
    return {"schema_version": "native-retrieval-quality-v1", "fixtures": results,
            "oracle_accuracy": sum(r["oracle"]["retrieved"] for r in results) / len(results),
            "target_accuracy": sum(r["target"]["retrieved"] for r in results) / len(results)}


def quality_run(bundle, engine, profile, output):
    from mlx_one.engine.qualification import execution_parity
    from mlx_one.engine.selection import compatibility_digest, host_identity, source_identity
    from mlx_one.inference_baseline_worker import qualification_evidence, write_json
    from mlx_one.inference_qualification import evaluate_quality, workload_envelope

    parity = execution_parity(bundle, mode=profile.inference_mode,
                              prompt="Shared facts and deterministic held-out continuation. " * 64,
                              count=profile.output_tokens, batch_size=profile.slots,
                              chunk_size=profile.chunk_tokens, cache_type=profile.cache_type,
                              page_tokens=profile.page_tokens, atol=profile.atol, rtol=profile.rtol,
                              quality_limits=profile.quality_limits)
    if profile.quality_manifest:
        from mlx_one.quality_dataset import QUALITY_SCHEMA, native_dataset_quality

        held_out, retrieval = native_dataset_quality(bundle, profile)
        write_json(output / "held-out-quality.json", held_out)
        parity.update({key: held_out[key] for key in (
            "max_error", "mean_error", "held_out_nll_mean_delta", "greedy_agreement_fraction",
            "coverage", "dataset_identity")})
        parity["quality_kind"] = QUALITY_SCHEMA
    else:
        retrieval = retrieval_quality(bundle, profile)
        parity["quality_kind"] = "synthetic-smoke-test"
    quality = evaluate_quality(parity, retrieval, profile.quality_limits)
    kernel_correctness = None
    if profile.kv_bits:
        kernel_correctness = attention_microbench(bundle, profile)
        write_json(output / "quantized-attention-correctness.json", kernel_correctness)
        if not kernel_correctness["within_tolerance"]:
            quality["status"] = "failed"
            quality["reason"] = "quantized kernel differs from dequantized-page oracle"
    if not profile.kv_bits and parity["status"] == "failed":
        quality["status"] = "failed"
        quality["reason"] = "floating execution must pass full logits and state parity"
    parity["status"] = quality["status"]
    scope = parity["runtime"]["scope"]
    scope["hardware"], scope["source_revision"] = host_identity(), source_identity()
    write_json(output / "parity.json", parity)
    write_json(output / "retrieval.json", retrieval)
    write_json(output / "quality.json", quality)
    write_json(output / "cleanup.json", parity["cleanup"])
    write_json(output / "summary.json", {"status": quality["status"], "profile": profile.to_dict(),
                                          "quality": quality, "qualification_promoted": False})
    from mlx_one.engine.paged import KERNEL_VERSION

    qualification_evidence(
        output, feature="quantized_paged_kv" if profile.kv_bits else "long_context_quality",
        scope=scope, parity=parity,
        compatibility_digest=compatibility_digest(engine._cache_runtime),
        workload_envelope=workload_envelope(profile), kernel_version=KERNEL_VERSION,
        kernel_correctness_passed=(kernel_correctness is not None
                                   and kernel_correctness["within_tolerance"]),
        quality_passed=quality["status"] == "passed",
        quality_kind=parity["quality_kind"],
        quality_coverage_passed=parity.get("coverage", {}).get("eligible", False),
        dataset_identity=parity.get("dataset_identity"))
    if quality["status"] == "failed":
        raise ValueError("declared quantized quality gates failed")


def advanced_checks(bundle, engine, scheduler, profile, summary, output):
    """Finish parity and gates; return proof fields without promoting anything."""
    from mlx_one.engine.qualification import execution_parity
    from mlx_one.engine.reporting import native_report
    from mlx_one.engine.selection import compatibility_digest, host_identity, source_identity
    from mlx_one.inference_baseline_worker import write_json
    from mlx_one.inference_qualification import (
        evaluate_gate_d,
        evaluate_gate_e,
        evaluate_gate_f,
        evaluate_quality,
        load_quality_reference,
        load_reference,
        workload_envelope,
    )

    runtime = engine._cache_runtime
    prompt = "A shared document with stable numbered facts. " * 64
    parity = scheduler.execute(lambda: execution_parity(
        bundle, mode=profile.inference_mode, prompt=prompt, count=32, batch_size=profile.slots,
        chunk_size=profile.chunk_tokens, cache_type=profile.cache_type,
        page_tokens=profile.page_tokens, atol=profile.atol, rtol=profile.rtol,
        quality_limits=profile.quality_limits))
    quality = None
    if profile.kv_bits:
        correctness = scheduler.execute(lambda: attention_microbench(bundle, profile))
        write_json(output / "quantized-attention-correctness.json", correctness)
        retrieval = scheduler.execute(lambda: retrieval_quality(bundle, profile))
        write_json(output / "retrieval.json", retrieval)
        quality = evaluate_quality(parity, retrieval, profile.quality_limits)
        if not correctness["within_tolerance"]:
            quality["status"] = "failed"
        write_json(output / "quality.json", quality)
        parity["status"] = quality["status"]
    write_json(output / "parity.json", parity)
    summary["prefix_isolation_passed"] = parity.get("prefix_isolation_passed", False)
    write_json(output / "summary.json", summary)
    trace_path = output / "forward-traces.jsonl"
    traces = ([json.loads(line) for line in trace_path.read_text().splitlines()]
              if trace_path.exists() else [])
    reference = load_reference(profile.reference_run) if profile.reference_run else None
    gates = {}
    if profile.inference_mode == "paged" and profile.workload != "soak":
        gates["d"] = evaluate_gate_d(profile, summary, traces, parity, reference)
    if profile.inference_mode == "experimental_hybrid_batch":
        gates["e"] = evaluate_gate_e(parity, traces)
        if profile.workload != "soak":
            gates["hybrid-performance"] = evaluate_gate_d(profile, summary, traces,
                                                         parity, reference)
    gates["f"] = evaluate_gate_f(profile, summary, engine.cache_stats())
    if profile.kv_bits and profile.quality_limits is None:
        gates["f"] = {"status": "diagnostic", "reason": "quantized quality limits are required"}
    for name, gate in gates.items():
        write_json(output / f"gate-{name}.json", gate)
    scope = native_report(bundle, runtime, mode=profile.inference_mode, hardware=host_identity(),
                          source_revision=source_identity())["scope"]
    from mlx_one.engine.paged import KERNEL_VERSION

    metadata = {"compatibility_digest": compatibility_digest(runtime),
                "kernel_version": KERNEL_VERSION if profile.inference_mode == "paged" else None,
                "workload_envelope": workload_envelope(profile),
                "quality_passed": quality is not None and quality["status"] == "passed",
                "kernel_correctness_passed": (profile.kv_bits is not None
                                               and correctness["within_tolerance"]),
                "memory_passed": gates["f"]["status"] == "passed",
                "service_passed": any(g.get("status") == "passed" for g in gates.values())}
    if profile.context_tokens + profile.output_tokens >= 128000 and profile.quality_run:
        quality_reference = load_quality_reference(profile.quality_run, profile=profile,
                                                    runtime=runtime)
        write_json(output / "quality-reference.json", quality_reference)
        metadata["quality_passed"] = True
        metadata["service_passed"] = gates["f"]["status"] == "passed"
    if any(g["status"] == "failed" for g in gates.values()) or parity["status"] == "failed":
        raise ValueError("declared optimized qualification gates failed")
    return parity, gates, scope, metadata
