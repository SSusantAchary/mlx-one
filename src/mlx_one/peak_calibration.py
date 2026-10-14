"""Synchronized incremental forward peaks, separate from cumulative telemetry."""
from __future__ import annotations

import math

CALIBRATION_POLICY = "activation-peak-v1-margin20"


def summarize_peaks(samples):
    if {s.get("temperature") for s in samples} != {"cold", "warm"} or any(
            sum(s.get("temperature") == phase for s in samples) != 3
            for phase in ("cold", "warm")):
        raise ValueError("peak calibration requires three cold and three warm samples")
    for sample in samples:
        if any(sample.get(k) is None or sample[k] < 0 for k in (
                "active_before", "active_after", "peak_active", "cached_before", "cached_after",
                "persistent_before", "persistent_after")):
            raise ValueError("incremental peak telemetry is incomplete")
        growth = max(sample["persistent_after"] - sample["persistent_before"], 0)
        sample["persistent_growth"] = growth
        sample["transient_increment"] = max(sample["peak_active"] - sample["active_before"]
                                              - growth, 0)
    maximum = max(s["transient_increment"] for s in samples)
    return {"policy": CALIBRATION_POLICY, "samples": samples,
            "observed_max_incremental_bytes": maximum,
            "incremental_bytes": math.ceil(maximum * 1.2)}


def native_peak_calibration(bundle, profile):
    import gc

    import mlx.core as mx

    from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.hybrid import HybridRunner
    from mlx_one.engine.paged import PagedRunner

    required = ("synchronize", "reset_peak_memory", "get_peak_memory", "get_active_memory",
                "get_cache_memory", "clear_cache")
    if not all(callable(getattr(mx, name, None)) for name in required):
        return {"status": "pending", "reason": "incremental peak APIs unavailable"}
    backend = {"paged": "paged", "experimental_hybrid_batch": "hybrid",
               "experimental_dense_batch": "dense"}[profile.inference_mode]
    runner = {"paged": PagedRunner, "hybrid": HybridRunner, "dense": BatchedDenseRunner}[backend]()
    phase = profile.calibration_phase
    chunk = 1 if phase == "decode" else profile.chunk_tokens
    history = profile.context_tokens - (chunk if phase == "prefill" else 0)
    if history < 0:
        raise ValueError("calibration chunk exceeds context")
    seed = tuple(bundle.tokenizer.encode("Memory calibration. " * 256))
    tokens = tuple(seed[i % len(seed)] for i in range(history))
    samples = []
    for temperature in ("cold", "warm"):
        for repetition in range(3):
            runtime = CacheManager(bundle, CacheConfig(backend=backend,
                prefix_cache_enabled=False, block_size_tokens=profile.page_tokens,
                key_type=profile.cache_type, value_type=profile.cache_type,
                memory_budget_bytes=profile.cache_capacity_bytes))
            if runtime.hybrid_state and backend == "paged":
                runner = HybridRunner()
            runtime.harness_only, runtime.mode = True, profile.inference_mode
            handles = [runtime.create_handle(str(i), key_type=profile.cache_type,
                        value_type=profile.cache_type) for i in range(profile.slots)]
            def request(handle, ids, kind, runtime=runtime):
                return ForwardRequest(bundle, ids, handle.bundle.entries, kind, tokens,
                                      runtime, handle, request_id=handle.request_id)
            try:
                for start in range(0, history, profile.chunk_tokens):
                    runner.execute(tuple(request(h, tokens[start:start + profile.chunk_tokens],
                                                 "prefill") for h in handles))
                if temperature == "warm":
                    # Warm the dispatch at identical dimensions on isolated clones.
                    clones = [runtime.create_handle("warm-" + h.request_id,
                              key_type=profile.cache_type, value_type=profile.cache_type)
                              for h in handles]
                    for original, clone in zip(handles, clones, strict=True):
                        clone.bundle.release()
                        clone.bundle = original.bundle.clone()
                    ids = tuple(seed[i % len(seed)] for i in range(chunk))
                    runner.execute(tuple(request(h, ids, phase) for h in clones))
                    for clone in clones:
                        runtime.release(clone)
                gc.collect()
                if temperature == "cold":
                    runtime.block_pool.discard_free()
                    mx.clear_cache()
                mx.synchronize()
                sample = {"temperature": temperature, "repetition": repetition,
                          "active_before": int(mx.get_active_memory()),
                          "cached_before": int(mx.get_cache_memory()),
                          "persistent_before": runtime.stats().physical_bytes}
                mx.reset_peak_memory()
                ids = tuple(seed[i % len(seed)] for i in range(chunk))
                result = runner.execute(tuple(request(h, ids, phase) for h in handles))
                mx.eval(*result.logits)
                mx.synchronize()
                sample.update(active_after=int(mx.get_active_memory()),
                              cached_after=int(mx.get_cache_memory()),
                              peak_active=int(mx.get_peak_memory()),
                              persistent_after=runtime.stats().physical_bytes,
                              replacement_overlap=runtime.block_pool.stats().get(
                                  "replacement_overlap_peak_bytes", 0))
                actual = runtime.state_observations()
                actual_dtypes = sorted({d for layer in actual if layer["kind"] in {
                    "full_kv", "sliding_kv"} for d in layer["actual_dtypes"]})
                samples.append(sample)
            finally:
                runtime.close()
    result = summarize_peaks(samples)
    result.update(status="passed", actual_cache_dtypes=actual_dtypes,
                  dimensions={"batch": profile.slots, "chunk": chunk,
        "history": history, "history_lengths": [history] * profile.slots,
        "widths": [chunk] * profile.slots, "phase": phase})
    return result
