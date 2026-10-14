"""Forced-token cache/batch parity and explicit, evidence-only gate evaluation."""

from __future__ import annotations

import random
from dataclasses import replace

from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.parity import compare_logits
from mlx_one.text.sampling import select_token
from mlx_one.text.schemas import TextGenerationOptions


def logit_nll(logits, token):
    import mlx.core as mx
    import numpy as np

    value = logits.astype(mx.float32) if hasattr(mx, "float32") else logits
    row = np.asarray(value, dtype=np.float64)[0, -1]
    maximum = row.max()
    return float(maximum + np.log(np.exp(row - maximum).sum()) - row[token])


def execution_parity(bundle, *, mode, prompt="The quick brown fox", count=32, batch_size=2,
                     chunk_size=512, atol=None, rtol=None, seed=17, cache_type="f16",
                     page_tokens=32, quality_limits=None):
    """Matched FP16 KV oracle, divergent prompts, forced IDs, and prefix adoption."""
    import mlx.core as mx

    if mode not in {"experimental_block", "experimental_dense_batch", "paged",
                    "experimental_hybrid_batch"}:
        raise ValueError("execution parity requires an optimized mode")
    if min(count, batch_size, chunk_size) < 1 or (atol is None) != (rtol is None):
        raise ValueError("invalid parity configuration")
    atol, rtol = ((5e-2, 1e-2) if cache_type == "bf16" else (1e-2, 1e-2)) if atol is None else (
        atol, rtol)
    batch_size = 1 if mode == "experimental_block" else batch_size
    oracle = CacheManager(bundle, CacheConfig(prefix_cache_enabled=False))
    backend = {"experimental_block": "block", "experimental_dense_batch": "dense",
               "paged": "paged", "experimental_hybrid_batch": "hybrid"}[mode]
    target = CacheManager(bundle, CacheConfig(backend=backend, block_size_tokens=page_tokens,
                                             key_type=cache_type, value_type=cache_type))
    target.harness_only = True
    target.mode = mode
    handles, references, captures, rows = [], [], [], []
    from mlx_one.engine.hybrid import HybridRunner
    from mlx_one.engine.paged import PagedRunner

    runner = ({"paged": PagedRunner, "experimental_hybrid_batch": HybridRunner}.get(
        mode, BatchedDenseRunner))()
    if target.hybrid_state:
        runner = HybridRunner()
    prompts = [tuple(bundle.tokenizer.encode(prompt + ("\nMore." * i))) for i in range(batch_size)]
    fixture = bundle.tokenizer.encode(" A deterministic continuation.")
    forced = tuple(fixture[i % len(fixture)] for i in range(count))
    fp = target.fingerprint(namespace="qualification", key_type=cache_type, value_type=cache_type)
    nll_deltas, component_rows = [], []
    def create_target(name):
        handle = target.create_handle(name, key_type=cache_type, value_type=cache_type)
        handle.metadata["fingerprint"] = fp
        return handle

    def forward(requests):
        if mode != "experimental_block":
            for request in requests:
                request.runtime = target
                request.handle = next(h for h in handles if h.bundle.entries is request.cache)
            if target.hybrid_state and requests[0].phase == "prefill":
                results = [None] * len(requests)
                for start in range(0, max(len(r.token_ids) for r in requests), page_tokens):
                    active, indices = [], []
                    for i, request in enumerate(requests):
                        part = request.token_ids[start:start + page_tokens]
                        if part:
                            active.append(replace(request, token_ids=part))
                            indices.append(i)
                    for index, logits in zip(indices, runner.execute(tuple(active)).logits,
                                             strict=True):
                        results[index] = logits
                return tuple(results)
            return runner.execute(tuple(requests)).logits
        request = requests[0]
        staged = CacheBundle(request.cache).clone()
        try:
            result = bundle.model(mx.array([request.token_ids]), cache=staged.entries)
            mx.eval(result.logits)
            for original, draft in zip(request.cache, staged.entries, strict=True):
                original.replace_from(draft)
            return (result.logits[:, -1:, :],)
        finally:
            staged.release()
    def dense_state(cache):
        if callable(getattr(cache, "state", None)):
            return cache.state()
        return (cache._materialize(cache._keys, cache.key_bits),
                cache._materialize(cache._values, cache.value_bits))

    def compare_components(index):
        for layer, (actual_cache, reference_cache) in enumerate(zip(
                handles[index].bundle.entries, references[index].bundle.entries, strict=True)):
            if hasattr(actual_cache, "_blocks"):
                expected = dense_state(reference_cache)
                start = 0
                for ref in actual_cache._blocks:
                    block = ref.block
                    if len(block.state) != 2:
                        from mlx_one.engine.paged import dequantize_page

                        actual = (dequantize_page(block.state[:3], block.layout.quantization_bits,
                                                  block.layout.head_dim),
                                  dequantize_page(block.state[3:], block.layout.quantization_bits,
                                                  block.layout.head_dim))
                    else:
                        actual = block.state
                    for left, right in zip(actual, expected, strict=True):
                        component_rows.append({"request": index, "layer": layer,
                            "committed_position": actual_cache.offset,
                            **compare_logits(right[:, :, start:start + block.token_count],
                                             left[:, :, :block.token_count], atol=atol, rtol=rtol)})
                    start += block.token_count
            elif hasattr(actual_cache, "kernel_size"):
                component_rows.append({"request": index, "layer": layer, "kind": "conv",
                    "committed_position": actual_cache.offset,
                    **compare_logits(reference_cache.values, actual_cache.values,
                                     atol=atol, rtol=rtol)})
            else:
                expected = dense_state(reference_cache)
                actual = dense_state(actual_cache)
                for left, right in zip(actual, expected, strict=True):
                    if getattr(actual_cache, "max_size", None):
                        right = right[:, :, -left.shape[2]:]
                    component_rows.append({"request": index, "layer": layer,
                        "committed_position": actual_cache.offset,
                        **compare_logits(right, left, atol=atol, rtol=rtol)})
    try:
        for i, ids in enumerate(prompts):
            handle = create_target(f"target-{i}")
            reference = oracle.create_handle(f"oracle-{i}",
                                             key_type="bf16" if cache_type == "bf16" else "f16",
                                             value_type="bf16" if cache_type == "bf16" else "f16")
            handles.append(handle)
            references.append(reference)
            output = bundle.model(mx.array([ids]), cache=reference.bundle.entries)
            mx.eval(output.logits)
            captures.append(output.logits[:, -1:, :])
        offsets = [0] * batch_size
        actual = [None] * batch_size
        while any(offset < len(ids) for offset, ids in zip(offsets, prompts, strict=True)):
            requests, indices = [], []
            for i, ids in enumerate(prompts):
                if offsets[i] < len(ids):
                    part = ids[offsets[i]:offsets[i] + chunk_size]
                    requests.append(ForwardRequest(bundle, part, handles[i].bundle.entries,
                                                   "prefill", ids, request_id=f"target-{i}"))
                    indices.append(i)
            for i, logits, request in zip(indices, forward(requests), requests, strict=True):
                offsets[i] += len(request.token_ids)
                actual[i] = logits
        for expected, logits in zip(captures, actual, strict=True):
            rows.append(compare_logits(expected, logits, atol=atol, rtol=rtol))
        for index in range(batch_size):
            compare_components(index)
        replay_logits = [captures[0][:, -1, :]]
        for step, token in enumerate(forced):
            requests = []
            for i, (handle, reference) in enumerate(zip(handles, references, strict=True)):
                # Drop one peer halfway; another request's state must remain independent.
                if i and step == count // 2:
                    target.release(handle)
                if handle.closed:
                    continue
                output = bundle.model(mx.array([[token]]), cache=reference.bundle.entries)
                mx.eval(output.logits)
                captures[i] = output.logits[:, -1:, :]
                requests.append(ForwardRequest(bundle, (token,), handle.bundle.entries,
                                               "decode", prompts[i], request_id=f"target-{i}"))
            for request, logits in zip(requests, forward(requests), strict=True):
                i = int(request.request_id.rsplit("-", 1)[-1])
                rows.append(compare_logits(captures[i], logits, atol=atol, rtol=rtol))
                next_token = forced[(step + 1) % len(forced)]
                nll_deltas.append(logit_nll(logits, next_token)
                                  - logit_nll(captures[i], next_token))
            replay_logits.append(captures[0][:, -1, :])
            for index in range(batch_size):
                if not handles[index].closed:
                    compare_components(index)
        # A fresh prompt snapshot exercises identical and divergent-suffix restoration.
        donor = create_target("prefix-donor")
        handles.append(donor)
        ids = prompts[0]
        forward([ForwardRequest(bundle, ids, donor.bundle.entries, "prefill", ids,
                                request_id="prefix-donor")])
        published = target.publish_prefix(donor, fp, ids)
        if target.hybrid_state:
            published = target.hybrid_prefix.preview(fp, ids + tuple(fixture[:3])) > 0
        dense_donor = None
        if not target.hybrid_state:
            dense_donor = oracle.create_handle("independent-prefix-donor",
                key_type="bf16" if cache_type == "bf16" else "f16",
                value_type="bf16" if cache_type == "bf16" else "f16")
            references.append(dense_donor)
            mx.eval(bundle.model(mx.array([ids]), cache=dense_donor.bundle.entries).logits)
        prefix_rows = []
        for suffix in ((), tuple(fixture[:3])):
            full = ids + suffix
            borrower = create_target("prefix-borrower")
            handles.append(borrower)
            reused = target.adopt_prefix(borrower, fp, full)
            expected_cache = oracle.create_handle(
                "prefix-oracle", key_type="bf16" if cache_type == "bf16" else "f16",
                value_type="bf16" if cache_type == "bf16" else "f16")
            references.append(expected_cache)
            expected = bundle.model(mx.array([full]), cache=expected_cache.bundle.entries)
            logits = forward([ForwardRequest(bundle, full[reused:], borrower.bundle.entries,
                                             "prefill", full, request_id="prefix-borrower")])[0]
            comparison = {"reused_tokens": reused, "avoided_prefill_tokens": reused,
                          **compare_logits(expected.logits[:, -1:, :], logits,
                                           atol=atol, rtol=rtol)}
            if dense_donor is not None:
                # Distinguish ownership/restore errors from shape-dependent native
                # full-prefill versus cached-forward arithmetic. The full oracle
                # comparison above remains the acceptance gate.
                staged = dense_donor.bundle.clone()
                try:
                    if not staged.trim(len(ids) - reused):
                        raise ValueError("independent dense snapshot cannot trim")
                    cached = bundle.model(mx.array([full[reused:]]), cache=staged.entries)
                    mx.eval(cached.logits)
                    comparison["cached_single_request"] = compare_logits(
                        cached.logits[:, -1:, :], logits, atol=atol, rtol=rtol)
                    comparison["full_vs_cached_single_request"] = compare_logits(
                        expected.logits[:, -1:, :], cached.logits[:, -1:, :],
                        atol=atol, rtol=rtol)
                finally:
                    staged.release()
            prefix_rows.append(comparison)
            target.release(borrower)
        isolation = {}
        for field in ("namespace", "model_identity", "adapter_identity", "tokenizer_identity",
                      "template_identity", "special_tokens_identity", "rope_identity",
                      "rendering_identity", "revision", "multimodal_identity"):
            borrower = create_target("negative-prefix")
            handles.append(borrower)
            try:
                from mlx_one.engine.cache_errors import PrefixReuseUnsupported

                try:
                    reused = target.adopt_prefix(borrower, replace(fp, **{field: "different"}), ids)
                except PrefixReuseUnsupported:
                    reused = 0
                isolation[field] = reused == 0
            finally:
                target.release(borrower)
        rngs = {name: random.Random(seed) for name in ("oracle", "a", "b")}
        replay = {name: [] for name in rngs}
        options = TextGenerationOptions(temperature=.7, seed=seed)
        for index, logits in enumerate(replay_logits):
            for name in (("a", "oracle", "b") if index % 2 else ("b", "oracle", "a")):
                replay[name].append(select_token(logits[0], options, rngs[name]))
        seeded = replay["a"] == replay["b"] == replay["oracle"]
        prefix_passed = bool(published) and all(r["reused_tokens"] > 0
                                               and r["within_tolerance"]
                                               and r["greedy_agreement"] for r in prefix_rows)
        passed = (all(r["within_tolerance"] and r["greedy_agreement"] for r in rows)
                  and all(r["within_tolerance"] for r in component_rows)
                  and seeded and prefix_passed and all(isolation.values()))
        from mlx_one.engine.reporting import native_report

        result = {"status": "passed" if passed else "failed", "execution_mode": mode,
                "oracle_path": ("single_request_dense_bf16_kv" if cache_type == "bf16"
                                else "single_request_dense_fp16_kv"), "batch_size": batch_size,
                "forced_steps": count, "prefill_chunk_tokens": chunk_size,
                "atol": atol, "rtol": rtol,
                "max_error": max(r["max_error"] for r in [*rows, *prefix_rows]),
                "mean_error": sum(r["mean_error"] for r in rows) / len(rows),
                "greedy_agreement_fraction": sum(r["greedy_agreement"] for r in rows) / len(rows),
                "comparisons": rows, "prefix_restoration": prefix_rows,
                "component_comparisons": component_rows,
                "prefix_isolation": isolation, "prefix_isolation_passed": all(isolation.values()),
                "held_out_nll_mean_delta": None,
                "quality_kind": "synthetic-smoke-test",
                "synthetic_replay_nll_mean_delta": sum(nll_deltas) / len(nll_deltas),
                "seeded_rng_isolation": seeded,
                "cancelled_request_isolated": (all(h.closed for h in handles[1:batch_size])
                                                if batch_size > 1 else None),
                "cache_accounting": target.stats().to_dict(),
                "runtime": native_report(bundle, target, backend=target.config.backend, mode=mode)}
        if cache_type in {"q4_0", "q8_0"}:
            # Quality additionally needs measured retrieval; no tolerant-greedy shortcut.
            result["status"] = "diagnostic"
            result["quality_limits"] = quality_limits
        return result
    finally:
        for handle in handles:
            target.release(handle)
        for handle in references:
            oracle.release(handle)
        target.close()
        oracle.close()
        if "result" in locals():
            result["cleanup"] = target.stats().to_dict()
            if result["cleanup"]["active_handles"] or result["cleanup"]["reserved_bytes"]:
                result["status"] = "failed"


def evaluate_gate_c(profile, summary, traces, parity, reference=None):
    """Evaluate a declared scope; this function never promotes registry records."""
    budgets = (profile.ttft_budget_ms, profile.decode_gap_budget_ms)
    if any(value is None for value in budgets):
        return {"status": "diagnostic", "reason": "explicit latency budgets are required",
                "qualification_promoted": False}
    if reference is None:
        return {"status": "pending", "reason": "matched reference evidence is required",
                "qualification_promoted": False}
    expected = replace(profile, inference_mode="reference_dense", reference_run=None)
    reference_profile = reference.get("profile", {})
    for field in ("model_id", "revision", "context_tokens", "slots", "output_tokens", "repeats",
                  "seed", "workload", "cache_type", "execution_dtype"):
        if reference_profile.get(field) != getattr(expected, field):
            raise ValueError(f"reference evidence differs in {field}")
    for field in ("hardware", "source", "weight_precision", "cache_precision", "mlx_version"):
        if reference.get("identity", {}).get(field) != summary.get("identity", {}).get(field):
            raise ValueError(f"reference evidence differs in {field}")
        if not reference.get("identity", {}).get(field):
            raise ValueError(f"reference identity missing {field}")
    rates = summary.get("aggregate_output_tokens_per_second_by_group", [])
    baseline = reference.get("summary", {}).get("aggregate_output_tokens_per_second_by_group", [])
    improvement = (len(rates) == len(baseline) == profile.repeats
                   and all(a is not None and b is not None and a > b
                           for a, b in zip(rates, baseline, strict=True)))
    memberships = {tuple(t["request_ids"]) for t in traces if t["phase"] == "decode"}
    ttft, gap = summary.get("warm_client_ttft_p95_ms"), summary.get("model_token_gap_p95_ms")
    checks = {"parity": parity.get("status") == "passed",
              "actual_batched_decode": any(t["phase"] == "decode" and t["batch_size"] > 1
                                           for t in traces),
              "dynamic_membership": len(memberships) > 1,
              "latency_budget": ttft is not None and gap is not None
                                and ttft <= budgets[0] and gap <= budgets[1],
              "repeatable_throughput_improvement": improvement,
              "no_failures": summary.get("failed") == 0,
              "fairness": summary.get("starved_requests") == 0}
    if profile.workload == "cancellation":
        checks["observed_cancellation"] = (
            summary.get("expected_cancellations", 0) > 0
            and summary.get("observed_cancellations", 0) >= summary["expected_cancellations"])
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "qualification_promoted": False}
