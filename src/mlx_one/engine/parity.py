"""Opt-in forced-token differential checks against native single-request generation."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import ExitStack
from dataclasses import replace
from typing import Any

from mlx_one.engine.contracts import BatchPlan, EngineRequest, Phase, RequestKind, ScheduledSequence
from mlx_one.engine.reporting import describe_cache, normalize_dtype
from mlx_one.engine.runners import CausalLMRunner
from mlx_one.inference_baseline import TOLERANCES
from mlx_one.text.generation import _stream_bundle
from mlx_one.text.sampling import select_token
from mlx_one.text.schemas import TextGenerationOptions


def compare_logits(reference, actual, *, atol, rtol, mx=None) -> dict[str, Any]:
    if mx is None:
        import mlx.core as mx
    if reference.shape != actual.shape:
        raise ValueError("parity logits shapes differ")
    reference = reference.astype(mx.float32)
    actual = actual.astype(mx.float32)
    error = mx.abs(reference - actual)
    return {
        "max_error": float(mx.max(error).item()), "mean_error": float(mx.mean(error).item()),
        "within_tolerance": bool(mx.all(error <= atol + rtol * mx.abs(reference)).item()),
        "greedy_agreement": bool(mx.all(mx.argmax(reference, axis=-1)
                                        == mx.argmax(actual, axis=-1)).item()),
    }


def prefix_failure_diagnostic(bundle, *, prompt=None, count=32, sink=None):
    """Four independent B1 paths; a reproduced mismatch remains failed qualification.

    This only observes existing cache/runner arithmetic. It does not repair it,
    loosen tolerances, sample tokens, or replace the fresh-full-prefill oracle.
    """
    import mlx.core as mx

    from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.cache_specs import CacheConfig

    if count < 1:
        raise ValueError("prefix diagnostic needs positive forced steps")
    prompt = prompt if prompt is not None else (
        "Native INT8 shared prefix and divergent suffixes. " * 32)
    ids = tuple(bundle.tokenizer.encode(prompt))
    fixture = tuple(bundle.tokenizer.encode(" A deterministic continuation."))
    if len(ids) < 2 or not fixture:
        raise ValueError("prefix diagnostic needs a reusable prompt and continuation")
    forced = tuple(fixture[i % len(fixture)] for i in range(count))
    atol, rtol = TOLERANCES["float16"]
    config = CacheConfig(key_type="f16", value_type="f16")
    managers, isolated, cases = [], [], []
    result = None

    def entries(cache):
        return cache.bundle.entries if hasattr(cache, "bundle") else cache.entries

    def tensors(cache):
        if callable(getattr(cache, "state", None)):
            return cache.state()
        return (cache._materialize(cache._keys, cache.key_bits, cache.offset),
                cache._materialize(cache._values, cache.value_bits, cache.offset))

    def direct(cache, tokens):
        output = bundle.model(mx.array([tokens]), cache=entries(cache))
        mx.eval(output.logits)
        return output.logits[:, -1:, :]

    def metadata(cache, plan):
        rows = describe_cache(entries(cache), plan)
        for row, layer in zip(rows, entries(cache), strict=True):
            row["tensor_shapes"] = [list(t.shape) for t in tensors(layer)]
        return rows

    def state_error(reference, actual):
        rows = []
        for layer, (a, b) in enumerate(zip(entries(reference), entries(actual), strict=True)):
            for component, (x, y) in enumerate(zip(tensors(a), tensors(b), strict=True)):
                rows.append({"layer": layer, "component": "key" if component == 0 else "value",
                             **compare_logits(x, y, atol=atol, rtol=rtol, mx=mx)})
        return rows

    try:
        with ExitStack() as lifetime:
            oracle = CacheManager(bundle, replace(config, prefix_cache_enabled=False))
            managers.append(oracle)
            lifetime.callback(oracle.close)
            target = CacheManager(bundle, config)
            managers.append(target)
            lifetime.callback(target.close)
            runner = BatchedDenseRunner(bundle)
            fingerprint = target.fingerprint(namespace="pr00-native-prefix",
                                             key_type="f16", value_type="f16")
            donor = target.create_handle("prefix-donor", key_type="f16", value_type="f16")
            donor.metadata["fingerprint"] = fingerprint
            native_donor = oracle.create_handle("native-donor", key_type="f16", value_type="f16")
            direct(native_donor, ids)

            def indexed(handle, tokens, phase="prefill"):
                return runner.execute((ForwardRequest(bundle, tuple(tokens), handle.bundle.entries,
                    phase, ids, runtime=target, handle=handle,
                    request_id=handle.request_id),)).logits[0]

            indexed(donor, ids)
            if not target.publish_prefix(donor, fingerprint, ids):
                raise ValueError("diagnostic donor prefix was not published")
            for case_index, suffix in enumerate(((), fixture[:3])):
                full = ids + suffix
                with ExitStack() as lifetime_case:
                    borrower = target.create_handle(f"borrower-{case_index}")
                    lifetime_case.callback(target.release, borrower)
                    reused = target.adopt_prefix(borrower, fingerprint, full)
                    if not 0 < reused < len(full):
                        raise ValueError("prefix diagnostic must reuse tokens and execute a suffix")
                    fresh = oracle.create_handle(f"fresh-{case_index}")
                    lifetime_case.callback(oracle.release, fresh)
                    cached = oracle.create_handle(f"cached-{case_index}")
                    lifetime_case.callback(oracle.release, cached)
                    restored = native_donor.bundle.clone()
                    isolated.append(restored)
                    lifetime_case.callback(restored.release)
                    if not restored.trim(len(ids) - reused):
                        raise ValueError("independent donor snapshot could not trim")
                    direct(cached, full[:reused])
                    caches = {"fresh_full": fresh, "ordinary_cached": cached,
                              "independent_restored": restored, "prefix_index": borrower}
                    logits = {"fresh_full": direct(fresh, full),
                              "ordinary_cached": direct(cached, full[reused:]),
                              "independent_restored": direct(restored, full[reused:]),
                              "prefix_index": indexed(borrower, full[reused:])}
                    case = {"case": case_index, "prompt_ids": list(full),
                            "reused_tokens": reused, "executed_suffix_ids": list(full[reused:]),
                            "boundaries": []}
                    cases.append(case)
                    for step in range(count + 1):
                        if step:
                            token = forced[step - 1]
                            logits = {name: indexed(cache, (token,), "decode")
                                      if name == "prefix_index" else direct(cache, (token,))
                                      for name, cache in caches.items()}
                        comparisons = {name: compare_logits(logits["fresh_full"], value,
                            atol=atol, rtol=rtol, mx=mx) for name, value in logits.items()
                            if name != "fresh_full"}
                        agreement = compare_logits(logits["independent_restored"],
                            logits["prefix_index"], atol=atol, rtol=rtol, mx=mx)
                        state = {name: metadata(cache, oracle.plan)
                                 for name, cache in caches.items()}
                        aligned = all(row["committed_tokens"] == len(full) + step
                                      for layers in state.values() for row in layers)
                        boundary = {"step": step,
                            "forced_input_id": forced[step - 1] if step else None,
                            "fresh_comparisons": comparisons, "restored_vs_index": agreement,
                            "committed_positions_aligned": aligned, "components": state,
                            "greedy_ids": {name: int(mx.argmax(value, axis=-1).item())
                                           for name, value in logits.items()}}
                        if step in {0, count}:
                            boundary["component_comparisons"] = {
                                name: state_error(fresh, cache) for name, cache in caches.items()
                                if name != "fresh_full"}
                            boundary["restored_vs_index_components"] = state_error(
                                restored, borrower)
                        case["boundaries"].append(boundary)
                        if sink:
                            for name, value in logits.items():
                                sink(f"case{case_index}.{name}.step{step}", value)
            boundaries = [r for case in cases for r in case["boundaries"]]
            passed = all(r["committed_positions_aligned"] and all(
                c["within_tolerance"] and c["greedy_agreement"]
                for c in r["fresh_comparisons"].values()) for r in boundaries)
            passed &= all(c["within_tolerance"] for r in boundaries for comparisons in
                          r.get("component_comparisons", {}).values() for c in comparisons)
            reproduced = any(not case["boundaries"][0]["fresh_comparisons"]["prefix_index"][
                "within_tolerance"] for case in cases) and all(
                    r["restored_vs_index"]["within_tolerance"] and r["committed_positions_aligned"]
                    for r in boundaries)
            reproduced &= all(c["within_tolerance"] for r in boundaries
                              for c in r.get("restored_vs_index_components", []))
            result = {"schema_version": "native-prefix-diagnostic-v1",
                "status": "passed" if passed else "failed", "failure_reproduced": reproduced,
                "atol": atol, "rtol": rtol,
                "tolerance_source": "inference_baseline.TOLERANCES.float16",
                "forced_token_ids": list(forced), "forced_steps": count, "cases": cases,
                "oracle": "fresh_full_prefill", "qualification_promoted": False}
    finally:
        if result is not None:
            result["cleanup"] = {"reference": managers[0].stats().to_dict(),
                                 "target": managers[1].stats().to_dict(),
                                 "isolated_snapshots_disposed": all(c.closed for c in isolated)}
            result["cleanup_passed"] = (result["cleanup"]["isolated_snapshots_disposed"]
                and all(m.stats().active_handles == m.stats().reserved_bytes == 0
                        for m in managers))
            if not result["cleanup_passed"]:
                result["status"], result["failure_reproduced"] = "failed", False
    return result


def forced_token_parity(bundle, *, prompt="The quick brown fox", count=32,
                        atol=None, rtol=None, seed=17, prefill_chunk_tokens=7) -> dict[str, Any]:
    if count < 3 or prefill_chunk_tokens < 1:
        raise ValueError("parity requires at least three steps and a positive prefill chunk")
    if (atol is None) != (rtol is None):
        raise ValueError("override both atol and rtol")
    import mlx.core as mx

    captures = []

    class CaptureModel:
        def __getattr__(self, name):
            return getattr(bundle.model, name)

        def __call__(self, *args, **kwargs):
            output = bundle.model(*args, **kwargs)
            mx.eval(output.logits)
            captures.append(output.logits[:, -1, :])
            return output

    fixture = bundle.tokenizer.encode(" A deterministic continuation with separate request state.")
    if not fixture:
        raise ValueError("parity fixture must tokenize to nonempty IDs")
    forced = tuple(fixture[index % len(fixture)] for index in range(count))
    oracle_cache = tuple(bundle.model.make_cache())
    list(_stream_bundle(
        replace(bundle, model=CaptureModel()), prompt,
        TextGenerationOptions(max_tokens=count), None, cache=oracle_cache,
        _forced_token_ids=forced,
    ))
    if len(captures) != count:
        raise ValueError("oracle stopped before the forced continuation completed")
    dtype = normalize_dtype(captures[0].dtype)
    default = TOLERANCES.get(dtype)
    if default is None:
        raise ValueError(f"no versioned parity tolerance for {dtype}")
    atol, rtol = (default if atol is None else (atol, rtol))
    runner = CausalLMRunner(bundle)
    state = runner.prepare(EngineRequest(RequestKind.TEXT, bundle.model_id, {"prompt": prompt}))
    survivor = runner.prepare(EngineRequest(RequestKind.TEXT, bundle.model_id, {"prompt": prompt}))
    results = []

    def step(sequence, phase, amount):
        return runner.prefill(BatchPlan((ScheduledSequence(
            sequence.request_id, phase, sequence.computed_len, amount
        ),), amount)) if phase == "prefill" else runner.decode(BatchPlan((ScheduledSequence(
            sequence.request_id, phase, sequence.computed_len, amount
        ),), amount))

    try:
        for sequence in (state, survivor):
            sequence.transition(Phase.PREFILL)
            # Chunk boundaries differ from the oracle, exercising exact carry state.
            output = None
            while sequence.remaining_prompt_tokens:
                output = step(sequence, "prefill", min(
                    prefill_chunk_tokens, sequence.remaining_prompt_tokens
                ))
            sequence.transition(Phase.DECODE)
            results.append(compare_logits(captures[0], output.logits[0][:, -1, :],
                                          atol=atol, rtol=rtol, mx=mx))
        for index, token in enumerate(forced[:-1]):
            # Cancel one request midway; the surviving hybrid carries must remain identical.
            if index == count // 2:
                state.cancel.set()
                state.transition(Phase.CANCELLED)
                runner.finalize(state)
                runner.finalize(state)
            for sequence in (state, survivor):
                if sequence.phase.terminal:
                    continue
                sequence.generated_ids.append(token)
                output = step(sequence, "decode", 1)
                results.append(compare_logits(captures[index + 1], output.logits[0][:, -1, :],
                                              atol=atol, rtol=rtol, mx=mx))
        carry_results = []
        for expected, actual in zip(oracle_cache, survivor.cache.entries, strict=True):
            expected_values = expected.state() if callable(getattr(expected, "state", None)) else (
                getattr(expected, "values", None),
            )
            actual_values = actual.state() if callable(getattr(actual, "state", None)) else (
                getattr(actual, "values", None),
            )
            for a, b in zip(expected_values, actual_values, strict=True):
                if a is not None and b is not None:
                    carry_results.append(bool(mx.allclose(a, b, atol=atol, rtol=rtol).item()))
        # Replay identical logits with independent RNG streams and changing request order.
        options = TextGenerationOptions(temperature=.7, seed=seed)
        independent = [select_token(logits[0], options, rng)
                       for rng in [random.Random(seed)] for logits in captures]
        rngs = {"a": random.Random(seed), "b": random.Random(seed)}
        replay = {"a": [], "b": []}
        for index, logits in enumerate(captures):
            for name in (("a", "b") if index % 2 else ("b", "a")):
                replay[name].append(select_token(logits[0], options, rngs[name]))
        seeded = all(tokens == independent for tokens in replay.values())
        return {
            "status": "passed" if all(r["within_tolerance"] for r in results)
                      and all(carry_results) and seeded else "failed",
            "dtype": dtype, "atol": atol, "rtol": rtol, "forced_steps": count,
            "seed": seed, "prefill_chunk_tokens": prefill_chunk_tokens,
            "prompt_tokens": len(bundle.tokenizer.encode(prompt)),
            "forced_ids_sha256": hashlib.sha256(json.dumps(forced).encode()).hexdigest(),
            "oracle_path": "single_request_dense", "runner_path": "causal_runner_dense",
            "max_error": max(r["max_error"] for r in results),
            "mean_error": sum(r["mean_error"] for r in results) / len(results),
            "greedy_agreement_fraction": sum(r["greedy_agreement"] for r in results) / len(results),
            "comparisons": results, "carry_state_matches": carry_results,
            "seeded_rng_isolation": seeded, "cancelled_request_isolated": state.cache.closed,
            "survivor_layers": describe_cache(survivor.cache.entries, runner.cache_plan()),
        }
    finally:
        runner.release()
        for cache in oracle_cache:
            cache.reset()
