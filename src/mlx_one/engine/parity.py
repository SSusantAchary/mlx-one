"""Opt-in forced-token differential checks against native single-request generation."""

from __future__ import annotations

import hashlib
import json
import random
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
