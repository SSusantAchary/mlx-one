"""Native P0.1 observers and forced-token oracles, used only by supervised workers.

No MLX import or model construction occurs when discovering these profiles.
Numerical results and profiled timings are deliberately separate from ordinary
throughput measurements. CPU test doubles cannot establish native evidence.
"""
from __future__ import annotations

import gc
import json
import time
from importlib import import_module
from pathlib import Path

from mlx_one.engine.observation import observe_request
from mlx_one.p0_engine_qualification import atomic_json
from mlx_one.p0_engine_worker import (
    CORPUS,
    Cases,
    Observer,
    component_state,
    deadline,
    error_metrics,
)

TRACE_LIMIT = 65536


class MathObserver(Observer):
    def __init__(self, prompt_length, *, profile=False, trace=None, capture_state=False):
        super().__init__()
        self.prompt_length = prompt_length
        self.profile_math = self.capture_math_dispatch = profile
        self.capture_state = capture_state
        self.components = self.logits_shape = None
        self.events, self.truncated, self.trace = [], 0, trace

    def __call__(self, event):
        if event["kind"] in {"math_phase", "math_dispatch"}:
            if len(self.events) < TRACE_LIMIT:
                # Layer phases can be nested: never sum them as disjoint model time.
                record = {**event, "inclusive": event["kind"] == "math_phase"}
                self.events.append(record)
                if self.trace is not None:
                    self.trace.write(json.dumps(record) + "\n")
            else:
                self.truncated += 1
            return
        if event["kind"] == "state":
            if event["phase"] == "prefill" and event["cache"][0].offset == self.prompt_length:
                self.components = component_state(event["cache"])
                self.logits_shape = list(event["logits"].shape)
                self.capture_state = False
            return
        super().__call__(event)


def engine_settings(policy, route):
    NativeMathPolicy = import_module("mlx_one.engine.optimization_policy").NativeMathPolicy

    if route not in {"reference", "optimized"}:
        raise ValueError("explicit reference or optimized route required")
    if not isinstance(policy, NativeMathPolicy):
        raise TypeError("validated immutable native math policy required")
    if route == "reference" and (policy.attention != "reference" or
                                 policy.projection != "reference" or policy.logits != "all"
                                 or policy.hybrid != "reference" or policy.enable_compile):
        raise ValueError("reference oracle cannot substitute optimized mathematical operations")
    return {"inference_mode": "reference_dense" if route == "reference" else "native_optimized",
            "execution_dtype": "float16", "math_policy": policy}


def prepare_cases(manifest, directory, *, policy, route):
    """Reuse pinned loading/accounting before installing request-local Engine policy.

    Loading is unchanged for the mathematical variants. Retargeting constructs a
    new, empty manager before requests; it never mutates an existing backend.
    Its separately observed initialization cost is not called cold Engine load.
    """
    from mlx_one import Engine

    settings = engine_settings(policy, route)
    cases = Cases(manifest, directory)
    if route == "reference" and policy.prefill_chunk_size is None:
        cases.math_initialization_seconds = 0.0
        return cases
    bundle, config = cases.bundle, cases.engine._cache_manager.config
    started = time.perf_counter()
    cases.engine.close()
    cases.engine = Engine(bundle, offline=True, cache_config=config, **settings)
    cases.bundle = cases.engine.model
    cases.math_initialization_seconds = time.perf_counter() - started
    # Preserve the selected policy across the existing lifecycle case inventory.
    def reload():
        prompt = cases.prompt(CORPUS[0])
        if not cases.engine._closed:
            first, _ = cases.generate(prompt, max_tokens=16)
            cases.reload_expected = first
        else:
            first = cases.reload_expected
        previous = cases.engine._cache_manager
        cases.engine.close()
        assert previous.stats().active_handles == previous.stats().reserved_bytes == 0
        try:
            previous.create_handle("obsolete")
        except RuntimeError:
            pass
        else:
            raise AssertionError("disposed manager accepted a stale request")
        cases.engine = cases.bundle = previous = None
        gc.collect()
        cases.mx.clear_cache()
        cases.engine = Engine(cases.metadata["snapshot_path"], offline=True,
            revision=cases.metadata["revision"], cache_config=config, **settings)
        cases.bundle = cases.engine.model
        second, _ = cases.generate(prompt, max_tokens=16)
        assert cases.engine._inference_mode == settings["inference_mode"]
        assert first["observer"]["token_ids"] == second["observer"]["token_ids"]
        return {"first": first, "reloaded": second, "retained_policy": policy.to_dict()}
    cases.reload = reload
    return cases


def observed_generate(cases, prompt, *, profile=False, trace=None, capture_state=False):
    observer = MathObserver(len(cases.bundle.tokenizer.encode(prompt)),
                            profile=profile, trace=trace, capture_state=capture_state)
    started = time.perf_counter()
    observer.timing.started = started
    with observe_request(observer), deadline(600):
        value = cases.engine.generate(prompt, max_tokens=128, seed=0)
    wall = time.perf_counter() - started
    stats = cases.engine._cache_manager.stats()
    assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
    return {**value.to_dict(), "engine_wall_seconds": wall,
            "observer": observer.summary()}, observer


def compare_components(reference, target):
    """Apply existing tolerances by evaluated component precision, never by preference."""
    if len(reference) != len(target):
        raise ValueError("component topology changed")
    results = []
    for layer, (expected, actual) in enumerate(zip(reference, target, strict=True)):
        if (expected["class"] != actual["class"] or
                expected["position"] != actual["position"]
                or expected["fields"].keys() != actual["fields"].keys()):
            raise ValueError("component type, position or fields changed")
        for name, tensor in expected["fields"].items():
            other = actual["fields"][name]
            if tensor["shape"] != other["shape"] or tensor["dtype"] != other["dtype"]:
                raise ValueError("component shape or actual dtype changed")
            metrics = error_metrics(tensor["array"], other["array"], tensor["dtype"])
            metrics.pop("greedy_agreement")  # State-array argmax is not token agreement.
            results.append({"layer": layer, "field": name, **metrics})
    return results


def forced_state(engine, token_ids, *, policy, chunk=None, forced_tokens=(), output_shapes=None,
                 observer=None):
    """Evaluate independent, owned B1 state under the ordinary memory guard.

    Full logits remain the reference. Only final-position vectors are copied to
    host, and state-copy overhead is explicitly a parity-only reservation.
    """
    import mlx.core as mx
    import numpy as np

    from mlx_one.engine.pressure import execution_forecast
    from mlx_one.engine.transactions import component_tensors

    math_execution = import_module("mlx_one.engine.optimization_policy").math_execution
    runtime = engine._cache_manager
    if not token_ids or chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError("nonempty forced-token input and positive chunk width required")
    if runtime.math_policy != policy:
        raise ValueError("oracle/target math policy does not match memory-planning identity")
    handle = engine._device.call(id(engine), lambda: runtime.create_handle(
        f"math-parity-{time.time_ns()}", key_type="f16", value_type="f16"))
    cache, vectors = handle.bundle.entries, []
    try:
        def reserve():
            growth = runtime.estimate_growth(handle, prompt_tokens=len(token_ids),
                                             max_new_tokens=max(len(forced_tokens), 1))
            return runtime.reserve(handle, tokens=len(token_ids) + len(forced_tokens),
                                   requested_bytes=growth)
        engine._device.call(id(engine), reserve)
        def execute():
            width = chunk or len(token_ids)
            parts = [tuple(token_ids[start:start + width])
                     for start in range(0, len(token_ids), width)]
            parts.extend((token,) for token in forced_tokens)
            for index, part in enumerate(parts):
                forecast = execution_forecast(runtime, engine.model.model.config,
                    batch=1, chunk=len(part), history=cache[0].offset)
                with runtime.transient_reservation(forecast["bytes"]):
                    with math_execution(policy, family=engine.model.architecture,
                                        phase="prefill" if index < len(parts) - len(
                                            forced_tokens) else "decode", observer=observer):
                        output = engine.model.model(mx.array([part]), cache=cache)
                        mx.eval(output.logits, *(tensor for entry in cache
                                                for tensor in component_tensors(entry)))
                    handle.computed_tokens = cache[0].offset
                    runtime.reconcile(handle)
                    if output_shapes is not None:
                        output_shapes.append({"logits": list(output.logits.shape),
                            "logits_bytes": output.logits.nbytes,
                            "logits_dtype": str(output.logits.dtype).rsplit(".", 1)[-1],
                            "hidden": list(output.last_hidden_state.shape),
                            "input_tokens": len(part), "committed_position": cache[0].offset})
                    if index >= len(parts) - len(forced_tokens) - 1:
                        vectors.append(np.asarray(output.logits[0, -1].astype(mx.float32)).copy())
            # Host state copies and GPU conversion scratch are not performance
            # measurements and must still fit the configured/observed headroom.
            storage = sum(t.nbytes * (2 if str(t.dtype).endswith("float16") else 1)
                          for entry in cache for t in component_tensors(entry))
            with runtime.transient_reservation(storage * 2):
                states = component_state(cache, arrays=True)
            return vectors, states
        return engine._device.call(id(engine), execute)
    finally:
        engine._device.call(id(engine), lambda: runtime.release(handle))


def numerical_parity(cases, token_ids, *, policy, chunk=None, continuation=(1, 2, 3, 4),
                     artifact=None):
    """Compare full reference, ordinary cached reference and explicit target math."""
    from mlx_one import Engine
    REFERENCE_MATH = import_module("mlx_one.engine.optimization_policy").REFERENCE_MATH

    config = cases.engine._cache_manager.config
    shapes = {"reference": [], "target": [], "full_reference": []}
    dispatch_observer = MathObserver(len(token_ids))
    dispatch_observer.capture_math_dispatch = True
    # The oracle has its own manager/forecast so ALL logits cannot borrow the
    # last-only target's smaller vocabulary reservation.
    with Engine(cases.bundle, execution_dtype="float16", inference_mode="reference_dense",
                cache_config=config) as oracle:
        expected, reference_state = forced_state(oracle, token_ids, policy=REFERENCE_MATH,
            forced_tokens=continuation, output_shapes=shapes["reference"])
        actual, target_state = forced_state(cases.engine, token_ids, policy=policy,
            chunk=chunk, forced_tokens=continuation, output_shapes=shapes["target"],
            observer=dispatch_observer)
        last_full, full_state = forced_state(oracle, (*token_ids, *continuation),
            policy=REFERENCE_MATH, output_shapes=shapes["full_reference"])
        metrics = [error_metrics(left, right) for left, right in zip(expected, actual, strict=True)]
        full_metric = error_metrics(last_full[-1], expected[-1])
        states = compare_components(reference_state, target_state)
        full_states = compare_components(full_state, reference_state)
        import numpy as np

        margins = []
        for vector in expected:
            top = np.sort(vector)[-2:]
            margins.append(float(top[-1] - top[-2]))
        if artifact is not None:
            np.savez_compressed(artifact, reference=np.stack(expected), target=np.stack(actual),
                full_prefill=np.stack(last_full), prompt_ids=np.asarray(token_ids),
                forced_tokens=np.asarray(continuation))
        passed = (all(m["within_tolerance"] and m["greedy_agreement"] for m in metrics)
                  and full_metric["within_tolerance"] and full_metric["greedy_agreement"]
                  and all(m["within_tolerance"] for m in [*states, *full_states]))
        return {"status": "PASS" if passed else "FAIL", "logits": metrics,
                "independent_full_prefill": full_metric, "components": states,
                "full_prefill_components": full_states, "top_two_margins": margins,
                "token_ids": [int(np.argmax(v)) for v in expected],
                "target_token_ids": [int(np.argmax(v)) for v in actual],
                "forced_tokens": list(continuation), "policy": policy.to_dict(),
                "observed_outputs": shapes,
                "dispatch": dispatch_observer.events,
                "dispatch_truncated": dispatch_observer.truncated,
                "logits_artifact": str(artifact) if artifact is not None else None,
                "qualification_promoted": False}


def controlled_forward(cases, token_ids, forced_tokens, *, policy, observer=None):
    """Synchronized raw forwards with fixed inputs; excludes Engine/sampling service.

    The complete prompt is evaluated before the decode timer starts. Every
    continuation input is declared in the fixture: these are not generated
    answers, and their timing cannot be substituted for user-facing throughput.
    """
    import mlx.core as mx

    from mlx_one.engine.pressure import execution_forecast
    from mlx_one.engine.transactions import component_tensors

    if not token_ids or not forced_tokens:
        raise ValueError("nonempty exact prompt and forced continuation required")
    runtime, engine = cases.engine._cache_manager, cases.engine
    if runtime.math_policy != policy:
        raise ValueError("controlled forward and reservation policy differ")
    math_execution = import_module("mlx_one.engine.optimization_policy").math_execution
    handle = engine._device.call(id(engine), lambda: runtime.create_handle(
        f"math-controlled-{time.time_ns()}", key_type="f16", value_type="f16"))
    try:
        def execute():
            growth = runtime.estimate_growth(handle, prompt_tokens=len(token_ids),
                                              max_new_tokens=len(forced_tokens))
            runtime.reserve(handle, tokens=len(token_ids) + len(forced_tokens),
                            requested_bytes=growth)
            cache, gaps = handle.bundle.entries, []
            def forward(values, phase):
                forecast = execution_forecast(runtime, engine.model.model.config,
                    batch=1, chunk=values.shape[-1], history=cache[0].offset)
                with runtime.transient_reservation(forecast["bytes"]):
                    with math_execution(policy, family=engine.model.architecture,
                                        phase=phase, observer=observer):
                        output = engine.model.model(values, cache=cache)
                        mx.eval(output.logits, *(tensor for entry in cache
                                                for tensor in component_tensors(entry)))
                        mx.synchronize()
                    handle.computed_tokens = cache[0].offset
                    runtime.reconcile(handle)
                return output
            # Inputs must be resident before synchronized forward timing.
            width = policy.prefill_chunk_size or len(token_ids)
            prompt_parts = [mx.array([token_ids[start:start + width]])
                            for start in range(0, len(token_ids), width)]
            continuation = [mx.array([[token]]) for token in forced_tokens]
            mx.eval(*prompt_parts, *continuation)
            mx.synchronize()
            started = time.perf_counter()
            for values in prompt_parts:
                forward(values, "prefill")
            prefill_seconds = time.perf_counter() - started
            history_position = cache[0].offset
            decode_started = time.perf_counter()
            for values in continuation:
                before = time.perf_counter()
                output = forward(values, "decode")
                gaps.append(time.perf_counter() - before)
            decode_seconds = time.perf_counter() - decode_started
            result = {"prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
                "prefill_tps": len(token_ids) / prefill_seconds,
                "decode_tps": len(forced_tokens) / decode_seconds,
                "decode_definition": "fixed B1 evaluated forwards / prepared-history decode time",
                "synchronized_decode_forward_seconds": gaps,
                "prepared_history_position": history_position,
                "committed_position": cache[0].offset, "actual_batch_sizes": [1],
                "actual_prefill_tokens": len(token_ids),
                "forced_tokens": list(forced_tokens), "logits_shape": list(output.logits.shape),
                "components": component_state(cache), "output_kind": "forced_not_generated"}
            if history_position != len(token_ids) or cache[0].offset != (
                    len(token_ids) + len(forced_tokens)):
                raise AssertionError("controlled forward committed incorrect token positions")
            return result
        return engine._device.call(id(engine), execute)
    finally:
        engine._device.call(id(engine), lambda: runtime.release(handle))


def write_profile(path, observer):
    path = Path(path)
    atomic_json(path, {"instrumented": observer.profile_math, "events": observer.events,
        "truncated_events": observer.truncated, "gpu_kernel_count": None,
        "components": observer.components, "logits_shape": observer.logits_shape,
        "phase_times_are_inclusive": True, "qualification_promoted": False})
