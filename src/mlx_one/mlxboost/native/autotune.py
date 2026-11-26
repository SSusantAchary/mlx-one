"""Explicit bounded offline search. Experimental evidence never self-approves."""

from __future__ import annotations

import random
import time
from dataclasses import asdict

from . import reference_axpby, request_for
from .capabilities import binary_sha256, load_extension, source_sha256
from .profiling import benchmark, environment
from .registry import digest_json, write_immutable
from .schemas import KERNEL_ID, THREADGROUPS, AutotuneCandidate, KernelCandidate


def rank_results(results):
    """Correctness precedes timing; ties are deterministic, no fabricated fallback winner."""
    valid = [
        r for r in results if all(r.get(k) is True for k in ("correct", "stable", "no_regression"))
    ]
    return sorted(valid, key=lambda r: (r["profile"]["median_ms"], r["threadgroup_size"]))


def export_proposal(evidence_path, record):
    """Export review material, with runtime approval false for every candidate."""
    entries = []
    for result in rank_results(record["results"]):
        identity = dict(
            request=record["request"],
            implementation_sha256=record["implementation_sha256"],
            binary_sha256=record["binary_sha256"],
            threadgroup_size=result["threadgroup_size"],
        )
        candidate = KernelCandidate(
            digest_json(identity),
            KERNEL_ID,
            "1.0.0",
            record["implementation_sha256"],
            record["binary_sha256"],
            result["threadgroup_size"],
            evidence_path.stem,
        )
        entries.append(
            dict(
                request=record["request"],
                implementation_sha256=record["implementation_sha256"],
                binary_sha256=record["binary_sha256"],
                candidate=asdict(candidate),
                numerical_status="pass",
                performance_status="experimental",
                approved_for_runtime=False,
                reviewed_by=None,
            )
        )
    return write_immutable(
        evidence_path.parent, dict(schema_version=1, kind="review_proposal", candidates=entries)
    )


def tune(
    directory,
    *,
    length=65536,
    dtype="float32",
    seed=7,
    warmup=5,
    trials=30,
    memory_budget_bytes=64 * 1024**2,
    budget_seconds=60,
):
    if type(length) is not int or not 1 <= length <= 4 * 1024**2:
        raise ValueError("tuning length must be 1..4194304")
    if dtype not in {"float16", "float32"}:
        raise ValueError("unsupported_dtype")
    if not 0 < memory_budget_bytes <= 256 * 1024**2 or length * 4 * 12 > memory_budget_bytes:
        raise ValueError("tuning memory budget exceeded")
    if not 0 < budget_seconds <= 600:
        raise ValueError("invalid time budget")
    started = time.monotonic()
    init_started = time.perf_counter_ns()
    module = load_extension()
    initialization_ms = (time.perf_counter_ns() - init_started) / 1e6
    import mlx.core as mx

    mx.random.seed(seed)
    x = mx.random.uniform(-1, 1, (length,)).astype(getattr(mx, dtype))
    y = mx.random.uniform(-1, 1, (length,)).astype(getattr(mx, dtype))
    mx.eval(x, y)
    a, b = 0.75, -0.25
    request = request_for(x, y, a, b)
    mx.synchronize()
    pipeline_started = time.perf_counter_ns()
    max_threads = module.max_threads(x.dtype)
    mx.synchronize()
    pipeline_setup_ms = (time.perf_counter_ns() - pipeline_started) / 1e6
    ref = reference_axpby(x, y, a, b)
    mx.eval(ref)
    result = dict(
        schema_version=1,
        scope="kernel",
        operation=request.operation,
        status="EXPERIMENTAL",
        approved_for_runtime=False,
        request=asdict(request),
        implementation_sha256=source_sha256(),
        binary_sha256=binary_sha256(),
        native_initialization_ms=initialization_ms,
        pipeline_setup_ms=pipeline_setup_ms,
        preparation="pipeline query and numerical checks precede profiling",
        environment=environment(mx),
        seed=seed,
        results=[],
        budgets=dict(memory_bytes=memory_budget_bytes, seconds=budget_seconds),
        gates=dict(
            atol=1e-3 if dtype == "float16" else 1e-6,
            rtol=1e-3 if dtype == "float16" else 1e-6,
            max_cv=0.25,
            max_regression_fraction=0.05,
        ),
    )
    functions = {"mlx_reference": lambda: reference_axpby(x, y, a, b)}
    sizes = list(THREADGROUPS)
    random.Random(seed).shuffle(sizes)
    # Every safe candidate must pass correctness BEFORE it is timed or ranked.
    for size in sizes:
        candidate = AutotuneCandidate(request, source_sha256(), binary_sha256(), size, max_threads)
        record = dict(
            threadgroup_size=size,
            candidate=asdict(candidate),
            correct=False,
            stable=False,
            no_regression=False,
        )
        if size > max_threads:
            record["reason"] = "device_threadgroup_limit"
        else:

            def fn(size=size):
                return module.axpby(x, y, a, b, threadgroup_size=size)

            out = fn()
            mx.eval(out)
            record["max_abs_error"] = float(mx.max(mx.abs(out - ref)).item())
            record["correct"] = bool(
                mx.allclose(
                    out, ref, atol=result["gates"]["atol"], rtol=result["gates"]["rtol"]
                ).item()
            )
            if record["correct"]:
                functions[str(size)] = fn
            else:
                record["reason"] = "failed_parity"
        result["results"].append(record)
    remaining = budget_seconds - (time.monotonic() - started)
    profiles = benchmark(
        functions, warmup=warmup, trials=trials, seed=seed, mx=mx, budget_seconds=remaining
    )
    result["reference_profile"] = profiles["mlx_reference"]
    for record in result["results"]:
        profile = profiles.get(str(record["threadgroup_size"]))
        if profile is not None:
            record["profile"] = profile
            mean = sum(profile["samples_ms"]) / profile["sample_count"]
            record["stable"] = profile["stdev_ms"] / mean <= result["gates"]["max_cv"]
            record["no_regression"] = profile["median_ms"] <= (
                profiles["mlx_reference"]["median_ms"] * 1.05
            )
    result["ranking"] = [r["threadgroup_size"] for r in rank_results(result["results"])]
    result["elapsed_seconds"] = time.monotonic() - started
    path = write_immutable(directory, result)
    return path, result
