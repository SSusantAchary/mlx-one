"""Opt-in forced-fused Metal diagnostics. Receipts never promote runtime dispatch."""

import csv
import gc
import json
import os
import statistics
import time
from functools import wraps
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_FUSED_TESTS") != "1":
    pytest.skip(
        "HARDWARE_PENDING: run scripts/qualification/fused-correctness.sh on M4",
        allow_module_level=True,
    )

import mlx.core as mx
import numpy as np

from mlx_one.engine.kernel_dispatch import sdpa_capability
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
from mlx_one.engine.parity import compare_logits
from mlx_one.inference_baseline import TOLERANCES, process_rss
from mlx_one.inference_qualification import write_integrity
from mlx_one.native_correctness import checkpoint_proof
from mlx_one.native_qualification import qualification_source, write
from mlx_one.ops.attention import (
    attention_fused,
    attention_reference,
    mlx_version,
    validate_attention_inputs,
)


def guarded(function):
    """An executed exception always leaves FAIL, including failures before model load."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        receipt, request = kwargs["receipt"], kwargs["request"]
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            rows = [row for row in receipt["cases"] if row["test"] == request.node.nodeid]
            if not rows:
                receipt["cases"].append({"test": request.node.nodeid})
                rows = receipt["cases"][-1:]
            for row in rows:
                row.update(
                    status="FAIL", exception={"type": type(exc).__name__, "message": str(exc)}
                )
            raise

    return wrapped


@pytest.fixture(scope="module")
def receipt():
    directory = Path(os.environ["MLX_ONE_FUSED_RESULTS_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "native-fused-diagnostic-v2",
        "source": qualification_source(),
        "scope": json.loads(os.environ["MLX_ONE_FUSED_SCOPE"]),
        "status": "NOT_RUN",
        "cases": [],
        "force_fused": True,
        "qualification_promoted": False,
    }
    try:
        assert mx.metal.is_available(), "Metal unavailable; no CPU fallback permitted"
        mx.set_default_device(mx.gpu)
        yield report
    finally:
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        report["status"] = (
            "COMPLETE_WITH_UNSUPPORTED"
            if report["cases"]
            and all(case["status"] in {"PASS", "UNSUPPORTED"} for case in report["cases"])
            and any(case["status"] == "UNSUPPORTED" for case in report["cases"])
            else "PASS"
            if report["cases"] and all(case["status"] == "PASS" for case in report["cases"])
            else "FAIL"
        )
        report["counts"] = {
            status: sum(row["status"] == status for row in report["cases"])
            for status in ("PASS", "FAIL", "UNSUPPORTED", "NOT_RUN")
        }
        write(directory / "fused-results.json", report)
        with (directory / "fused-results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "test",
                    "status",
                    "max_error",
                    "mean_error",
                    "reference_seconds",
                    "fused_seconds",
                    "latency_speedup",
                ],
            )
            writer.writeheader()
            for row in report["cases"]:
                writer.writerow({key: row.get(key) for key in writer.fieldnames})
        write_integrity(directory)


def metrics(reference, actual, dtype):
    left = np.asarray(reference.astype(mx.float32))
    right = np.asarray(actual.astype(mx.float32))
    error = np.abs(left - right)
    atol, rtol = TOLERANCES[dtype]
    return {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "finite": bool(np.isfinite(right).all()),
        "atol": atol,
        "rtol": rtol,
        "within_tolerance": bool(np.all(error <= atol + rtol * np.abs(left))),
    }


def measure(function):
    value = function()  # Warmup is excluded from the five measured repetitions.
    mx.eval(value)
    del value
    samples = []
    for _ in range(5):
        mx.synchronize()
        mx.reset_peak_memory()
        started = time.perf_counter()
        value = function()
        mx.eval(value)
        mx.synchronize()
        samples.append(
            {
                "seconds": time.perf_counter() - started,
                "peak_mlx_bytes": mx.get_peak_memory(),
                "active_mlx_bytes": mx.get_active_memory(),
                "rss_bytes": process_rss(os.getpid()),
            }
        )
        del value
    return samples


@guarded
def test_fused_kernel_smoke(receipt, request):
    row = {"test": request.node.nodeid, "status": "RUNNING"}
    receipt["cases"].append(row)
    q = mx.ones((1, 2, 2, 64), dtype=mx.float16)
    k = mx.ones((1, 1, 2, 64), dtype=mx.float16)
    value = attention_fused(q, k, k, scale=0.125, mask="causal")
    mx.eval(value)
    good = value.shape == q.shape and bool(mx.all(mx.isfinite(value)).item())
    row["status"] = "PASS" if good else "FAIL"
    assert good


@pytest.mark.parametrize("batch", (1, 2, 4))
@pytest.mark.parametrize("heads,kv_heads", ((8, 8), (8, 2), (8, 1)))
@pytest.mark.parametrize("dtype", ("float16", "bfloat16"))
@pytest.mark.parametrize("queries,history", ((1, 0), (1, 17), (7, 19), (32, 0)))
@pytest.mark.parametrize("mask_kind", ("causal", "boolean", "additive"))
@guarded
def test_fused_kernel_identical_operands(
    receipt, request, batch, heads, kv_heads, dtype, queries, history, mask_kind
):
    row = {
        "test": request.node.nodeid,
        "status": "RUNNING",
        "dtype": dtype,
        "batch": batch,
        "heads": heads,
        "kv_heads": kv_heads,
        "queries": queries,
        "history": history,
        "mask": mask_kind,
        "gpu_kernel_count": None,
    }
    receipt["cases"].append(row)
    mx.random.seed(616)
    q = mx.random.normal((batch, heads, queries, 64)).astype(getattr(mx, dtype))
    k = mx.random.normal((batch, kv_heads, queries + history, 64)).astype(getattr(mx, dtype))
    v = mx.random.normal(k.shape).astype(getattr(mx, dtype))
    valid = mx.arange(queries + history)[None, :] <= history + mx.arange(queries)[:, None]
    if mask_kind != "causal":
        # Per-lane ragged histories retain at least one safe key per real query.
        left_padding = mx.array([min(i, history) for i in range(batch)])
        valid = valid[None, None] & (
            mx.arange(queries + history)[None, None, None] >= left_padding[:, None, None, None]
        )
    mask = (
        "causal"
        if mask_kind == "causal"
        else valid
        if mask_kind == "boolean"
        else mx.where(valid, 0, -float("inf")).astype(getattr(mx, dtype))
    )
    mx.eval(q, k, v, valid)
    reference = attention_reference(q, k, v, scale=0.125, mask=mask, stable=True, promote=True)
    mx.eval(reference)
    results = {}
    capability = sdpa_capability(
        q_shape=q.shape, k_shape=k.shape, v_shape=v.shape, mlx_version=mlx_version()
    )
    row.update(
        capability=capability.__dict__,
        q_shape=list(q.shape),
        k_shape=list(k.shape),
        selected_fallback="reference" if capability.selected == "reference" else None,
    )
    for accumulation in ("native", "float32"):
        try:
            actual = attention_fused(q, k, v, scale=0.125, mask=mask, accumulation=accumulation)
            mx.eval(actual)
            results[accumulation] = {
                **metrics(reference, actual, dtype),
                "status": "PASS",
                "actual_dispatch": "forced_fused",
            }
        except ValueError as exc:
            if (
                capability.selected != "reference"
                or "force_fused=True but no fused kernel is available" not in str(exc)
            ):
                results[accumulation] = {"status": "FAIL", "exception": str(exc)}
            else:
                # Expected rejection is evidence of UNSUPPORTED, never a fused pass.
                results[accumulation] = {
                    "status": "UNSUPPORTED",
                    "exception": str(exc),
                    "actual_dispatch": "rejected_before_kernel",
                }
        except Exception as exc:
            results[accumulation] = {"status": "FAIL", "exception": repr(exc)}
    row["results"] = results
    if capability.selected == "reference" and all(
        r["status"] == "UNSUPPORTED" for r in results.values()
    ):
        fallback = attention_reference(q, k, v, scale=0.125, mask=mask, stable=True, promote=True)
        mx.eval(fallback)
        row.update(
            status="UNSUPPORTED",
            fallback_result=metrics(reference, fallback, dtype),
            reason=capability.reason,
        )
        assert row["fallback_result"]["within_tolerance"]
        return
    row.update(
        max_error=max(r.get("max_error", 0) for r in results.values()),
        mean_error=max(r.get("mean_error", 0) for r in results.values()),
    )
    good = all(
        r["status"] == "PASS" and r["finite"] and r["within_tolerance"] for r in results.values()
    )
    row["status"] = "PASS" if good else "FAIL"
    assert good, row
    del reference, actual
    if os.environ.get("MLX_ONE_FUSED_PERFORMANCE") == "1":
        row["reference_samples"] = measure(
            lambda: attention_reference(q, k, v, scale=0.125, mask=mask, stable=True, promote=True)
        )
        row["fused_samples"] = measure(lambda: attention_fused(q, k, v, scale=0.125, mask=mask))
        row["reference_seconds"] = statistics.median(s["seconds"] for s in row["reference_samples"])
        row["fused_seconds"] = statistics.median(s["seconds"] for s in row["fused_samples"])
        row["latency_speedup"] = row["reference_seconds"] / row["fused_seconds"]


@pytest.mark.parametrize("context", (512, 2048, 4096))
@pytest.mark.parametrize("dim", (64, 128))
@guarded
def test_fused_kernel_long_prefill(receipt, request, context, dim):
    row = {
        "test": request.node.nodeid,
        "status": "RUNNING",
        "context": context,
        "head_dim": dim,
        "scope": "B1 long prefill identical operands",
    }
    receipt["cases"].append(row)
    mx.random.seed(616)
    q = mx.random.normal((1, 8, context, dim)).astype(mx.float16)
    k = mx.random.normal((1, 2, context, dim)).astype(mx.float16)
    v = mx.random.normal(k.shape).astype(mx.float16)
    mx.eval(q, k, v)
    reference = attention_reference(
        q, k, v, scale=dim**-0.5, mask="causal", stable=True, promote=True
    )
    actual = attention_fused(q, k, v, scale=dim**-0.5, mask="causal")
    mx.eval(reference, actual)
    result = metrics(reference, actual, "float16")
    row.update(result, status="PASS" if result["finite"] and result["within_tolerance"] else "FAIL")
    assert row["status"] == "PASS", row
    del reference, actual
    if os.environ.get("MLX_ONE_FUSED_PERFORMANCE") == "1":
        row["reference_samples"] = measure(
            lambda: attention_reference(
                q, k, v, scale=dim**-0.5, mask="causal", stable=True, promote=True
            )
        )
        row["fused_samples"] = measure(
            lambda: attention_fused(q, k, v, scale=dim**-0.5, mask="causal")
        )
        row["reference_seconds"] = statistics.median(s["seconds"] for s in row["reference_samples"])
        row["fused_seconds"] = statistics.median(s["seconds"] for s in row["fused_samples"])
        row["latency_speedup"] = row["reference_seconds"] / row["fused_seconds"]


@pytest.mark.parametrize("checkpoint", json.loads(os.environ.get("MLX_ONE_FUSED_KEYS", "[]")))
@guarded
def test_fused_checkpoint_logits_state_and_continuation(receipt, monkeypatch, request, checkpoint):
    from mlx_one.models.shared import transformer
    from mlx_one.text import load_text_model

    path = json.loads(os.environ["MLX_ONE_FUSED_PATHS"])[checkpoint]
    asset = checkpoint_proof(checkpoint, path)
    bundle = load_text_model(
        asset["model_id"],
        checkpoint_path=path,
        expected_files=asset["files"],
        revision=asset["revision"],
        offline=True,
    )
    configure_execution_dtype(bundle, "float16")
    row = {
        "test": request.node.nodeid,
        "status": "RUNNING",
        "checkpoint": asset,
        "comparisons": [],
        "scope": "B1 direct model; full/chunked/forced continuation",
    }
    receipt["cases"].append(row)
    if not any(
        isinstance(module, transformer.GroupedQueryAttention)
        for _, module in bundle.model.named_modules()
    ):
        row.update(
            status="UNSUPPORTED", reason="model does not use the shared attention dispatcher"
        )
        bundle = None
        gc.collect()
        mx.clear_cache()
        return
    caches = [bundle.model.make_cache() for _ in range(3)]
    ids = bundle.tokenizer.encode(
        "An engineer measures a quiet river, then records seven distinct observations. " * 8
    )[:64]
    forced = bundle.tokenizer.encode(" The measured result remains reproducible." * 8)[:32]
    row.update(
        prompt_ids=ids, forced_ids=forced, fused_calls=0, dispatches=[], atol=0.01, rtol=0.01
    )

    def forward(cache, tokens, *, fused):
        with monkeypatch.context() as patch:
            if fused:

                def select(q, k, v, *, mask=None, history=0, requires_stable_rounding=False):
                    validate_attention_inputs(q, k, v, mask=mask, history=history)
                    shape = list(k.shape)
                    shape[2] += history
                    vshape = list(v.shape)
                    vshape[2] += history
                    decision = sdpa_capability(
                        q_shape=q.shape,
                        k_shape=tuple(shape),
                        v_shape=tuple(vshape),
                        mlx_version=mlx_version(),
                    )
                    row["fused_calls"] += int(decision.selected == "mlx_sdpa")
                    row["dispatches"].append(
                        {
                            "q_shape": list(q.shape),
                            "k_shape": shape,
                            "selected": decision.selected,
                            "reason": decision.reason,
                        }
                    )
                    return decision

                patch.setattr(transformer, "select_attention", select)
                def diagnostic_attention(q, k, v, *, scale, mask, stable, promote, decision):
                    if decision.selected == "mlx_sdpa":
                        return attention_fused(q, k, v, scale=scale, mask=mask,
                                               accumulation="float32")
                    return attention_reference(q, k, v, scale=scale, mask=mask,
                                               stable=stable, promote=promote)

                patch.setattr(transformer, "attention", diagnostic_attention)
            with math_execution(
                NativeMathPolicy(attention="mlx_sdpa" if fused else "reference"),
                family=bundle.architecture,
            ):
                value = bundle.model(mx.array([tokens]), cache=cache).logits
                mx.eval(value)
                return value[:, -1:, :]

    def compare(left, right, *, state=False, stage="prefill"):
        result = compare_logits(left, right, atol=0.01, rtol=0.01)
        result["kind"] = "state" if state else "logits"
        result["stage"] = stage
        row["comparisons"].append(result)

    def compare_states(others, stage):
        for lane, other_cache in enumerate(others, 1):
            for layer, (reference_cache, other) in enumerate(
                zip(caches[0], other_cache, strict=True)
            ):
                assert reference_cache.offset == other.offset
                for component, (left, right) in enumerate(
                    zip(reference_cache.state(), other.state(), strict=True)
                ):
                    compare(left, right, state=True,
                            stage=f"{stage}/lane={lane}/layer={layer}/component={component}")

    try:
        reference = forward(caches[0], ids, fused=False)
        compare(reference, forward(caches[1], ids, fused=True))
        compare_states(caches[1:2], "full_prefill_kv")
        for start in range(0, len(ids), 7):
            chunked = forward(caches[2], ids[start : start + 7], fused=True)
        compare(reference, chunked, stage="chunked_prefill")
        compare_states(caches[1:], "chunked_prefill_kv")
        for step, token in enumerate(forced):
            reference = forward(caches[0], [token], fused=False)
            for cache in caches[1:]:
                compare(reference, forward(cache, [token], fused=True), stage="continuation")
            compare_states(caches[1:], f"continuation_kv/step={step}")
        good = row["fused_calls"] > 0 and all(
            r["within_tolerance"] and (r["kind"] == "state" or r["greedy_agreement"])
            for r in row["comparisons"]
        )
        row.update(
            status="PASS" if good else "FAIL",
            max_error=max(r["max_error"] for r in row["comparisons"]),
            mean_error=max(r["mean_error"] for r in row["comparisons"]),
        )
        assert good, row
    finally:
        row["checkpoint_unchanged"] = checkpoint_proof(checkpoint, path) == asset
        for cache in caches:
            for layer in cache:
                layer.reset()
        row["cleanup_offsets"] = [[layer.offset for layer in cache] for cache in caches]
        bundle = None
        caches.clear()
        gc.collect()
        mx.clear_cache()
