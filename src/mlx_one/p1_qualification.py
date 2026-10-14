"""P1 acceptance checks over checksummed native harness results; no promotion."""
from __future__ import annotations

from statistics import median

P1_GATE_VERSION = "native-p1-gate-v1"
P1_CHECKS = {"native_parity", "five_repetitions", "matched_serial", "actual_batch",
             "dynamic_membership", "decode_target", "b1_regression", "relative_token_gap",
             "absolute_service", "bounded_progress", "cleanup", "direct_execution"}


def evaluate_p1(profile, summary, traces, parity, *, serial=None, b1=None, no_prefix=None):
    limits = (profile.ttft_budget_ms, profile.decode_gap_budget_ms,
              profile.max_queue_wait_ms, profile.max_service_gap_ms)
    if any(v is None for v in limits):
        return {"schema_version": P1_GATE_VERSION, "status": "diagnostic",
                "reason": "absolute TTFT, model-token gap and fairness limits are required"}
    if serial is None or (profile.slots > 1 and b1 is None):
        return {"schema_version": P1_GATE_VERSION, "status": "pending",
                "reason": "checksummed matched serial and optimized B1 runs are required"}
    matched = True
    for run in (serial, b1):
        if run is None:
            continue
        expected = run["profile"]
        for key in ("model_id", "revision", "context_tokens", "output_tokens", "cache_type",
                    "execution_dtype", "chunk_tokens", "seed", "workload"):
            if expected.get(key) != getattr(profile, key):
                raise ValueError("P1 reference mismatch: " + key)
        for key in ("hardware", "source", "weight_precision", "cache_precision", "mlx_version",
                    "quantization_digest", "floating_policy"):
            if (not summary["identity"].get(key)
                    or summary["identity"][key] != run["identity"].get(key)):
                raise ValueError("P1 reference identity mismatch: " + key)
        matched &= (run.get("native_execution") is True and expected.get("slots") == 1
                    and expected.get("repeats", 0) >= 5
                    and run["summary"].get("failed") == 0)
    matched &= serial["profile"].get("inference_mode") == "reference_dense"
    if b1 is not None:
        matched &= b1["profile"].get("inference_mode") == profile.inference_mode
    rates = summary.get("aggregate_decode_tokens_per_second_by_group", [])
    old = serial["summary"].get("aggregate_decode_tokens_per_second_by_group", [])
    b1_summary = b1["summary"] if b1 else summary
    b1_rates = b1_summary.get("aggregate_decode_tokens_per_second_by_group", [])
    complete = (len(rates) == len(old) == len(b1_rates) == profile.repeats == 5
                and all(v is not None and v > 0 for v in [*rates, *old, *b1_rates]))
    ratios = [a / b for a, b in zip(rates, old, strict=False)] if complete else []
    b1_ratios = [a / b for a, b in zip(b1_rates, old, strict=False)] if complete else []
    memberships = [set(t.get("request_ids", ())) for t in traces if t.get("phase") == "decode"]
    ttft, gap = summary.get("warm_client_ttft_p95_ms"), summary.get("model_token_gap_p95_ms")
    base_gap = b1_summary.get("model_token_gap_p95_ms")
    checks = {"native_parity": parity.get("status") == "passed",
        "five_repetitions": complete, "matched_serial": bool(matched),
        "actual_batch": any(t.get("phase") == "decode" and t.get("batch_size") == profile.slots
                            for t in traces),
        "dynamic_membership": profile.slots == 1 or any(a != b and a & b
                              for a, b in zip(memberships, memberships[1:], strict=False)),
        "decode_target": bool(ratios) and all(r >= (1.25 if profile.slots == 4 else
                                 (.95 if profile.slots == 1 else 1)) for r in ratios),
        "b1_regression": bool(b1_ratios) and all(r >= .95 for r in b1_ratios),
        "relative_token_gap": gap is not None and base_gap is not None
                              and gap <= base_gap * 1.5,
        "absolute_service": ttft is not None and gap is not None
                              and ttft <= limits[0] and gap <= limits[1],
        "bounded_progress": summary.get("starved_requests") == 0
            and summary.get("queue_wait_p95_ms") is not None
            and summary["queue_wait_p95_ms"] <= limits[2]
            and summary.get("max_runnable_service_gap_ms") is not None
            and summary["max_runnable_service_gap_ms"] <= limits[3],
        "cleanup": summary.get("failed") == 0 and summary.get("cleanup_passed") is True,
        "direct_execution": profile.inference_mode != "paged" or bool(traces) and all(
            t.get("history_materializations") == 0 for t in traces) and any(
                t.get("direct_attention_calls", 0) for t in traces)}
    if profile.context_tokens >= 8192 and profile.workload in {"prefix", "multi_turn"}:
        warm = summary.get("warm_client_ttft_p50_ms")
        cold = (no_prefix or {}).get("summary", {}).get("warm_client_ttft_p50_ms")
        valid_prefix_reference = bool(no_prefix and no_prefix.get("native_measurement")
                                      and no_prefix["profile"].get("reuse_prefix") is False)
        if valid_prefix_reference:
            for field in ("model_id", "revision", "context_tokens", "slots", "output_tokens",
                          "execution_dtype", "cache_type", "inference_mode", "chunk_tokens"):
                if no_prefix["profile"].get(field) != getattr(profile, field):
                    raise ValueError("P1 no-prefix reference mismatch: " + field)
            if no_prefix["identity"] != summary["identity"]:
                raise ValueError("P1 no-prefix execution identity mismatch")
        checks["8k_prefix"] = (valid_prefix_reference and warm is not None and cold is not None
                               and warm <= cold * .9
                               and summary.get("prefix_tokens_reused", 0) > 0)
    return {"schema_version": P1_GATE_VERSION,
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "aggregate_decode_speedup_median": median(ratios) if ratios else None,
            "b1_decode_ratio_median": median(b1_ratios) if b1_ratios else None,
            "reference_artifacts": [artifact for run in (serial, b1, no_prefix) if run
                                    for artifact in run.get("reference_artifacts", [])],
            "qualification_promoted": False}
