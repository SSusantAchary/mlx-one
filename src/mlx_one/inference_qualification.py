"""Backend-free PR-07--12 profiles, quality gates, and artifact matrix export."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def qualification_profiles(mode, *, models, workload):
    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile

    for key in models:
        hybrid = key == "lfm"
        if (mode == "paged" and hybrid) or (mode == "experimental_hybrid_batch" and not hybrid):
            continue
        if mode == "experimental_dense_batch" and hybrid:
            continue
        model, revision = CHECKPOINTS[key]
        long_context = 127872 if hybrid else 130944
        contexts = ((long_context,) if workload == "long" else ((8192, long_context)
                    if workload in {"quality", "soak"} else
                    ((2048, 8192, 16384, 32640) if workload == "baseline" else (8192,))))
        pages = (16, 32, 64, 128) if mode == "paged" and workload == "baseline" else (32,)
        chunks = ((512, 1024, 2048, 4096) if workload in {"baseline", "batching"} else (512,))
        ratios = (0, .25, .5, .8) if workload == "multi_turn" else (None,)
        policies = (("deadline", "cache_aware")
                    if workload == "multi_turn" and mode != "reference_dense" else ("deadline",))
        durations = (1800, 3600) if workload == "soak" else (None,)
        for context in contexts:
            for slots in (1, 2, 4, 8):
                for page in pages:
                    for chunk in chunks:
                        for ratio in ratios:
                            for policy in policies:
                                for duration in durations:
                                    yield BaselineProfile(
                                        model, revision, mode, context, slots, workload=workload,
                                        cache_type="bf16" if hybrid else "f16",
                                        execution_dtype=None if hybrid else "float16",
                                        page_tokens=page, chunk_tokens=chunk,
                                        max_prefill_tokens=max(1024, chunk),
                                        max_scheduled_tokens=max(2048, chunk * 2),
                                        shared_prefix_fraction=ratio, admission_policy=policy,
                                        duration_seconds=duration,
                                        timeout_seconds=(duration + 600 if duration else 600))


def workload_envelope(profile):
    return {"context_tokens": profile.context_tokens, "slots": profile.slots,
            "chunk_tokens": profile.chunk_tokens, "ttft_budget_ms": profile.ttft_budget_ms,
            "decode_gap_budget_ms": profile.decode_gap_budget_ms,
            "output_tokens": profile.output_tokens, "page_tokens": profile.page_tokens,
            "cache_type": profile.cache_type, "admission_policy": profile.admission_policy}


def qualified_active_decoders(rows, profile):
    """Count simultaneous service-qualified model decode intervals, excluding queue/SSE time."""
    from mlx_one.inference_baseline import percentile

    floor = profile.min_decode_tokens_per_second
    if floor is None or profile.ttft_budget_ms is None or profile.decode_gap_budget_ms is None:
        return None
    events = []
    for row in rows:
        metrics = row.get("metrics", {})
        times = metrics.get("model_token_times_ms") or ()
        started = metrics.get("model_generation_started_at")
        gap = percentile(row.get("model_token_gaps_ms", []), .95)
        if (row.get("temperature_phase") != "warm" or row.get("status") != "completed"
                or row.get("client_ttft_ms") is None
                or row["client_ttft_ms"] > profile.ttft_budget_ms or gap is None
                or gap > profile.decode_gap_budget_ms or started is None or len(times) < 2
                or (metrics.get("decode_tokens_per_second") or 0) < floor):
            continue
        if times[-1] > times[0]:
            events.extend(((started + times[0] / 1000, 1),
                           (started + times[-1] / 1000, -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def evaluate_quality(parity, retrieval, limits):
    metrics = {"max_logit_error": parity["max_error"],
               "mean_nll_delta": parity["held_out_nll_mean_delta"],
               "retrieval_accuracy_drop": (retrieval["oracle_accuracy"]
                                           - retrieval["target_accuracy"])}
    if limits is None:
        return {"status": "diagnostic", "reason": "explicit quality limits are required",
                "metrics": metrics}
    checks = {"logits": metrics["max_logit_error"] <= limits["max_logit_error"],
              "held_out_nll": metrics["mean_nll_delta"] <= limits["max_mean_nll_delta"],
              "retrieval": (metrics["retrieval_accuracy_drop"]
                            <= limits["max_retrieval_accuracy_drop"]),
              "valid_oracle": retrieval["oracle_accuracy"] > 0,
              "prefix_isolation": parity.get("prefix_isolation_passed") is True,
              "seeded_rng": parity.get("seeded_rng_isolation") is True,
              "state_isolation": parity.get("cancelled_request_isolated") is not False}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "metrics": metrics, "limits": limits}


def validate_integrity(directory, *, required=()):
    root = Path(directory)
    integrity = json.loads((root / "result-integrity.json").read_text())
    if not set(required) <= integrity.keys():
        raise ValueError("reference artifact checksums are incomplete")
    for name, digest in integrity.items():
        if (Path(name).name != name or hashlib.sha256((root / name).read_bytes()).hexdigest()
                != digest):
            raise ValueError("reference evidence is stale or has an invalid artifact path")
    return integrity


def load_reference(directory):
    root = Path(directory)
    validate_integrity(root, required={"manifest.json", "summary.json",
                                       "runtime.json", "requests.jsonl"})
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if not str(manifest.get("schema_version", "")).startswith("native-baseline-"):
        raise ValueError("reference must use native execution")
    return {"profile": summary["profile"], "identity": summary["identity"], "summary": summary}


def load_quality_reference(directory, *, profile, runtime):
    from mlx_one.engine.reporting import native_report
    from mlx_one.engine.selection import compatibility_digest, host_identity, source_identity

    root = Path(directory)
    feature = "quantized_paged_kv" if profile.kv_bits else "long_context_quality"
    name = f"evidence-{feature}.json"
    validate_integrity(root, required={"manifest.json", "parity.json", "quality.json",
                                       "retrieval.json", name})
    proof = json.loads((root / name).read_text())
    scope = native_report(runtime.bundle, runtime, mode=profile.inference_mode,
                          hardware=host_identity(), source_revision=source_identity())["scope"]
    scope["cache_precision"] = {"f16": "float16", "bf16": "bfloat16",
                                "q4_0": "float16,uint8", "q8_0": "float16,uint8"}[
                                    profile.cache_type]
    if (proof.get("quality_passed") is not True
            or proof.get("compatibility_digest") != compatibility_digest(runtime)
            or proof.get("workload_envelope") != workload_envelope(profile)
            or proof.get("scope") != scope):
        raise ValueError("long-context quality evidence does not match execution identity/envelope")
    return {"path": str((root / name).resolve()),
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(), "proof": proof}


def write_integrity(output):
    root = Path(output)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
                 if p.is_file() and p.suffix in {".json", ".jsonl", ".csv"}
                 and p.name != "result-integrity.json"}
    (root / "result-integrity.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n")


def evaluate_gate_d(profile, summary, traces, parity, reference=None):
    if profile.kv_bits and profile.quality_limits is None:
        return {"status": "diagnostic",
                "reason": "quantized execution requires declared quality limits"}
    if profile.ttft_budget_ms is None or profile.decode_gap_budget_ms is None:
        return {"status": "diagnostic", "reason": "declare TTFT and model-token gap limits"}
    if reference is None:
        return {"status": "pending", "reason": "checksummed matched reference is required"}
    if reference["summary"].get("failed") != 0:
        raise ValueError("reference run has failures or incomplete failure metadata")
    for field in ("model_id", "revision", "context_tokens", "slots", "output_tokens", "repeats",
                  "seed", "workload", "execution_dtype", "chunk_tokens",
                  "shared_prefix_fraction"):
        if reference["profile"].get(field) != getattr(profile, field):
            raise ValueError(f"reference mismatch in {field}")
    if reference["profile"].get("inference_mode") not in {
        "reference_dense", "experimental_dense_batch"
    }:
        raise ValueError("Gate D requires a dense reference")
    if reference["profile"].get("admission_policy", "deadline") != "deadline":
        raise ValueError("the cooperative reference must retain default admission")
    for field in ("hardware", "source", "weight_precision", "mlx_version"):
        if (not summary["identity"].get(field)
                or summary["identity"][field] != reference["identity"].get(field)):
            raise ValueError(f"reference identity mismatch in {field}")
    if not profile.kv_bits and summary["identity"].get("cache_precision") != reference[
            "identity"].get("cache_precision"):
        raise ValueError("reference cache precision mismatch")
    measured = summary.get("aggregate_output_tokens_per_second_by_group", [])
    baseline = reference["summary"].get("aggregate_output_tokens_per_second_by_group", [])
    speed = (len(measured) == len(baseline) == profile.repeats
             and all(a is not None and b is not None and a > b
                     for a, b in zip(measured, baseline, strict=True)))
    memory = summary.get("peak_cache_bytes")
    old_memory = reference["summary"].get("peak_cache_bytes")
    memory_win = memory is not None and old_memory is not None and memory < old_memory
    ttft, gap = summary.get("warm_client_ttft_p95_ms"), summary.get("model_token_gap_p95_ms")
    direct = profile.inference_mode == "paged"
    checks = {"direct_attention" if direct else "batched_hybrid_execution": (
        any(t.get("direct_attention_calls", 0) > 0 for t in traces) if direct else
        any(t.get("batch_size", 1) > 1 and "conv" in t.get("component_kinds", ()) for t in traces)),
              "configured_batch_demonstrated": profile.slots == 1 or any(
                  t.get("phase") == "decode" and t.get("batch_size", 1) > 1 for t in traces),
              "no_materialization": (bool(traces) and all(t.get("history_materializations") == 0
                                                          for t in traces)) if direct else True,
              "parity_or_quality": parity.get("status") == "passed",
              "service_limits": ttft is not None and gap is not None
                                and ttft <= profile.ttft_budget_ms
                                and gap <= profile.decode_gap_budget_ms,
              "measured_advantage": speed or memory_win,
              "no_failures": summary.get("failed") == 0,
              "no_starvation": summary.get("starved_requests") == 0}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "throughput_improved_each_repetition": speed, "cache_memory_improved": memory_win,
            "throughput_by_repetition": measured, "reference_throughput_by_repetition": baseline,
            "qualification_promoted": False}


def evaluate_gate_e(parity, traces):
    checks = {"logits_and_state_parity": parity.get("status") == "passed",
              "request_isolation": parity.get("cancelled_request_isolated") is True,
              "seeded_rng": parity.get("seeded_rng_isolation") is True,
              "batched_carries": any(t.get("batch_size", 1) > 1
                                     and "conv" in t.get("component_kinds", ()) for t in traces),
              "actual_prefix_restore": bool(parity.get("prefix_restoration")) and all(
                  r["reused_tokens"] > 0 and r["within_tolerance"]
                  for r in parity.get("prefix_restoration", []))}
    return {"status": "passed" if all(checks.values()) else "pending", "checks": checks,
            "qualification_promoted": False}


def evaluate_gate_f(profile, summary, cleanup):
    if profile.workload != "soak" or profile.ttft_budget_ms is None:
        return {"status": "diagnostic", "reason": "soak and explicit service budgets are required"}
    initial = summary.get("soak_initial_active_bytes")
    final = summary.get("soak_final_active_bytes")
    if initial is None or final is None or profile.leak_budget_bytes is None:
        return {"status": "pending",
                "reason": "memory reconciliation and declared leak budget required"}
    ttft, gap = summary.get("warm_client_ttft_p95_ms"), summary.get("model_token_gap_p95_ms")
    checks = {"duration": summary.get("soak_elapsed_seconds", 0) >= profile.duration_seconds,
              "no_failures": summary.get("failed") == 0,
              "all_requests_disposed": cleanup.get("active_handles") == 0
                                       and cleanup.get("reserved_bytes") == 0,
              "memory_reconciled": final - initial <= profile.leak_budget_bytes,
              "recovery_parity": summary.get("recovery_output_matches") is True,
              "pressure_recovery": summary.get("pressure_recovery_passed") is True,
              "no_false_prefix_hits": summary.get("prefix_isolation_passed") is True,
              "service_limits": ttft is not None and gap is not None
                                and ttft <= profile.ttft_budget_ms
                                and gap <= profile.decode_gap_budget_ms}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "qualification_promoted": False}


def export_matrix(results, output):
    """Only observed results enter tables; missing gates remain pending."""
    root, target = Path(results), Path(output)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        profile = manifest.get("profile", {})
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        status_path = directory / "supervision.json"
        status = (json.loads(status_path.read_text()).get("status")
                  if status_path.exists() else "pending")
        row = {"run": str(directory), "model": profile.get("model_id"),
               "revision": profile.get("revision"),
               "chip": manifest.get("hardware", {}).get("chip"),
               "ram_bytes": manifest.get("hardware", {}).get("memory_bytes"),
               "mode": profile.get("inference_mode"),
               "context_tokens": profile.get("context_tokens"),
               "slots": profile.get("slots"), "cache_type": profile.get("cache_type"),
               "workload": profile.get("workload"), "status": status,
               "ttft_p95_ms": summary.get("warm_client_ttft_p95_ms"),
               "model_gap_p95_ms": summary.get("model_token_gap_p95_ms"),
               "sse_gap_p95_ms": summary.get("sse_delivery_gap_p95_ms"),
               "qualified_active_decoders": summary.get("qualified_active_decoders"),
               "failures": summary.get("failed"), "regressions": []}
        for gate in ("d", "e", "f"):
            path = directory / f"gate-{gate}.json"
            row[f"gate_{gate}"] = "pending"
            if path.exists():
                try:
                    validate_integrity(directory, required={"manifest.json", "summary.json",
                                                            path.name})
                    result = json.loads(path.read_text())
                    row[f"gate_{gate}"] = result.get("status", "pending")
                    row["regressions"].extend(f"{gate}:{name}" for name, passed in
                                             result.get("checks", {}).items()
                                             if passed is False and name in {
                                                 "service_limits", "measured_advantage",
                                                 "memory_reconciled", "pressure_recovery"})
                except (OSError, ValueError, KeyError):
                    row[f"gate_{gate}"] = "pending"
                    row["status"] = "stale_or_incomplete_evidence"
        rows.append(row)
    (target / "matrix.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    columns = (list(rows[0]) if rows
               else ["model", "revision", "status", "gate_d", "gate_e", "gate_f"])
    with (target / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    readable = ["# Native inference qualification", "",
                "| Model | Revision | Chip | Mode | Context | Slots | Status | D / E / F |",
                "|---|---|---|---|---:|---:|---|---|"]
    for row in rows:
        readable.append("| " + " | ".join(str(row.get(k) or "pending").replace("|", "\\|")
                                         for k in ("model", "revision", "chip", "mode",
                                                   "context_tokens",
                                                   "slots", "status")) + " | "
                        + " / ".join(row[f"gate_{g}"] for g in "def") + " |")
    (target / "matrix.md").write_text("\n".join(readable) + "\n")
    return rows


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export_matrix(args.results, args.output)
    return 0
