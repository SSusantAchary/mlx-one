"""Offline traffic manifests, progress fairness, and paired policy comparison."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

TRAFFIC_SCHEMA = "native-traffic-v1"


def traffic_identity(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_traffic(data):
    if data.get("schema_version") != TRAFFIC_SCHEMA or not data.get("requests"):
        raise ValueError("versioned nonempty traffic manifest is required")
    seen = set()
    allowed = {"request_id", "arrival_ms", "messages", "prompt_tokens", "principal_label",
               "output_tokens", "deadline_ms", "cancel_after_events", "cancel_after_ms",
               "consumer", "shared_prefix_fraction"}
    for row in data["requests"]:
        if (set(row) - allowed or not isinstance(row.get("request_id"), str)
                or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", row["request_id"])
                or row["request_id"] in seen
                or not isinstance(row.get("arrival_ms"), (int, float))
                or not math.isfinite(row["arrival_ms"]) or row["arrival_ms"] < 0
                or not isinstance(row.get("output_tokens"), int) or row["output_tokens"] < 1
                or not (row.get("messages") or row.get("prompt_tokens"))):
            raise ValueError("invalid traffic request identity, fields or arrival")
        for name in ("deadline_ms", "cancel_after_ms", "cancel_after_events"):
            if row.get(name) is not None and (not isinstance(row[name], (int, float))
                    or not math.isfinite(row[name]) or row[name] <= 0):
                raise ValueError("traffic deadlines and cancellation limits must be positive")
        consumer = row.get("consumer", {})
        if set(consumer) - {"pause_after_events", "pause_ms", "read_delay_ms"} or any(
                not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in consumer.values()):
            raise ValueError("invalid consumer behavior")
        seen.add(row["request_id"])
    return data


def load_traffic_manifest(path):
    data = validate_traffic(json.loads(Path(path).read_text()))
    return {"manifest": data, "sha256": traffic_identity(data)}


def generated_traffic(profile):
    rows = []
    count = profile.slots * 3
    for index in range(count):
        row = {"request_id": f"request-{index:04}", "arrival_ms": index * 20,
               "prompt_tokens": (min(2048, profile.context_tokens) if index % 2 == 0
                                 else min(8192, profile.context_tokens))
                                if profile.workload in {"mixed", "soak"}
                                else profile.context_tokens,
               "output_tokens": min(profile.output_tokens, 128)
                                if profile.workload == "soak" else profile.output_tokens,
               "principal_label": "principal-a",
               "shared_prefix_fraction": profile.shared_prefix_fraction
                                         if profile.shared_prefix_fraction is not None
                                         else .5 if profile.workload == "soak" else 0,
               "consumer": {}}
        if profile.workload in {"cancellation", "soak"} and index % 6 == 1:
            row["cancel_after_events"] = 3
        if profile.workload in {"slow_consumer", "soak"} and index % 6 == 2:
            row["consumer"] = {"pause_after_events": 1, "pause_ms": 31_000}
            row["output_tokens"] = profile.output_tokens
        rows.append(row)
    return validate_traffic({"schema_version": TRAFFIC_SCHEMA, "seed": profile.seed,
                             "requests": rows})


def progress_fairness(events, *, max_queue_wait_ms, max_service_gap_ms):
    """Measure runnable intervals; terminal failures are not a starvation proxy."""
    jobs = {}
    service_gaps, queue_waits = [], []
    violations, exclusions = [], {"backpressure": 0, "pressure": 0, "overload": 0}
    for event in sorted(events, key=lambda x: x["time"]):
        identity = event["request_id"]
        row = jobs.setdefault(identity, {"submitted": event["time"], "last": None,
                                         "blocked": None, "blocked_at": None,
                                         "excluded_wait": 0, "admitted": None})
        kind, now = event["event"], event["time"]
        if kind == "blocked":
            reason = event.get("reason", "backpressure")
            exclusions[reason] = exclusions.get(reason, 0) + 1
            row["blocked"] = reason
            row["blocked_at"] = now
        elif kind == "unblocked":
            if row["blocked"] in {"pressure", "backpressure"}:
                row["excluded_wait"] += now - row["blocked_at"]
            row["blocked"] = None
            row["last"] = now
        elif kind == "admitted":
            row["admitted"], row["last"] = now, now
            wait = (now - row["submitted"] - row["excluded_wait"]) * 1000
            queue_waits.append(wait)
            if max_queue_wait_ms is not None and wait > max_queue_wait_ms:
                violations.append({"request_id": identity, "kind": "admission", "ms": wait})
        elif kind == "forward":
            if row["last"] is not None and row["blocked"] is None:
                gap = (now - row["last"]) * 1000
                service_gaps.append(gap)
                if max_service_gap_ms is not None and gap > max_service_gap_ms:
                    violations.append({"request_id": identity, "kind": "service", "ms": gap})
            row["last"] = now
        elif kind == "terminal":
            if event.get("reason") == "overload":
                exclusions["overload"] += 1
            if row["last"] is not None and row["blocked"] is None:
                gap = (now - row["last"]) * 1000
                service_gaps.append(gap)
                if max_service_gap_ms is not None and gap > max_service_gap_ms:
                    violations.append({"request_id": identity, "kind": "service", "ms": gap})
            if row["admitted"] is None and row["blocked"] is None:
                wait = (now - row["submitted"] - row["excluded_wait"]) * 1000
                if max_queue_wait_ms is not None and wait > max_queue_wait_ms:
                    violations.append({"request_id": identity, "kind": "admission", "ms": wait})
    return {"status": "diagnostic" if max_queue_wait_ms is None or max_service_gap_ms is None
            else "passed" if events and not violations else "failed",
            "max_runnable_service_gap_ms": max(service_gaps) if service_gaps else None,
            "max_admission_wait_ms": max(queue_waits) if queue_waits else None,
            "violations": violations,
            "starved_requests": len({v["request_id"] for v in violations}),
            "exclusions": exclusions, "observed_requests": len(jobs)}


def compare_policy_runs(default, cache_aware):
    from mlx_one.inference_qualification import load_reference, validate_integrity

    a, b = load_reference(default), load_reference(cache_aware)
    if (a["profile"].get("admission_policy") != "deadline"
            or b["profile"].get("admission_policy") != "cache_aware"):
        raise ValueError("policy comparison requires default and cache-aware runs")
    ignored = {"admission_policy", "profile_id", "reference_run"}
    if ({k: v for k, v in a["profile"].items() if k not in ignored}
            != {k: v for k, v in b["profile"].items() if k not in ignored}
            or a["identity"] != b["identity"]):
        raise ValueError("policy run identity or execution settings mismatch")
    traces = []
    for path in (default, cache_aware):
        validate_integrity(path, required={"traffic.json", "tokenization.json"})
        root = Path(path)
        traces.append((json.loads((root / "traffic.json").read_text()),
                       json.loads((root / "tokenization.json").read_text())))
    if traces[0] != traces[1]:
        raise ValueError("policy comparison requires identical arrivals and rendered tokens")
    fields = ("prefix_hits", "prefix_tokens_reused", "queue_wait_p95_ms", "warm_client_ttft_p95_ms",
              "model_token_gap_p95_ms", "warm_aggregate_output_tokens_per_second", "failed",
              "peak_physical_bytes", "starved_requests")
    metrics = {}
    for key in fields:
        left, right = a["summary"].get(key), b["summary"].get(key)
        metrics[key] = {"default": left, "cache_aware": right,
                        "delta": right - left if left is not None and right is not None else None}
    return {"schema_version": "native-policy-comparison-v1", "metrics": metrics,
            "runs": {"default": str(Path(default).resolve()),
                     "cache_aware": str(Path(cache_aware).resolve())},
            "traffic_sha256": traffic_identity(traces[0][0]), "qualification_promoted": False}


def main(argv=None):
    import argparse
    import csv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-run", required=True)
    parser.add_argument("--cache-aware-run", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_policy_runs(args.default_run, args.cache_aware_run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "policy-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "policy-comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "default", "cache_aware", "delta"))
        writer.writeheader()
        writer.writerows({"metric": key, **value} for key, value in result["metrics"].items())
    from mlx_one.inference_qualification import write_integrity

    write_integrity(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


class QualificationMiddleware:
    """Harness-only trusted trace/deadline mapping; never installed by public server."""
    def __init__(self, app, manifest):
        self.app = app
        self.records = {r["request_id"]: r for r in manifest["requests"]}

    async def __call__(self, scope, receive, send):
        from mlx_one.engine.execution_context import qualification_request

        headers = dict(scope.get("headers", []))
        value = headers.get(b"x-qualification-request-id", b"").decode("ascii", errors="ignore")
        record = self.records.get(value.rsplit("/", 1)[-1])
        token = qualification_request.set({"request_id": value,
                  "deadline_ms": record.get("deadline_ms")} if record else None)
        try:
            await self.app(scope, receive, send)
        finally:
            qualification_request.reset(token)
