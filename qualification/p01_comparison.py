"""Compare checksummed P0.1 artifacts without inferring unmeasured qualification."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from mlx_one.inference_qualification import validate_integrity, write_integrity
from mlx_one.p0_engine_qualification import atomic_json
from qualification.p01_math import compare, summarize, validate_pair


def promotion_gates(reference, candidate, *, competitor=False, protected=None):
    comparison = compare(reference, candidate)
    reasons = []
    failures = []
    if comparison["status"] != "MEASURED":
        reasons.append("missing correctness, repetitions or matched memory/timing telemetry")
    if candidate.get("near_tie_token_changes"):
        failures.append("token changes require investigation; near ties are not exemptions")
    if not candidate.get("independent_processes", 0) >= 2:
        reasons.append("fewer than two independent processes")
    if candidate.get("execution_order") != "ABBA":
        reasons.append("balanced independent ABBA execution not established")
    if candidate.get("thermal_noise_checked") is not True:
        reasons.append("repeatability and thermal/noise check missing")
    if candidate.get("optimized_competitor") and not candidate.get("engaged_patch_calls", 0) > 0:
        reasons.append("competitor optimized patch did not demonstrably execute")
    ratios = comparison["ratios"]
    if not reasons:
        if competitor:
            passes = (ratios["prefill_tps"] >= 1.10 and ratios["decode_tps"] >= 1.10
                      and ratios["peak_mlx_bytes"] <= 1.05)
        else:
            passes = (ratios["prefill_tps"] >= 1.10 or ratios["decode_tps"] >= 1.10
                      or ratios["peak_mlx_bytes"] <= .85)
        if not passes:
            failures.append("declared materiality/competitor target not met")
    required = candidate.get("required_protected_profiles")
    if not isinstance(required, list) or not required:
        reasons.append("declared protected workload coverage missing")
    if not isinstance(protected, dict) or not protected:
        reasons.append("protected decode/throughput regression matrix missing")
    else:
        if required and set(required) - protected.keys():
            reasons.append("protected workload matrix incomplete")
        if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
               for value in protected.values()):
            reasons.append("protected measurements are invalid")
        elif any(value < .97 for value in protected.values()):
            failures.append("protected route has more than 3% throughput regression")
    status = "FAIL" if failures else "PENDING" if reasons else "PASS"
    return {**comparison, "gate_status": status,
            "reasons": reasons + failures, "regressions": failures,
            "competitor_win": competitor and status == "PASS",
            "qualification_promoted": False}


def read_worker(path):
    path = Path(path).resolve()
    validate_integrity(path.parent, required={path.name})
    result = json.loads(path.read_text())
    parent = path.parent.parent
    validate_integrity(parent, required={"supervision.json", "manifest.json", "command.json"})
    supervision = json.loads((parent / "supervision.json").read_text())
    if result.get("status") != "PASS" or supervision.get("status") != "completed":
        raise ValueError("failed, partial or unsupervised worker cannot enter comparison")
    return result


def baseline_repeatability(results, *, noise_limit=.05):
    if len(results) < 2:
        return {"status": "PENDING", "reason": "two independent workers required"}
    first = results[0]
    for other in results[1:]:
        validate_pair(first["identity"], other["identity"])
    report = {"status": "PASS", "noise_limit_fraction": noise_limit, "metrics": {},
              "qualification_promoted": False}
    for metric in ("prefill_tps", "decode_tps"):
        processes = []
        for result in results:
            values = [row.get(metric) for row in result["measurements"]]
            if not values or any(value is None for value in values):
                report.update(status="PENDING", reason="undefined measured intervals")
                break
            processes.append(summarize(values)["p50"])
        if len(processes) != len(results):
            report["metrics"][metric] = None
            continue
        span = max(processes) / min(processes) - 1 if min(processes) > 0 else None
        report["metrics"][metric] = {"independent_worker_p50": processes,
                                      "relative_span": span}
        if span is None or span > noise_limit:
            report.update(status="PENDING", reason="baseline noise exceeds declared limit")
    return report


def aggregate_workers(paths):
    """Derive independence from distinct supervised processes, never user flags."""
    paths = [Path(path).resolve() for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("distinct independent worker artifacts required")
    values, chronology = [], []
    for path in paths:
        result = read_worker(path)
        from qualification.p01_experiments import VERSION

        if result.get("schema_version") != VERSION:
            raise ValueError("legacy worker evidence is diagnostic; current schema required")
        if values:
            validate_pair(values[0]["identity"], result["identity"])
        parent = path.parent.parent
        manifest = json.loads((parent / "manifest.json").read_text())
        supervision = json.loads((parent / "supervision.json").read_text())
        if result["identity"]["source_sha256"] != manifest["source"]["source_sha256"]:
            raise ValueError("supervised source and measured source differ")
        origin = supervision.get("monotonic_origin")
        finished = supervision.get("monotonic_finished")
        if (type(origin) not in {int, float} or not math.isfinite(origin)
                or type(finished) not in {int, float} or not math.isfinite(finished)
                or finished <= origin):
            raise ValueError("independent worker chronology missing")
        chronology.append({"started": origin, "finished": finished,
                           "route": manifest["task"]["route"],
                           "artifact": str(path)})
        values.append(result)
    repeatability = baseline_repeatability(values)
    return {"identity": values[0]["identity"],
        "measurements": [row for value in values for row in value["measurements"]],
        "correctness_passed": all(value.get("correctness_passed") is True for value in values),
        "independent_processes": len(paths), "execution_order": "UNVERIFIED",
        "thermal_noise_checked": repeatability["status"] == "PASS",
        "repeatability": repeatability, "worker_chronology": chronology,
        "optimized_competitor": any(v.get("optimized_competitor") for v in values),
        "engaged_patch_calls": min((v.get("engaged_patch_calls", 0) for v in values), default=0),
        "artifacts": [str(path) for path in paths], "qualification_promoted": False}


def verify_abba(reference, candidate):
    timeline = sorted([*({**row, "side": "reference"} for row in reference["worker_chronology"]),
                       *({**row, "side": "optimized"} for row in candidate["worker_chronology"])],
                      key=lambda row: row["started"])
    actual = [row["side"] for row in timeline]
    if (len(actual) < 4 or len(actual) % 4
            or actual != ["reference", "optimized", "optimized", "reference"] * (len(actual) // 4)):
        raise ValueError("supervised worker order is not independent balanced ABBA")
    if any(left["finished"] > right["started"] for left, right in
           zip(timeline, timeline[1:], strict=False)):
        raise ValueError("overlapping or ambiguous worker chronology")
    reference["execution_order"] = candidate["execution_order"] = "ABBA"


def baseline_matrix(root):
    """Keep missing/failed workers visible; repeated tokens are not numerical proof."""
    root = Path(root).resolve()
    validate_integrity(root, required={"manifest.json"})
    manifest = json.loads((root / "manifest.json").read_text())
    groups = defaultdict(list)
    for task in manifest["tasks"]:
        groups[(task["checkpoint"], task["prompt_tokens"], task["route"])].append(task)
    rows = []
    for (checkpoint, width, route), tasks in groups.items():
        failures = [t for t in tasks if t["status"] != "PASS"]
        results = [read_worker(Path(t["artifact"]) / "data/experiment.json")
                   for t in tasks if t["status"] == "PASS"]
        repeatability = baseline_repeatability(results)
        rows.append({"checkpoint": checkpoint, "prompt_tokens": width, "route": route,
            "status": "PENDING" if failures else repeatability["status"],
            "workers": len(results), "failures": failures, "repeatability": repeatability,
            "metrics": {key: summarize(row[key] for result in results
                for row in result["measurements"] if row.get(key) is not None)
                for key in ("prefill_tps", "decode_tps", "peak_mlx_bytes", "rss_peak_bytes")},
            "numerical_qualification": "NOT_RUN", "qualification_promoted": False})
    return {"schema_version": "native-math-baseline-matrix-v1", "source": manifest["source"],
            "rows": rows, "qualification_promoted": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+")
    parser.add_argument("--candidate", type=Path, nargs="+")
    parser.add_argument("--baseline-matrix", type=Path)
    parser.add_argument("--competitor", action="store_true")
    parser.add_argument("--protected-matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.baseline_matrix:
        report = baseline_matrix(args.baseline_matrix)
        atomic_json(args.output / "baseline-matrix.json", report)
        write_integrity(args.output)
        print(json.dumps(report, indent=2))
        return 0
    if not args.reference or not args.candidate:
        parser.error("distinct --reference and --candidate worker artifacts required")
    reference, candidate = aggregate_workers(args.reference), aggregate_workers(args.candidate)
    try:
        verify_abba(reference, candidate)
    except ValueError as error:
        candidate["order_limitation"] = str(error)
    protected = None
    if args.protected_matrix:
        validate_integrity(args.protected_matrix.parent, required={args.protected_matrix.name})
        protected = json.loads(args.protected_matrix.read_text())
        candidate["required_protected_profiles"] = protected.get("required_profiles")
        protected = protected.get("ratios")
    report = promotion_gates(reference, candidate, competitor=args.competitor, protected=protected)
    if candidate.get("order_limitation"):
        report["reasons"].append(candidate["order_limitation"])
    atomic_json(args.output / "comparison.json", report)
    write_integrity(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
