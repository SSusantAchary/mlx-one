"""Derive reviewable summaries and CSVs from retained, unfiltered measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def requests(receipt):
    for session in receipt["sessions"]:
        for sample in session["samples"]:
            yield session, sample


def summary(receipt):
    rows = []
    for model in receipt["models"]:
        sessions = [s for s in receipt["sessions"] if s["model"] == model["key"]]
        backends = sorted({s["backend"] for s in sessions})
        by_backend = {
            b: {
                sample["prompt_index"]: sample
                for s in sessions
                if s["backend"] == b
                for sample in s["samples"]
            }
            for b in backends
        }
        shared = set.intersection(*(set(v) for v in by_backend.values())) if by_backend else set()
        full = sorted(
            i
            for i in shared
            if all(
                v[i]["completion_tokens"] == receipt["protocol"]["output_budget"]
                and v[i]["finish_reason"] == "length"
                for v in by_backend.values()
            )
        )
        for backend, values in by_backend.items():
            for view, selected in [("unfiltered", list(values)), ("shared_full_length", full)]:
                samples = [values[i] for i in selected]
                if not samples:
                    continue
                row = {
                    "model": model["key"],
                    "backend": backend,
                    "view": view,
                    "samples": len(samples),
                    "prompt_indices": sorted(selected),
                    "processes": sum(s["backend"] == backend for s in sessions),
                    "output_token_range": [
                        min(s["completion_tokens"] for s in samples),
                        max(s["completion_tokens"] for s in samples),
                    ],
                }
                for key in ["ttft_ms", "latency_seconds", "output_tokens_per_second"]:
                    numbers = [s[key] for s in samples]
                    row[key] = {
                        "median": statistics.median(numbers),
                        "min": min(numbers),
                        "max": max(numbers),
                        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else None,
                    }
                row["peak_rss_bytes"] = max(
                    s.get("peak_rss_bytes", 0) for s in sessions if s["backend"] == backend
                )
                rows.append(row)
    return rows


def seal(directory):
    receipt = json.loads((directory / "results.json").read_text())
    fields = [
        "model",
        "backend",
        "round",
        "prompt_index",
        "prompt_sha256",
        "completion_tokens",
        "finish_reason",
        "ttft_ms",
        "latency_seconds",
        "output_tokens_per_second",
    ]
    with (directory / "per_request.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session, sample in requests(receipt):
            writer.writerow(
                {
                    **{k: session[k] for k in ["model", "backend", "round"]},
                    **{k: sample[k] for k in fields if k in sample},
                }
            )
    provenance = json.loads((ROOT / "local/mlxboost/sources/sources.json").read_text())
    manifest = {k: v for k, v in receipt.items() if k != "sessions"}
    manifest.update(
        status="PASS" if all(s["status"] == "PASS" for s in receipt["sessions"]) else "FAIL",
        source_pins=provenance,
        evidence_sha256=hashlib.sha256((directory / "results.json").read_bytes()).hexdigest(),
        thermal="not sampled",
        power_state="not sampled",
        memory_method="RSS sampled every 20 ms, including load/warmup; MLX metrics separate",
        original_evidence_untouched=True,
    )
    one_commands = [s["command"] for s in receipt["sessions"] if s["backend"] == "mlx-one"]
    manifest["flags"] = {
        "fast_decode": any("--mlxboost-fast-decode" in c for c in one_commands),
        "inference_modes": sorted({c[c.index("--inference-mode") + 1] for c in one_commands}),
        "other_mlxboost_features": "all disabled",
    }
    manifest["metadata_limits"] = (
        "Original runner captures production Python hashes and server commands; "
        "it does not snapshot the untracked wrapper or full dirty tree at launch. "
        "Initial dirty-tree audit and native manifests are retained separately."
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary(receipt)


def paired_process_bootstrap(off, on):
    def group(r):
        return {(s["model"], s["round"], v["prompt_index"]): v for s, v in requests(r)}

    left, right = group(off), group(on)
    common = sorted(set(left) & set(right))
    ratios = {}
    exact = 0
    for key in common:
        a, b = left[key], right[key]
        if a["prompt_sha256"] != b["prompt_sha256"]:
            raise ValueError("paired prompt identity differs")
        exact += (
            a["text"] == b["text"]
            and a["completion_tokens"] == b["completion_tokens"]
            and a["finish_reason"] == b["finish_reason"]
        )
        if (
            a["completion_tokens"] == b["completion_tokens"] == 128
            and a["finish_reason"] == b["finish_reason"] == "length"
        ):
            ratios.setdefault(key[:2], []).append(
                b["output_tokens_per_second"] / a["output_tokens_per_second"]
            )
    processes = [statistics.median(v) for v in ratios.values()]
    if not processes:
        return {"paired_requests": len(common), "exact_output_matches": exact}
    rng = random.Random(42)
    bootstrap = sorted(
        statistics.median(rng.choices(processes, k=len(processes))) for _ in range(10000)
    )
    return {
        "paired_requests": len(common),
        "exact_output_matches": exact,
        "process_ratios": processes,
        "median_process_ratio": statistics.median(processes),
        "bootstrap_95_percent_interval": [bootstrap[250], bootstrap[9749]],
        "limitation": "resampled process medians; small process count and unmeasured thermals",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    historical = ROOT / "local/competitor-comparison/runs/comparison-20261101/results.json"
    data["frozen_historical"] = summary(json.loads(historical.read_text()))
    baseline = ROOT / "local/mlxboost/runs/baseline-original-20261101"
    data["reproduction"] = seal(baseline)
    for path in sorted(args.confirmation.glob("*/results.json")):
        data[path.parent.name] = seal(path.parent)
    for model in ["qwen2", "qwen3"]:
        off = args.confirmation / f"expanded-{model}-off/results.json"
        on = args.confirmation / f"expanded-{model}-on/results.json"
        if off.exists() and on.exists():
            data[f"paired-{model}"] = paired_process_bootstrap(
                json.loads(off.read_text()), json.loads(on.read_text())
            )
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
