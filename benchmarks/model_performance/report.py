"""Validate public benchmark evidence and deterministically generate README tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/benchmarks"
START = "<!-- model-performance:start -->"
END = "<!-- model-performance:end -->"
METRICS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "ttft_ms",
    "peak_inference_mlx_bytes",
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_samples(samples, inputs, budget, repeats):
    require(len(samples) == repeats and repeats >= 3, "need at least three measured repetitions")
    for sample in samples:
        actual = sample["generated_tokens"]
        require(sample["input_tokens"] == inputs, "actual input count mismatch")
        require(type(actual) is int and 1 <= actual <= budget, "invalid actual output count")
        require(len(sample["token_ids"]) == actual, "output token count differs from token IDs")
        require(all(type(t) is int and t >= 0 for t in sample["token_ids"]), "invalid token ID")
        require(sample["cleanup"] is True, "cache cleanup failed")
        for key in METRICS:
            value = sample[key]
            require(
                type(value) in (int, float)
                and math.isfinite(value)
                and (
                    value >= 0 if key == "decode_tokens_per_second" and actual == 1 else value > 0
                ),
                f"invalid metric {key}",
            )
        require(
            math.isclose(
                sample["prefill_tokens_per_second"], inputs * 1000 / sample["ttft_ms"], rel_tol=1e-8
            ),
            "prefill rate does not use actual input count / prefill time",
        )
        if "decode_seconds" in sample:
            require(sample["decode_seconds"] > 0, "invalid decode duration")
            require(
                math.isclose(
                    sample["decode_tokens_per_second"],
                    (actual - 1) / sample["decode_seconds"],
                    rel_tol=1e-8,
                ),
                "decode rate does not use actual decode steps",
            )


def load_records(evidence=EVIDENCE):
    inventory = read(evidence / "inventory.json")["checkpoints"]
    models = {m["model_id"]: m for m in inventory}
    support = read(ROOT / "src/mlx_one/data/compatibility/alpha_release.json")["checkpoints"]
    require(
        [(m["model_id"], m["revision"], m["primary_profile"]) for m in inventory]
        == [(m["model_id"], m["revision"], m["primary_profile"]) for m in support],
        "inventory drifted from support manifest",
    )
    cohort = read(evidence / "historical/cohort.json")
    require(
        sha(evidence / cohort["runner"]) == cohort["runner_sha256"], "historical runner changed"
    )
    paths = sorted((evidence / "historical").glob("*-performance.json"))
    require({p.name for p in paths} == set(cohort["records"]), "historical evidence set changed")
    paths += sorted((evidence / "runs").glob("*.json"))
    records = []
    seen = set()
    for path in paths:
        data = read(path)
        model = models[data["checkpoint"]]
        require(
            data["revision"] == model["revision"]
            and re.fullmatch("[0-9a-f]{40}", data["revision"]),
            "checkpoint revision mismatch",
        )
        require(data["profile"] in model["available_profiles"], "unknown precision profile")
        historical = path.parent.name == "historical"
        if historical:
            provenance = cohort["records"][path.name]
            require(sha(path) == provenance["sha256"], f"evidence changed: {path.name}")
            require(
                provenance["source_files"]["config.json"] == model["config_sha256"],
                "source config identity mismatch",
            )
            require(
                any(n.endswith(".safetensors") for n in provenance["source_files"]),
                "source weight hashes missing",
            )
            require(
                data["diagnostic_direct_model"] and not data["public_engine_benchmark"],
                "historical interface mismatch",
            )
            environment = cohort
            source_hashes = provenance["production_source_sha256"]
            runner_hash = cohort["runner_sha256"]
            prompt_source = "historical-repeated-seed"
            require(data["reference_resident"] is False, "reference model was resident")
            require(data["status"] == "MEASURED_NOT_RELEASE_APPROVED", "run was not measured")
            rows = [
                r
                for r in data["rows"]
                if r.get("context") == 128 and r.get("mode") == "production_fp16"
            ]
            require(len(rows) == 1 and data["output_budget"] == 16, "historical protocol mismatch")
            samples = rows[0]["samples"]
        else:
            environment = data["environment"]
            source_hashes = data["source_sha256"]
            runner_hash = data["runner_sha256"]
            prompt_source = data.get(
                "prompt_source", "legacy-recorded-ids:" + data["prompt_sha256"]
            )
            require(data["schema_version"] == "mlx-one-performance-v1", "unknown schema")
            require(environment["measurement_type"] == "direct-model", "interface must be separate")
            require(environment["protocol"] == "standard-128-128", "unknown protocol")
            require(data["config_sha256"] == model["config_sha256"], "config identity mismatch")
            require(
                data["weight_sha256"]
                and all(re.fullmatch("[0-9a-f]{64}", h) for h in data["weight_sha256"].values()),
                "weight hashes missing",
            )
            require(re.fullmatch("[0-9a-f]{64}", data["prompt_sha256"]), "prompt hash missing")
            require(
                len(data["prompt_token_ids"]) == 128
                and hashlib.sha256(json.dumps(data["prompt_token_ids"]).encode()).hexdigest()
                == data["prompt_sha256"],
                "prompt identity mismatch",
            )
            require(data["output_budget"] == 128, "standard output budget mismatch")
            require(re.fullmatch("[0-9a-f]{64}", data["runner_sha256"]), "runner identity missing")
            samples = data["samples"]
            require(all("decode_seconds" in s for s in samples), "decode durations missing")
        require(
            environment["hardware"]["chip_type"]
            and environment["hardware"]["physical_memory"]
            and environment["packages"]["mlx"]
            and environment["backend"],
            "hardware / MLX / backend missing",
        )
        require(environment["warmups_per_shape"] >= 1, "warmup missing")
        require(environment["statistic"] == "per-column median", "unknown statistic")
        validate_samples(samples, 128, data["output_budget"], environment["repetitions"])
        outputs = tuple(s["generated_tokens"] for s in samples)
        require(
            source_hashes and all(re.fullmatch("[0-9a-f]{64}", h) for h in source_hashes.values()),
            "production source identity missing",
        )
        source_fingerprint = hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest()
        group = (
            source_fingerprint,
            runner_hash,
            prompt_source,
            environment["protocol"],
            environment["backend"],
            json.dumps(environment["hardware"], sort_keys=True),
            json.dumps(environment["packages"], sort_keys=True),
            environment["platform"],
            environment["python"],
            environment["repetitions"],
            environment["warmups_per_shape"],
            outputs,
        )
        key = (model["id"], data["profile"], group)
        require(key not in seen, "duplicate comparable run; explicitly curate one run per cohort")
        seen.add(key)
        primary = data["profile"] == model["primary_profile"] or (
            data["profile"] == "stored" and not model["primary_profile"].startswith("affine")
        )
        records.append(
            dict(
                source_fingerprint=source_fingerprint,
                runner_hash=runner_hash,
                model=model,
                data=data,
                samples=samples,
                environment=environment,
                group=group,
                primary=primary,
                path=path.relative_to(ROOT).as_posix(),
            )
        )
    return inventory, records


def render(inventory, records):
    parts = [
        START,
        "## Model Performance",
        "",
        "Actual **direct-model inference** measurements; experimental, batch size 1. "
        "Python Engine, CLI and HTTP serving throughput are separate measurement types "
        "and are not represented by these numbers. Parameters are nominal model sizes.",
        "",
        "**Historical 128-input / 16-output protocol.** These archived runs are not "
        "the new standardized 128-input / 128-output results. Source and primary "
        "precision measurements are listed separately; “source” means the stored weights.",
        "",
    ]
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    for group_records in groups.values():
        env = group_records[0]["environment"]
        outputs = [s["generated_tokens"] for s in group_records[0]["samples"]]
        parts += [
            f"**{env['protocol']}** — {env['hardware']['chip_type']}, "
            f"{env['hardware']['physical_memory']}; {env['platform']}; "
            f"Python {env['python']}; MLX {env['packages']['mlx']} / "
            f"Metal {env['packages'].get('mlx-metal', 'unrecorded')}. "
            f"Backend: {env['backend']}. Actual input: 128 tokens; actual output: "
            f"{', '.join(map(str, sorted(set(outputs))))} tokens per repetition. "
            f"{env['warmups_per_shape']} shape warmup; {env['repetitions']} measured "
            "repetitions; per-column median. FP16 KV; fresh state, no prefix reuse. "
            "TTFT includes prefill and greedy selection of the first token. "
            "Peak MLX resets before each request after load/conversion/warmup; "
            "it includes resident model allocations. "
            f"Implementation `{group_records[0]['source_fingerprint'][:12]}`; "
            f"runner `{group_records[0]['runner_hash'][:12]}`.",
            "",
            "| Model name | Parameters | Quantization/precision | Prefill tokens/sec | "
            "Decode tokens/sec | Time to first token (ms) | Peak MLX memory (GiB) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in group_records:
            m = r["model"]
            precision = (
                m["primary_precision"] if r["primary"] else m["source_precision"] + " (source)"
            )
            values = [statistics.median(s[k] for s in r["samples"]) for k in METRICS]
            parts.append(
                f"| [{m['model_id']}]({r['path']}) | {m['parameters']} | {precision} | "
                f"{values[0]:.2f} | "
                + (
                    f"{values[1]:.2f}"
                    if all(s["generated_tokens"] > 1 for s in r["samples"])
                    else "—"
                )
                + f" | {values[2]:.2f} | "
                f"{values[3] / 1024**3:.3f} |"
            )
        parts.append("")
    standard = {
        r["model"]["id"]
        for r in records
        if r["primary"]
        and r["environment"]["protocol"] == "standard-128-128"
        and all(s["generated_tokens"] == 128 for s in r["samples"])
    }
    if standard:
        parts += [
            f"**Standardized 128/128 coverage: {len(standard)} / {len(inventory)} "
            "primary configurations.** Remaining runs are pending; see the coverage report.",
            "",
        ]
    if not standard:
        parts += [
            "**Standardized 128-input / 128-output results: pending.** "
            "No existing qualifying measurements use this protocol. Early EOS runs "
            "retain actual output counts and cannot populate a 128-output comparison.",
            "",
        ]
    missing = [
        m["model_id"]
        for m in inventory
        if not any(r["model"]["id"] == m["id"] and r["primary"] for r in records)
    ]
    parts += [
        "**Missing primary-precision performance:** "
        + ", ".join(f"`{m}`" for m in missing)
        + ". No speed or memory figures are inferred for these checkpoints.",
        "",
        "See the [methodology and manual commands](docs/benchmarks/README.md), "
        "[checkpoint/precision inventory](docs/benchmarks/inventory.md), and "
        "[coverage report](docs/benchmarks/coverage.md). Full samples, output token IDs, "
        "immutable revisions, source hashes and hardware evidence live in "
        "[docs/benchmarks](docs/benchmarks/).",
        END,
    ]
    return "\n".join(parts)


def derived(inventory, records):
    intro = "# Text inference checkpoint inventory\n\nPrimary means the support manifest’s "
    intro += "selected configuration, not certification. Q4/Q8 conversions use group size 64. "
    intro += "Source BF16/FP32 runs are distinct. Parameter sizes are nominal labels; "
    intro += "quantized bytes are not parameter counts.\n\n"
    lines = [
        intro,
        "| ID / checkpoint | Revision | Parameters | Primary precision | "
        "Available benchmark profiles | Functional status |",
        "|---|---|---|---|---|---|",
    ]
    coverage = [
        "# Benchmark coverage\n",
        "All 13 pinned text inference checkpoints are inventoried. Other training, "
        "audio, vision and discovery candidates are outside this comparison. "
        "Performance evidence does not change functional admission status.\n",
        "| Checkpoint | Historical primary 128/16 | Standard primary 128/128 | Gap |",
        "|---|---|---|---|",
    ]
    result = []
    for m in inventory:
        measured = [r for r in records if r["model"]["id"] == m["id"]]
        primary = [r for r in measured if r["primary"]]
        old = any(r["environment"]["protocol"] == "historical-128-16" for r in primary)
        new = any(
            r["environment"]["protocol"] == "standard-128-128"
            and all(s["generated_tokens"] == 128 for s in r["samples"])
            for r in primary
        )
        profiles = ", ".join(sorted({r["data"]["profile"] for r in measured})) or "None"
        lines.append(
            f"| {m['id']} `{m['model_id']}` | `{m['revision']}` | {m['parameters']} | "
            f"{m['primary_precision']} | {profiles} | {m['status']} / {m['checkpoint_status']} |"
        )
        gap = (
            (m["benchmark_gap"] or "Primary performance missing.")
            if not primary
            else ("Standardized run pending." if not new else "None")
        )
        coverage.append(
            f"| `{m['model_id']}` | {'Measured' if old else 'Missing'} | "
            f"{'Measured' if new else 'Missing'} | {gap} |"
        )
        result.append(
            dict(
                model_id=m["model_id"],
                revision=m["revision"],
                historical_primary=old,
                standardized_primary=new,
                records=[r["path"] for r in measured],
                gap=gap,
            )
        )
    return {
        "inventory.md": "\n".join(lines) + "\n",
        "coverage.md": "\n".join(coverage) + "\n",
        "coverage.json": json.dumps({"checkpoints": result}, indent=2) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="validate evidence and generated files"
    )
    parser.add_argument("--write", action="store_true", help="update README and coverage files")
    args = parser.parse_args()
    inventory, records = load_records()
    table = render(inventory, records)
    readme = ROOT / "README.md"
    text = readme.read_text()
    require(text.count(START) == text.count(END) == 1, "README needs one generated table block")
    replacement = text[: text.index(START)] + table + text[text.index(END) + len(END) :]
    files = {
        readme: replacement,
        **{EVIDENCE / k: v for k, v in derived(inventory, records).items()},
    }
    for path, content in files.items():
        if args.write:
            path.write_text(content)
        elif args.check:
            require(path.exists() and path.read_text() == content, f"stale generated file: {path}")
    if not args.write and not args.check:
        print(table)
    else:
        print(
            f"Validated {len(records)} measurements across {len(inventory)} checkpoint revisions."
        )


if __name__ == "__main__":
    main()
