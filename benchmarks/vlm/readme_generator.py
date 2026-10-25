"""Validate VLM evidence and update only the dedicated README marker block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.vlm.metrics import EVIDENCE, ROOT, read, require, validate

START = "<!-- MLX-ONE-VLM-BENCHMARK-START -->"
END = "<!-- MLX-ONE-VLM-BENCHMARK-END -->"


def records(evidence=EVIDENCE):
    plan = read(evidence / "benchmark_plan.json")
    result = [validate(read(path), plan) for path in sorted((evidence / "results").glob("*.json"))]
    seen = set()
    for record in result:
        key = (record["checkpoint"], record["workload"])
        require(
            key not in seen, "duplicate checkpoint/workload; select one immutable run explicitly"
        )
        seen.add(key)
    return plan, result


def table(models, result, workload="VLM-S1"):
    lines = [
        "| Model | Precision | Images | Vision processing (ms) | Prefill TPS | Decode TPS "
        "| TTFT (ms) | Peak MLX memory (GiB) | Scores |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model in models:
        record = next((r for r in result if r["checkpoint"] == model["checkpoint"]), None)
        measured = record is not None and record["status"] == "MEASURED"

        def metric(key, bold=False, record=record, measured=measured):
            value = record["metrics"][key] if measured else None
            if value is None:
                return "—"
            formatted = f"{value:.3f}" if key == "peak_mlx_memory_gib" else f"{value:.2f}"
            return "**" + formatted + "**" if bold else formatted

        evidence_path = f"docs/benchmarks/vlm/results/{model['id']}-{workload}.json"
        label = f"[{model['name']}]({evidence_path})"
        status = "Scores" if measured else "Pending hardware test"
        lines.append(
            f"| {label} | {model['precision']} | 1 | {metric('preprocessing_ms')} "
            f"| {metric('prefill_tps', True)} | {metric('decode_tps', True)} "
            f"| {metric('ttft_ms')} | {metric('peak_mlx_memory_gib')} "
            f"| [{status}]({evidence_path}) |"
        )
    return "\n".join(lines)


def render(plan, result):
    primary_workload = plan.get("readme_workload", "VLM-S1")
    hf = primary_workload.startswith("VLM-HF-")
    standard = [r for r in result if r["workload"] == primary_workload]
    measured = [r for r in standard if r["status"] == "MEASURED"]
    lines = ["## Vision-Language Model Performance", ""]
    if measured:
        lines += [
            "Actual **direct-model native MLX/Metal** inference on pinned checkpoint "
            "revisions. These are image-inference measurements; Python Engine, CLI and HTTP "
            "serving overhead are separate. Functional and quality limits are recorded with "
            "the evidence.",
            "",
        ]
    else:
        lines += [
            "Native static-image source paths are implemented for the five checkpoints "
            "below. Exact checkpoint execution is **pending hardware testing**; synthetic "
            "tests do not establish model quality or speed.",
            "",
        ]
    if hf:
        lines += [
            "**VLM-HF-S1 v1:** one 512 × 512 padded image from "
            "[Hugging Face documentation-images](https://huggingface.co/datasets/"
            "huggingface/documentation-images), with dataset revision and image hashes pinned "
            "in the [workload manifest](benchmarks/vlm/hf-images.json). "
        ]
    else:
        lines += ["**VLM-S1 v1:** one 512 × 512 synthetic source image. "]
    lines[-1] += (
        "Batch 1, fixed prompt, greedy generation up to 128 emitted tokens; one "
        "warmup and five repetitions, per-column median. EOS is respected and actual output "
        "counts determine speed. Each checkpoint retains its own resize/tiling. TTFT includes "
        "image decoding/preprocessing; prefill combines vision encoding, projection and "
        "language prefill. “Vision processing” means image decoding/preprocessing only."
    )
    lines += [
        "",
        "Q4/Q8 denote the published checkpoint format; vision or projector "
        "modules may retain floating-point weights.",
        "",
    ]
    if measured and primary_workload == "VLM-HF-S1":
        lines += [
            "![MLX-One VLM decode speed by nominal model size, with time to first token]"
            "(docs/benchmarks/figures/vlm-model-performance.png)",
            "",
            "[VLM size-versus-speed scatter plot]"
            "(docs/benchmarks/figures/vlm-model-size-vs-decode.svg) · "
            "[VLM chart SVG](docs/benchmarks/figures/vlm-model-performance.svg) · "
            "[Chart data and reproduction]"
            "(docs/benchmarks/figures/README.md#vlm-and-audio-charts)",
            "",
        ]
    cohorts = {}
    for record in measured:
        hardware = record["hardware"]
        key = (
            hardware["device"],
            hardware["memory_bytes"],
            hardware["os"],
            record["mlx_version"],
            record["python_version"],
            record["implementation_sha256"],
            record["runner_sha256"],
        )
        cohorts.setdefault(key, []).append(record)
    for key, group in cohorts.items():
        lines += [
            f"**Measured hardware:** {key[0]}, {key[1] / 1024**3:g} GiB; {key[2]}; "
            f"MLX {key[3]}; Python {key[4]}. Separate execution cohort.",
            "",
            table(
                [
                    m
                    for m in plan["checkpoints"]
                    if any(r["checkpoint"] == m["checkpoint"] for r in group)
                ],
                group,
                primary_workload,
            ),
            "",
        ]
    pending = [
        m
        for m in plan["checkpoints"]
        if not any(r["checkpoint"] == m["checkpoint"] for r in measured)
    ]
    if pending:
        lines += [
            table(pending, [], primary_workload),
            "",
            "Hardware target: Apple M4 with 32 GB unified memory. Runtime versions and actual "
            "processed image/token counts will be recorded when measured; "
            "no hardware performance is inferred.",
            "",
        ]
    if measured:
        counts = []
        for model in plan["checkpoints"]:
            record = next((r for r in measured if r["checkpoint"] == model["checkpoint"]), None)
            if record is None:
                continue

            def span(values):
                unique = sorted(set(values))
                return str(unique[0]) if len(unique) == 1 else f"{unique[0]}–{unique[-1]}"

            inputs = span([s["prompt_tokens"] for s in record["samples"]])
            outputs = span([s["output_tokens"] for s in record["samples"]])
            counts.append(f"{model['name']}: {inputs} / {outputs}")
        lines += [
            "**Actual fused input / emitted output tokens per repetition:** "
            + "; ".join(counts)
            + ".",
            "",
        ]
    lines += [
        "",
        "High-resolution (1024 × 1024), four-image and OCR workloads remain separate. "
        "The HF image protocol is not pooled with the earlier synthetic-image protocol. "
        "See the [methodology and coverage](docs/benchmarks/vlm/README.md), "
        "[pinned checkpoint inventory](docs/benchmarks/vlm/benchmark_plan.json), and "
        "[manual download/inference/benchmark runbook](docs/VLM_TESTING.md). "
        "Text-model measurements above use different protocols. "
        "See the [checkpoint smoke and cleanup report]"
        "(docs/validation/vlm-hf-2026-10-24/report.md).",
    ]
    return "\n".join(lines)


def update(text, block):
    require(
        text.count(START) == 1 and text.count(END) == 1 and text.index(START) < text.index(END),
        "README must contain exactly one ordered VLM marker pair",
    )
    start, end = text.index(START) + len(START), text.index(END)
    return text[:start] + "\n" + block + "\n" + text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan, result = records(args.evidence)
    original = args.readme.read_text()
    generated = update(original, render(plan, result))
    if args.check:
        require(generated == original, "README VLM table is stale")
    else:
        args.readme.write_text(generated)
    coverage = {
        "source_status": "NATIVE_IMPLEMENTED",
        "checkpoint_status": (
            "CHECKPOINT_SMOKE_PASSED"
            if all(
                m.get("checkpoint_status") == "CHECKPOINT_SMOKE_PASSED" for m in plan["checkpoints"]
            )
            else "PENDING_HARDWARE_TEST"
        ),
        "measured": sum(
            r["status"] == "MEASURED"
            and r["workload"] in plan.get("active_workloads", plan["workloads"])
            for r in result
        ),
        "planned": len(plan["checkpoints"]) * len(plan.get("active_workloads", plan["workloads"])),
        "active_workloads": plan.get("active_workloads", plan["workloads"]),
        "missing": [
            {
                "checkpoint": m["checkpoint"],
                "revision": m["revision"],
                "workloads": [
                    w
                    for w in plan.get("active_workloads", plan["workloads"])
                    if not any(
                        r["checkpoint"] == m["checkpoint"]
                        and r["workload"] == w
                        and r["status"] == "MEASURED"
                        for r in result
                    )
                ],
            }
            for m in plan["checkpoints"]
        ],
    }
    coverage["missing"] = [m for m in coverage["missing"] if m["workloads"]]
    if not args.check:
        (args.evidence / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(
        f"Validated {len(result)} records; {coverage['measured']} measured, "
        f"{len(coverage['missing'])} checkpoints awaiting evidence."
    )


if __name__ == "__main__":
    main()
