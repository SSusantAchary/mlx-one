"""Validate OmniAI corpus evidence and render it without mixing synthetic fixture protocols."""

import json
import math
import re

from benchmarks.ocr.corpus import DATASET, EVIDENCE, REVISION, accuracy, sha, summarize, write


def validate(record, entry, manifest, *, hand=False):
    from benchmarks.ocr.handwriting import (
        DATASET as HAND_DATASET,
    )
    from benchmarks.ocr.handwriting import (
        HAND_EVIDENCE,
    )
    from benchmarks.ocr.handwriting import (
        REVISION as HAND_REVISION,
    )

    evidence = HAND_EVIDENCE if hand else EVIDENCE
    count, budget = (10, 128) if hand else (100, 1023)
    for key in ("id", "checkpoint", "revision"):
        if record.get(key) != entry[key]:
            raise ValueError(f"wrong checkpoint {key}")
    schema = "mlx-one-ocr-hand10-v1" if hand else "mlx-one-ocr-omni100-v1"
    if record.get("schema_version") != schema:
        raise ValueError("unknown corpus schema")
    if record.get("status") not in {"INCOMPATIBLE", "FAILED", "MEASURED", "RUNNING"}:
        raise ValueError("unknown corpus status")
    if record["status"] != "MEASURED":
        return
    if record.get("mock"):
        raise ValueError("mock measurements cannot be published")
    dataset, revision = (HAND_DATASET, HAND_REVISION) if hand else (DATASET, REVISION)
    if record.get("dataset") != dataset or record.get("dataset_revision") != revision:
        raise ValueError("wrong pinned dataset")
    if record.get("sample_sha256") != sha(evidence / "sample.json"):
        raise ValueError("sample manifest was changed")
    fixtures = manifest["fixtures"]
    if len(fixtures) != count or len({f["id"] for f in fixtures}) != count:
        raise ValueError(f"{count} distinct corpus images required")
    samples = record["samples"]
    if [s["id"] for s in samples] != [f["id"] for f in fixtures]:
        raise ValueError(f"all {count} images required, in pinned order")
    if not record.get("hardware", {}).get("chip") or not record["hardware"]["memory_bytes"]:
        raise ValueError("hardware identity missing")
    if not record.get("software", {}).get("mlx"):
        raise ValueError("MLX version missing")
    if record.get("measurement_type") != "direct-model":
        raise ValueError("only direct-model corpus measurements belong in this table")
    protocol = record["protocol"]
    expected = {
        "batch_size": 1,
        "warmups": 1,
        "warmup_max_tokens": 16,
        "repetitions_per_image": 1,
        "images": count,
        "max_tokens": budget,
        "temperature": 0,
        "backend": "native MLX/Metal",
    }
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("incompatible corpus protocol")
    for name in ("implementation_sha256", "runner_sha256"):
        if not re.fullmatch("[a-f0-9]{64}", record.get(name, "")):
            raise ValueError("implementation/runner identity missing")
    assets = record.get("checkpoint_files", [])
    if not any(a["path"].endswith(".safetensors") for a in assets):
        raise ValueError("checkpoint weight hashes required")
    if any(not re.fullmatch("[a-f0-9]{64}", a["sha256"]) or a["bytes"] <= 0 for a in assets):
        raise ValueError("invalid checkpoint assets")
    precision = record["precision"]
    if entry["precision"] == "Affine Q4":
        if precision.get("bits") != 4 or precision.get("mode", "affine") != "affine":
            raise ValueError("wrong quantization")
    else:
        dtype = {"FP16": "float16", "BF16": "bfloat16", "FP32": "float32"}[entry["precision"]]
        if precision.get("dtypes") != [dtype]:
            raise ValueError("wrong actual floating precision")
    for sample, fixture in zip(samples, fixtures, strict=True):
        if sample["image_sha256"] != fixture["sha256"]:
            raise ValueError("wrong image identity")
        if sample["accuracy"] != accuracy(fixture["reference"], sample["transcript"]):
            raise ValueError("accuracy differs from recorded transcript")
        if sample["status"] not in {"OK", "FAILED"}:
            raise ValueError("unfinished image")
        if not math.isfinite(sample["latency_seconds"]) or sample["latency_seconds"] <= 0:
            raise ValueError("invalid measured latency")
        if sample["status"] == "OK":
            if not 0 <= sample["output_tokens"] <= protocol["max_tokens"]:
                raise ValueError("invalid actual output tokens")
            if (
                not math.isfinite(sample["peak_mlx_memory_gib"])
                or sample["peak_mlx_memory_gib"] <= 0
            ):
                raise ValueError("invalid memory measurement")
    if record.get("metrics") != summarize(samples):
        raise ValueError("summary does not match actual samples")


def load(inventory, *, hand=False):
    from benchmarks.ocr.handwriting import HAND_EVIDENCE

    evidence = HAND_EVIDENCE if hand else EVIDENCE
    manifest = json.loads((evidence / "sample.json").read_text())
    records = []
    for entry in inventory:
        path = evidence / (entry["id"] + ".json")
        if not path.exists():
            raise ValueError("missing corpus checkpoint record " + entry["id"])
        record = json.loads(path.read_text())
        validate(record, entry, manifest, hand=hand)
        records.append((entry, record))
    return records


def coverage(records, *, hand=False):
    return {
        "schema_version": (
            "mlx-one-ocr-hand10-coverage-v1" if hand else "mlx-one-ocr-omni100-coverage-v1"
        ),
        "sample_images": 10 if hand else 100,
        "checkpoints": [
            {
                "id": e["id"],
                "checkpoint": e["checkpoint"],
                "revision": e["revision"],
                "status": r["status"],
                "completed": r.get("metrics", {}).get("completed", 0),
                "reason": r.get("reason", r.get("error")),
            }
            for e, r in records
        ],
    }


def render_hand(inventory, *, write_coverage=False):
    from benchmarks.ocr.handwriting import HAND_EVIDENCE

    records = load(inventory, hand=True)
    measured = [r for _, r in records if r["status"] == "MEASURED"]
    lines = [
        "### Handwriting: 10 line images",
        "",
        "The same seeded sample of 10 test-split images from "
        "[handwritten_text_ocr]"
        "(https://huggingface.co/datasets/toghrultahirov/handwritten_text_ocr). "
        "One 16-token warmup; one request per image; image-dependent input lengths; "
        "128-token output cap. "
        "Micro CER/WER use the supplied plain-text references after NFKC and whitespace "
        "normalization; failed requests count as empty outputs.",
        "",
    ]
    if measured:
        identities = {
            (r["hardware"]["chip"], r["hardware"]["memory_bytes"], r["software"]["mlx"])
            for r in measured
        }
        if len(identities) != 1:
            raise ValueError("mixed handwriting hardware/software cohorts")
        chip, memory, mlx = next(iter(identities))
        lines += [
            f"**{chip}, {memory / 1024**3:g} GiB · MLX {mlx} · native MLX/Metal · "
            "direct-model inference · batch 1.**",
            "",
            "![OCR throughput on 10 handwritten lines]"
            "(docs/benchmarks/figures/ocr-hand10-performance.png)",
            "",
        ]
    lines += [
        "| Model | Precision | Completed | Median latency (s) | Images/s | CER ↓ | WER ↓ | "
        "Peak MLX (GiB) | Truncated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry, record in records:
        link = "docs/benchmarks/ocr/hand10-20261025/" + entry["id"] + ".json"
        label = entry["name"] + (" (printed baseline)" if entry["id"] == "O01" else "")
        if record["status"] == "MEASURED":
            m = record["metrics"]
            cells = (
                f"[{m['completed']}/10]({link}) | {m['median_latency_seconds']:.2f} | "
                f"**{m['completed_pages_per_second']:.3f}** | {m['corpus_cer']:.1%} | "
                f"{m['corpus_wer']:.1%} | {m['max_peak_mlx_memory_gib']:.3f} | {m['truncated']}"
            )
        else:
            cells = f"[Failed]({link}) | — | — | — | — | — | —"
        lines.append(f"| [{label}]({link}) | {entry['precision']} | {cells} |")
    lines += [
        "",
        "TrOCR Printed is an out-of-domain baseline. Page models process the same "
        "cropped lines in their OCR mode; this small sample is not a general handwriting "
        "qualification. Handwriting timings are separate from the 100-page workload. "
        "[Protocol, predictions, truncations and coverage]"
        "(docs/benchmarks/ocr/hand10-20261025/README.md).",
        "",
    ]
    report = coverage(records, hand=True)
    if write_coverage:
        write(HAND_EVIDENCE / "coverage.json", report)
    elif json.loads((HAND_EVIDENCE / "coverage.json").read_text()) != report:
        raise ValueError("handwriting coverage is stale")
    return lines


def render(inventory, *, write_coverage=False):
    from benchmarks.ocr.report import END, START

    records = load(inventory)
    measured = [r for _, r in records if r["status"] == "MEASURED"]
    lines = [
        START,
        "## OCR Model Performance",
        "",
        "Experimental native OCR on **100 randomly selected document images** from "
        "[OmniAI OCR Benchmark](https://huggingface.co/datasets/getomni-ai/ocr-benchmark). "
        "All page models use the same pinned sample (seed 20261025).",
        "",
    ]
    if measured:
        identities = {
            (r["hardware"]["chip"], r["hardware"]["memory_bytes"], r["software"]["mlx"])
            for r in measured
        }
        if len(identities) != 1:
            raise ValueError("separate tables required for different hardware/software cohorts")
        chip, memory, mlx = next(iter(identities))
        lines += [
            f"**{chip}, {memory / 1024**3:g} GiB unified memory · MLX {mlx} · "
            "direct-model inference · native MLX/Metal · batch 1.** One 16-token warmup, "
            "then one timed request per image; image-dependent input lengths and a "
            "1,023-token output cap, greedy decoding. Actual token counts are retained in JSON. "
            "Latency is the median of completed requests; pages/s is completed pages "
            "divided by total inference time, including failed requests. Peak memory is "
            "the maximum observed MLX allocation, including weights.",
            "",
            "![OCR throughput on 100 document images]"
            "(docs/benchmarks/figures/ocr-omni100-performance.png)",
            "",
        ]
    lines += [
        "| Model | Precision | Completed | Median latency (s) | Pages/s | "
        "Markdown-ref. CER ↓ | Peak MLX (GiB) | Truncated |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry, record in records:
        path = "docs/benchmarks/ocr/omni100-20261025/" + entry["id"] + ".json"
        label = f"[{entry['name']}]({path})"
        if record["status"] == "MEASURED":
            m = record["metrics"]
            values = (
                f"[{m['completed']}/100]({path}) | {m['median_latency_seconds']:.2f} | "
                f"**{m['completed_pages_per_second']:.3f}** | {m['corpus_cer']:.1%} | "
                f"{m['max_peak_mlx_memory_gib']:.3f} | {m['truncated']}"
            )
        else:
            status = {
                "INCOMPATIBLE": "Line crops required",
                "FAILED": "Failed",
                "RUNNING": "In progress",
            }[record["status"]]
            values = f"[{status}]({path}) | — | — | — | — | —"
        lines.append(f"| {label} | {entry['precision']} | {values} |")
    lines += [
        "",
        "CER uses NFKC and whitespace normalization against Markdown references; "
        "formatting and reading order affect this score, which can exceed 100%. "
        "Failed requests count as empty predictions. This is an MLX-One corpus test, "
        "separate from OmniAI’s official leaderboard and the synthetic OCR-S* protocol.",
        "",
        "TrOCR requires annotated line crops, absent from this page dataset. "
        "PaddleOCR-VL uses direct page transcription; its separate layout pipeline is "
        "not included. No layout accuracy or JSON extraction score is claimed by this page test.",
        "",
        "[Methodology, per-image predictions and cleanup evidence]"
        "(docs/benchmarks/ocr/omni100-20261025/README.md) · "
        "[OCR limitations and manual commands](docs/OCR_TESTING.md)",
    ]
    handwriting_directory = EVIDENCE.parent / "hand10-20261025"
    if (handwriting_directory / "sample.json").is_file() and all(
        (handwriting_directory / (entry["id"] + ".json")).is_file() for entry in inventory
    ):
        lines += ["", *render_hand(inventory, write_coverage=write_coverage)]
    lines.append(END)
    report = coverage(records)
    if write_coverage:
        write(EVIDENCE / "coverage.json", report)
    elif json.loads((EVIDENCE / "coverage.json").read_text()) != report:
        raise ValueError("corpus coverage is stale")
    return "\n".join(lines)
