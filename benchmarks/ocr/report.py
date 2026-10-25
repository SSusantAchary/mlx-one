"""Generate README OCR tables only from selected, validated JSON measurements."""

import argparse
import json
from pathlib import Path

from benchmarks.ocr.metrics import validate

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/benchmarks/ocr"
START = "<!-- MLX-ONE-OCR-BENCHMARK-START -->"
END = "<!-- MLX-ONE-OCR-BENCHMARK-END -->"


def render(inventory, records):
    measured = {}
    for record, path in records:
        if record.get("mock"):
            continue
        validate(record)
        if record["status"] != "MEASURED":
            continue
        key = (
            record["checkpoint"],
            record["revision"],
            record["task"],
            record["workload"],
            record["measurement_type"],
        )
        if key in measured:
            raise ValueError("duplicate OCR measurement selection")
        measured[key] = (record, path)
    lines = [
        START,
        "## OCR Model Performance",
        "",
        "Native OCR is experimental. Full checkpoint image tests and M4 performance runs are "
        "**pending**; synthetic integration tests and tensor-header checks are not OCR accuracy "
        "measurements. “—” means not measured.",
        "",
    ]
    for title, entries in [
        ("Line recognition", [m for m in inventory if m["family"] == "trocr"]),
        ("Page transcription and regions", [m for m in inventory if m["family"] != "trocr"]),
    ]:
        lines += [
            "### " + title,
            "",
            "| Model | Precision | Task | Latency/request (ms) | "
            + ("Images/s" if title == "Line recognition" else "Pages/s")
            + " | CER | "
            "Peak MLX (GiB) | Status |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
        notes = []
        for model in entries:
            candidates = [
                (r, p)
                for r, p in measured.values()
                if r["checkpoint"] == model["checkpoint"]
                and r["revision"] == model["revision"]
                and r["task"] == model["primary_task"]
                and r["workload"] == model["primary_workload"]
                and r["measurement_type"] == "direct-model"
            ]
            if len(candidates) > 1:
                raise ValueError("ambiguous publication selection")
            if candidates:
                record, path = candidates[0]
                precision = record.get("precision", {})
                if model["precision"] == "Affine Q4":
                    if precision.get("bits") != 4 or precision.get("mode", "affine") != "affine":
                        raise ValueError("published precision differs from primary checkpoint")
                else:
                    expected = {"FP32": "float32", "FP16": "float16", "BF16": "bfloat16"}[
                        model["precision"]
                    ]
                    if precision.get("mode") != "floating" or precision.get("dtypes") != [expected]:
                        raise ValueError("published floating dtype differs from checkpoint")
                m = record["metrics"]
                status = f"Measured; experimental; {record['workload']}"
                values = (
                    f"{m['latency_seconds'] * 1000:.2f} | **{m['pages_per_second']:.3f}** | "
                    f"{m['cer']:.4f} | {m['peak_mlx_memory_gib']:.3f}"
                )
                label = f"[{model['name']}]({path})"
                # Identify every cohort; never pool hardware, precision, protocol or workloads.
            else:
                values = "— | — | — | —"
                status = "Not tested"
                label = model["name"]
            lines.append(
                f"| {label} | {model['precision']} | {model['scope']} | {values} | {status} |"
            )
            if candidates:
                notes += [
                    f"{model['name']}: {record['hardware']['chip']}, "
                    f"{record['hardware']['memory_bytes'] / 1024**3:g} GiB; "
                    f"MLX {record['software']['mlx']}; "
                    f"{record['measurement_type']}; native MLX/Metal; "
                    f"{record['workload']}; batch 1; "
                    "one warmup, five repetitions, per-column median. "
                    f"Exact settings: [{record['workload']}]({path}).",
                    "",
                ]
        lines += ["", *notes, ""]
    lines += [
        "TrOCR accepts cropped lines; Florence returns geometry only in region mode. "
        "PaddleOCR-VL currently provides page/crop transcription and task-specific table/formula "
        "generation; the separate Paddle layout detector and full structured page pipeline are "
        "not included. Text-only outputs carry no invented boxes or confidence scores.",
        "",
        "[OCR implementation, limitations and manual commands](docs/OCR_TESTING.md) · "
        "[Benchmark methodology and missing coverage](docs/benchmarks/ocr/README.md)",
        END,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    inventory = json.loads((EVIDENCE / "inventory.json").read_text())["checkpoints"]
    selection = json.loads((EVIDENCE / "publication.json").read_text())
    records = [(json.loads((ROOT / path).read_text()), path) for path in selection["records"]]
    corpus_directory = EVIDENCE / "omni100-20261025"
    corpus_selected = (corpus_directory / "sample.json").is_file() and all(
        (corpus_directory / (model["id"] + ".json")).is_file() for model in inventory
    )
    if corpus_selected:
        from benchmarks.ocr.corpus_report import render as render_corpus

        block = render_corpus(inventory, write_coverage=args.write)
    else:
        block = render(inventory, records)
    readme = ROOT / "README.md"
    original = readme.read_text()
    if START in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError("ambiguous OCR markers")
        a, tail = original.split(START)
        _, b = tail.split(END)
        updated = a + block + b
    else:
        if args.check:
            raise ValueError("README has no OCR section")
        anchor = "## Architecture\n"
        if original.count(anchor) != 1:
            raise ValueError("README insertion anchor missing")
        updated = original.replace(anchor, block + "\n\n" + anchor)
    if args.write:
        readme.write_text(updated)
    elif args.check:
        if original != updated:
            raise ValueError("README OCR section is stale")
    else:
        print(block)
    coverage = {
        "primary_checkpoints": len(inventory),
        "measured": sum(r.get("status") == "MEASURED" and not r.get("mock") for r, _ in records),
        "missing": [
            {
                "checkpoint": m["checkpoint"],
                "revision": m["revision"],
                "task": m["primary_task"],
                "workload": m["primary_workload"],
            }
            for m in inventory
            if not any(
                r.get("status") == "MEASURED"
                and not r.get("mock")
                and r["checkpoint"] == m["checkpoint"]
                and r["revision"] == m["revision"]
                and r["task"] == m["primary_task"]
                and r["workload"] == m["primary_workload"]
                and r["measurement_type"] == "direct-model"
                for r, _ in records
            )
        ],
    }
    text = json.dumps(coverage, indent=2) + "\n"
    if args.write:
        (EVIDENCE / "coverage.json").write_text(text)
    elif args.check:
        if (EVIDENCE / "coverage.json").read_text() != text:
            raise ValueError("OCR coverage is stale")
    print(
        f"Validated {len(records)} synthetic OCR fixture records; "
        f"{coverage['measured']} measured; {len(coverage['missing'])} unmeasured."
    )
    if corpus_selected:
        print("Validated selected real-image corpus records and README tables separately.")


if __name__ == "__main__":
    main()
