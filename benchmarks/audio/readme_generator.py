"""Publish only validated, explicitly selected audio results inside README markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.audio.result_schema import validate

START = "<!-- MLX-ONE-AUDIO-BENCHMARK-START -->"
END = "<!-- MLX-ONE-AUDIO-BENCHMARK-END -->"


def render(catalog, records):
    for record in records:
        validate(record)
    lines = [
        START,
        "## Audio Model Performance",
        "",
        "Native audio source support is experimental. **No standardized audio checkpoint "
        "benchmarks have been run yet**; “—” means unmeasured. Historical corpus results "
        "are [documented separately](docs/audio-qualification-m4-32gb.md).",
        "",
        "Protocol: batch 1, one warmup, five repetitions, median; actual audio durations. "
        "ASR uses a fixed 10-second clip (ASR-S1); TTS uses fixed 30-word text (TTS-S1). "
        "RTF is processing seconds divided by input audio seconds for ASR, or generated "
        "audio seconds for synthesis. Lower RTF is faster. Hardware and MLX version "
        "are listed for each measured cohort. Direct-model, Python Engine and full "
        "file-pipeline runs remain separate.",
        "",
    ]
    tables = [
        (
            "Speech Recognition Performance",
            {"transcribe"},
            [
                ("Audio (s)", "audio_seconds", 1),
                ("Latency (ms)", "processing_seconds", 1000),
                ("RTF ↓", "real_time_factor", 1),
                ("Audio sec/sec ↑", "audio_seconds_per_second", 1),
                ("WER (%)", "wer", 100),
                ("Peak MLX (GiB)", "peak_mlx_gib", 1),
            ],
        ),
        (
            "Forced Alignment Performance",
            {"align"},
            [
                ("Audio (s)", "audio_seconds", 1),
                ("Latency (ms)", "processing_seconds", 1000),
                ("Mean boundary error (ms)", "alignment_mean_error_ms", 1),
                ("Peak MLX (GiB)", "peak_mlx_gib", 1),
            ],
        ),
        (
            "Speech Generation Performance",
            {"synthesize", "speech-to-speech"},
            [
                ("First audio (ms)", "time_to_first_audio_ms", 1),
                ("Output (s)", "output_audio_seconds", 1),
                ("RTF ↓", "real_time_factor", 1),
                ("Audio sec/sec ↑", "audio_seconds_per_second", 1),
                ("Peak MLX (GiB)", "peak_mlx_gib", 1),
            ],
        ),
    ]
    for title, tasks, columns in tables:
        labels = ["Model", "Precision"] + (["Task"] if len(tasks) > 1 else [])
        labels += [name for name, _, _ in columns] + ["Scores"]
        lines += [f"### {title}", ""]
        if tasks == {"transcribe"} and any(
            r["status"] == "MEASURED" and r["task"] in tasks for r in records
        ):
            lines += [
                "![MLX-One speech recognition throughput by nominal model size, "
                "with single-clip WER and peak memory]"
                "(docs/benchmarks/figures/audio-asr-performance.png)",
                "",
                "[ASR size-versus-throughput scatter plot]"
                "(docs/benchmarks/figures/audio-model-size-vs-asr.svg) · "
                "[ASR chart SVG](docs/benchmarks/figures/audio-asr-performance.svg)",
                "",
            ]
        if "synthesize" in tasks and any(
            r["status"] == "MEASURED" and r["task"] in tasks for r in records
        ):
            lines += [
                "![MLX-One speech generation throughput, with separate TTS and "
                "speech-to-speech workloads and the failed longer case]"
                "(docs/benchmarks/figures/audio-speech-generation.png)",
                "",
                "[Speech generation chart SVG]"
                "(docs/benchmarks/figures/audio-speech-generation.svg) · "
                "[Chart data and reproduction]"
                "(docs/benchmarks/figures/README.md#vlm-and-audio-charts). "
                "Speech quality and alignment accuracy retain their documented limits.",
                "",
            ]
        lines += [
            "| " + " | ".join(labels) + " |",
            "| " + " | ".join("---" for _ in labels) + " |",
        ]
        notes, cohorts, inputs = [], set(), {}
        for entry in catalog["entries"]:
            if entry["task"] not in tasks:
                continue
            candidates = [
                r
                for r in records
                if r["checkpoint"] == entry["checkpoint"]
                and r["revision"] == entry["revision"]
                and r["task"] == entry["task"]
                and r["workload"] == entry["workload"]
                and r["status"] == "MEASURED"
            ]
            if len(candidates) > 1:
                raise ValueError("select one measurement per checkpoint/task in publication.json")
            record = candidates[0] if candidates else None
            failed = next(
                (
                    r
                    for r in records
                    if r["checkpoint"] == entry["checkpoint"]
                    and r["revision"] == entry["revision"]
                    and r["task"] == entry["task"]
                    and r["workload"] == entry["workload"]
                    and r.get("attempt_status") == "FAILED"
                ),
                None,
            )
            if record and record["precision"] != entry["precision"]:
                raise ValueError("measurement precision differs from checkpoint inventory")
            metrics = record["metrics"] if record else {}
            label = entry["name"]
            if record and record.get("_evidence_path"):
                label = f"[{label}]({record['_evidence_path']})"
            elif failed and failed.get("_evidence_path"):
                label = f"[{label}]({failed['_evidence_path']})"
            cells = [label, entry["precision"]] + ([entry["task"]] if len(tasks) > 1 else [])
            for _, key, scale in columns:
                value = metrics.get(key)
                number = f"{value * scale:.3f}" if value is not None else "—"
                cells.append(
                    f"**{number}**"
                    if key == "audio_seconds_per_second" and value is not None
                    else number
                )
            cells.append(
                "Failed; WIP"
                if failed and not record
                else "Scores"
                if record and entry.get("status") == "WIP"
                else "Scores"
                if record
                else "Pending; WIP"
                if entry.get("status") == "WIP"
                else "Pending"
            )
            evidence = record if record else failed
            evidence_path = evidence.get("_evidence_path") if evidence else None
            target = evidence_path or "docs/benchmarks/audio/coverage.json"
            cells[-1] = f"[{cells[-1]}]({target})"
            lines.append("| " + " | ".join(cells) + " |")
            if record:
                identity = json.dumps(record["input"], sort_keys=True)
                key = (record["task"], record["workload"])
                if key in inputs and inputs[key] != identity:
                    raise ValueError("different input fixtures/settings in one benchmark workload")
                inputs[key] = identity
                cohorts.add(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "hardware",
                                "software",
                                "measurement_type",
                                "implementation_sha256",
                            )
                        },
                        sort_keys=True,
                    )
                )
                notes.append(
                    f"Measured on {record['hardware']['chip']}, "
                    f"{record['hardware']['memory_bytes'] / 1024**3:.0f} GiB; MLX "
                    f"{record['software']['mlx']}; native MLX/Metal; {record['measurement_type']}; "
                    f"{record['workload']}."
                )
        if len(cohorts) > 1:
            raise ValueError("incompatible hardware/software/implementation cohorts in one table")
        lines += ["", *dict.fromkeys(notes), ""]
    if any(record["status"] == "MEASURED" for record in records):
        lines[3] = (
            "Native audio support is experimental. Values below are validated "
            "measurements; “—” means unavailable. Recognition uses one pinned English "
            "LibriSpeech utterance (9.9 seconds, padded to 10); its WER is a single-clip "
            "check, not corpus accuracy. See the linked raw evidence."
        )
    for record in records:
        if record["workload"] != "STS-HF-Q1" or record["status"] != "MEASURED":
            continue
        m = record["metrics"]
        lines += [
            "### Speech-to-speech: separate short-question workload",
            "",
            "**STS-HF-Q1:** a different 2.04-second spoken question from the same pinned "
            "dataset, without padding; greedy generation, one warmup, five repetitions, "
            "median. This completed case does not replace the failed 10-second STS-S1 run.",
            "",
            "| Model | First audio (ms) | Output (s) | RTF ↓ | Audio sec/sec ↑ | Peak MLX (GiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| [LFM2.5 Audio 1.5B Q4]({record.get('_evidence_path', '')}) | "
            f"{m['time_to_first_audio_ms']:.3f} | {m['output_audio_seconds']:.3f} | "
            f"{m['real_time_factor']:.3f} | **{m['audio_seconds_per_second']:.3f}** | "
            f"{m['peak_mlx_gib']:.3f} |",
            "",
            f"{record['hardware']['chip']}, "
            f"{record['hardware']['memory_bytes'] / 1024**3:.0f} GiB; MLX "
            f"{record['software']['mlx']}; native MLX/Metal; {record['measurement_type']}. "
            "Experimental; speech quality requires listening review.",
            "",
        ]
    lines += [
        "Qwen alignment remains WIP because historical timestamp discrepancies are "
        "unresolved; boundary errors need independently annotated reference timings. "
        "LFM speech-to-speech exceeded its output bound on this fixture; its sampled "
        "diagnostic also had unstable waveform amplitude. No speed is published for "
        "that failed task. TTS timings do not establish perceptual quality. "
        "Qwen ASR streaming reprocesses a bounded accumulated utterance; it is not a "
        "causal streaming encoder.",
        "",
        "[Methodology and coverage](docs/benchmarks/audio/README.md) · "
        "[Manual download, Python/CLI/HTTP and benchmark commands](docs/AUDIO_TESTING.md)",
        END,
    ]
    return "\n".join(lines) + "\n"


def update_readme(path, content):
    text = path.read_text()
    if START in text:
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError("README must contain exactly one audio marker pair")
        first, tail = text.split(START)
        _, last = tail.split(END)
        path.write_text(first + content.rstrip() + last)
    else:
        # Initial insertion after the existing VLM performance section preserves its content.
        anchor = "<!-- MLX-ONE-VLM-BENCHMARK-END -->"
        if text.count(anchor) != 1:
            raise ValueError("README insertion anchor is absent or ambiguous")
        path.write_text(text.replace(anchor, anchor + "\n\n" + content.rstrip(), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    catalog = json.loads(Path("benchmarks/audio/checkpoints.json").read_text())
    publication = json.loads(Path("docs/benchmarks/audio/publication.json").read_text())
    records = [
        {**json.loads(Path(path).read_text()), "_evidence_path": path}
        for path in publication["records"]
    ]
    content = render(catalog, records)
    if args.check:
        text = Path("README.md").read_text()
        if (
            START not in text
            or text.split(START)[1].split(END)[0] != content.split(START)[1].split(END)[0]
        ):
            parser.exit(1, "README audio section differs from validated evidence\n")
        print(
            f"Validated {len(records)} audio records; "
            + str(sum(r["status"] == "MEASURED" for r in records))
            + " actual measurements."
        )
    elif args.write:
        update_readme(Path("README.md"), content)
    else:
        print(content)
    coverage = [
        {
            "key": entry["key"],
            "checkpoint": entry["checkpoint"],
            "revision": entry["revision"],
            "task": entry["task"],
            "status": entry["status"],
            "performance": "MEASURED"
            if any(
                r["status"] == "MEASURED"
                and r["checkpoint"] == entry["checkpoint"]
                and r["task"] == entry["task"]
                and r["revision"] == entry["revision"]
                and r["workload"] == entry["workload"]
                for r in records
            )
            else "FAILED"
            if any(
                r.get("attempt_status") == "FAILED"
                and r["checkpoint"] == entry["checkpoint"]
                and r["revision"] == entry["revision"]
                and r["task"] == entry["task"]
                and r["workload"] == entry["workload"]
                for r in records
            )
            else "NOT_RUN",
            "limitation": entry["notes"],
        }
        for entry in catalog["entries"]
    ]
    coverage_json = json.dumps(coverage, indent=2) + "\n"
    coverage_lines = [
        "# Audio performance coverage",
        "",
        f"Measured rows: {sum(row['performance'] == 'MEASURED' for row in coverage)} / "
        f"{len(coverage)}. Source status does not imply checkpoint quality or performance.",
        "",
        "| Key | Checkpoint | Task | Source status | Performance |",
        "| --- | --- | --- | --- | --- |",
    ]
    coverage_lines += [
        f"| {row['key']} | {row['checkpoint']} | {row['task']} | "
        f"{row['status']} | {row['performance']} |"
        for row in coverage
    ]
    coverage_md = "\n".join(coverage_lines) + "\n"
    for path, content in [
        (Path("docs/benchmarks/audio/coverage.json"), coverage_json),
        (Path("docs/benchmarks/audio/coverage.md"), coverage_md),
    ]:
        if args.write:
            path.write_text(content)
        elif args.check and (not path.exists() or path.read_text() != content):
            parser.exit(1, f"stale audio coverage: {path}\n")


if __name__ == "__main__":
    main()
