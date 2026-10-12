"""Diagnose selected/full-sequence BF16 classification on recorded alignment failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_qualification import digest, fixtures
from qwen_audio_qualification import PINS


def run(args) -> dict:
    import mlx.core as mx
    import soundfile as sf

    from mlx_one.audio import load_audio_model
    from mlx_one.models.audio.qwen3_asr.alignment import fix_timestamps, text_units
    from mlx_one.models.audio.qwen3_asr.processing import preprocess

    reference = json.loads(args.reference.read_text())
    comparison = json.loads(args.comparison.read_text())
    if reference["manifest_sha256"] != digest(args.manifest):
        raise ValueError("diagnostic manifest differs from recorded reference")
    selected = {item["id"] for item in comparison["boundary_mismatches"]}
    if not selected:
        raise ValueError("diagnostic requires recorded boundary failures")
    _, records = fixtures(args.manifest, 100)
    expected = {item["id"]: item for item in reference["predictions"]}
    model_name = "Qwen/Qwen3-ForcedAligner-0.6B"
    bundle = load_audio_model(model_name, task="align", revision=PINS[model_name], offline=True)
    reports = []
    for record in records:
        if record["id"] not in selected:
            continue
        waveform, rate = sf.read(args.manifest.parent / record["audio"], dtype="float32")
        if rate != 16000:
            raise ValueError("diagnostic requires 16-kHz audio")
        words = text_units(record["reference"], "English", bundle.path)
        features, count = preprocess(mx.array(waveform), offline_padding=False)
        prompt = "<|audio_start|>" + "<|audio_pad|>" * count + "<|audio_end|>"
        prompt += "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        ids = bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
        positions = mx.array([i for i, token in enumerate(ids) if token == 151705])
        selected_logits = bundle.model(
            mx.array([ids]), features=features, output_positions=positions
        )[0]
        full_logits = mx.take(
            bundle.model(mx.array([ids]), features=features)[0], positions, axis=0
        )
        mx.eval(selected_logits, full_logits)
        row = {
            "id": record["id"],
            "sequence_tokens": len(ids),
            "timestamp_tokens": positions.size,
            "maximum_logit_difference": float(
                mx.max(
                    mx.abs(selected_logits.astype(mx.float32) - full_logits.astype(mx.float32))
                ).item()
            ),
        }
        for name, logits in (("selected", selected_logits), ("full", full_logits)):
            classes = mx.argmax(logits, axis=-1).tolist()
            boundaries = fix_timestamps([value * 80 for value in classes])
            units = [
                {
                    "text": word,
                    "start": round(boundaries[2 * i] / 1000, 3),
                    "end": round(boundaries[2 * i + 1] / 1000, 3),
                }
                for i, word in enumerate(words)
            ]
            if [u["text"] for u in units] != [u["text"] for u in expected[record["id"]]["units"]]:
                raise ValueError("diagnostic text units differ from the pinned reference")
            difference = max(
                abs(a[key] - b[key])
                for a, b in zip(units, expected[record["id"]]["units"], strict=True)
                for key in ("start", "end")
            )
            row[name] = {
                "timestamp_classes": classes,
                "units": units,
                "maximum_boundary_difference_seconds": difference,
                "within_one_tick": difference <= 0.08 + 1e-8,
            }
        reports.append(row)
        print(
            json.dumps(
                {
                    "id": row["id"],
                    "selected_pass": row["selected"]["within_one_tick"],
                    "full_pass": row["full"]["within_one_tick"],
                }
            ),
            flush=True,
        )
    return {
        "scope": "diagnostic selected/full head comparison on previously failing clips only",
        "model": model_name,
        "revision": PINS[model_name],
        "reference_sha256": digest(args.reference),
        "comparison_sha256": digest(args.comparison),
        "manifest_sha256": digest(args.manifest),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reference", "comparison", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    report = run(args)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
