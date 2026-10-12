"""Separate-process Qwen audio checkpoint qualification, never a runtime fallback.

Reference: pinned Qwen source, Transformers 4.57.6, PyTorch 2.7.1, BF16 SDPA.
Persist reference outputs before starting native Metal runs. Limited runs are diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import resource
import subprocess
import time
from pathlib import Path

from audio_qualification import corpus_error, digest, fixtures

SOURCE = "7c6daf77a2421100f5fb066495372c00129d39ff"
PINS = {
    "Qwen/Qwen3-ASR-0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "Qwen/Qwen3-ASR-1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
    "Qwen/Qwen3-ForcedAligner-0.6B": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
}
BUDGET = 24 * 1024**3


def run(args: argparse.Namespace) -> dict:
    manifest, records = fixtures(args.manifest, args.limit)
    if any(record.get("language", args.language) != args.language for record in records):
        raise ValueError("fixture language differs from the declared qualification subset")
    import soundfile as sf
    from huggingface_hub import snapshot_download

    root = snapshot_download(args.model, revision=PINS[args.model], local_files_only=True)
    started = time.perf_counter()
    alignment = args.model == "Qwen/Qwen3-ForcedAligner-0.6B"
    if alignment and args.mode != "offline":
        raise ValueError("alignment is not a streaming transcription task")
    if args.backend == "reference":
        import qwen_asr
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

        source_root = Path(qwen_asr.__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
        if commit != SOURCE:
            raise ValueError("reference source checkout does not match the frozen commit")

        if (
            importlib.metadata.version("transformers") != "4.57.6"
            or importlib.metadata.version("torch") != "2.7.1"
        ):
            raise ValueError("reference environment does not match pinned versions")
        torch.set_num_threads(4)
        if alignment:
            model = Qwen3ForcedAligner.from_pretrained(
                root, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa"
            )
        else:
            model = Qwen3ASRModel.from_pretrained(
                root,
                dtype=torch.bfloat16,
                device_map="cpu",
                attn_implementation="sdpa",
                max_new_tokens=512,
                max_inference_batch_size=1,
            )
    else:
        import mlx.core as mx

        from mlx_one.audio import align_audio, load_audio_model, open_audio_stream, transcribe

        model = load_audio_model(
            args.model,
            task="align" if alignment else "transcribe",
            revision=PINS[args.model],
            offline=True,
        )
        if getattr(args, "numerical_candidate", False):
            from qwen_audio_rounding_candidate import install_candidate

            install_candidate(model.model)
        if getattr(args, "text_math_candidate", False):
            from qwen_audio_text_candidate import install_candidate

            install_candidate(model.model)
        if getattr(args, "layernorm_candidate", False):
            from qwen_audio_text_candidate import install_layernorm

            install_layernorm(model.model)
        mx.eval(model.model.parameters())
    loaded = time.perf_counter()
    print(
        json.dumps({"stage": "loaded", "backend": args.backend, "seconds": loaded - started}),
        flush=True,
    )
    predictions = []
    for record in records:
        waveform, rate = sf.read(args.manifest.parent / record["audio"], dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError("qualification requires checksummed mono 16-kHz fixtures")
        language = record.get("language", args.language)
        before = time.perf_counter()
        result, events = None, None
        if args.mode == "stream":
            if args.backend == "reference":
                from qwen_audio_stream_reference import reference_stream

                events = reference_stream(model, waveform, language, args.chunk_samples)
            else:
                events = []
                with open_audio_stream(model, sample_rate=rate, language=language) as session:
                    for start in range(0, len(waveform), args.chunk_samples):
                        for event in session.push(waveform[start : start + args.chunk_samples]):
                            events.append(
                                {
                                    "kind": event.kind,
                                    "text": event.text,
                                    "language": event.language,
                                    "audio_seconds": event.audio_seconds,
                                    "sequence": event.sequence,
                                    "latency_seconds": time.perf_counter() - before,
                                }
                            )
                    event = session.finish()
                    events.append(
                        {
                            "kind": event.kind,
                            "text": event.text,
                            "language": event.language,
                            "audio_seconds": event.audio_seconds,
                            "sequence": event.sequence,
                            "latency_seconds": time.perf_counter() - before,
                        }
                    )
            text, units = events[-1]["text"], None
        elif args.backend == "reference":
            with torch.inference_mode():
                result = (
                    model.align((waveform, rate), text=record["reference"], language=language)[0]
                    if alignment
                    else model.transcribe((waveform, rate), language=language)[0]
                )
            text = record["reference"] if alignment else result.text
            units = (
                [
                    {"text": item.text, "start": item.start_time, "end": item.end_time}
                    for item in result.items
                ]
                if alignment
                else None
            )
        else:
            result = (
                align_audio(
                    model, waveform, record["reference"], language=language, sample_rate=rate
                )
                if alignment
                else transcribe(model, waveform, language=language, sample_rate=rate)
            )
            text = result.text
            units = (
                [{"text": item.text, "start": item.start, "end": item.end} for item in result.units]
                if alignment
                else None
            )
        item = {
            "id": record["id"],
            "anchor": record["anchor"],
            "reference": record["reference"],
            "prediction": text,
            "duration": len(waveform) / rate,
            "language": language,
            "seconds": time.perf_counter() - before,
        }
        if alignment:
            item["units"] = units
        if events is not None:
            item["events"] = events
            item["first_partial_seconds"] = next(
                (e["latency_seconds"] for e in events if e["kind"] == "partial"), None
            )
        del result
        gc.collect()
        if args.backend == "native":
            mx.synchronize()
            item["active_mlx_bytes"] = mx.get_active_memory()
            item["cached_mlx_bytes"] = mx.get_cache_memory()
        predictions.append(item)
        print(json.dumps({"stage": "clip", **item}, ensure_ascii=False), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise RuntimeError("24 GiB process-memory budget exceeded")
    result = {
        "backend": args.backend,
        "model": args.model,
        "revision": PINS[args.model],
        "task": "align" if alignment else "transcribe",
        "mode": args.mode,
        "chunk_samples": args.chunk_samples if args.mode == "stream" else None,
        "stream_reference": "upstream rollback wrapper + pinned SDPA decoder"
        if args.mode == "stream"
        else None,
        "dtype": "bfloat16",
        "reference_source_commit": SOURCE,
        "attention": "sdpa-global",
        "language_subset": [args.language],
        "manifest_sha256": digest(args.manifest),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "predictions": predictions,
        "load_seconds": loaded - started,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if args.backend == "native":
        result["numerical_candidate"] = (
            "audio-fp32-layernorm"
            if getattr(args, "layernorm_candidate", False)
            else "cpu-text-math"
            if getattr(args, "text_math_candidate", False)
            else "cpu-bf16-rounding"
            if getattr(args, "numerical_candidate", False)
            else None
        )
        source_root = Path(__file__).resolve().parents[1]
        result["native_code_sha256"] = {
            "model": digest(source_root / "src/mlx_one/models/audio/qwen3_asr/model.py"),
            "preprocessing": digest(
                source_root / "src/mlx_one/models/audio/qwen3_asr/processing.py"
            ),
        }
        if result["numerical_candidate"]:
            result["native_code_sha256"]["candidate"] = digest(
                source_root
                / (
                    "benchmarks/qwen_audio_text_candidate.py"
                    if (
                        getattr(args, "text_math_candidate", False)
                        or getattr(args, "layernorm_candidate", False)
                    )
                    else "benchmarks/qwen_audio_rounding_candidate.py"
                )
            )
            if getattr(args, "layernorm_candidate", False):
                result["native_code_sha256"]["audio_candidate"] = digest(
                    source_root / "benchmarks/qwen_audio_rounding_candidate.py"
                )
        active = [p["active_mlx_bytes"] for p in predictions]
        result["cleanup_growth_bytes"] = max(active[-20:]) - min(active[:20])
        result["peak_mlx_bytes"] = mx.get_peak_memory()
    result["real_time_factor"] = sum(p["seconds"] for p in predictions) / sum(
        p["duration"] for p in predictions
    )
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in (
            ("torch", "transformers", "numpy", "safetensors")
            if args.backend == "reference"
            else ("mlx", "soundfile", "safetensors")
        )
    }
    return result


def compare(args: argparse.Namespace) -> dict:
    ref, native = [json.loads(path.read_text()) for path in (args.reference, args.native)]
    for key in (
        "model",
        "revision",
        "task",
        "mode",
        "dtype",
        "reference_source_commit",
        "attention",
        "manifest_sha256",
        "language_subset",
    ):
        if ref[key] != native[key]:
            raise ValueError(f"qualification inputs differ: {key}")
    for key in ("chunk_samples", "stream_reference"):
        if ref.get(key) != native.get(key):
            raise ValueError(f"qualification inputs differ: {key}")
    left, right = ref["predictions"], native["predictions"]
    if [(p["id"], p["reference"], p["anchor"]) for p in left] != [
        (p["id"], p["reference"], p["anchor"]) for p in right
    ]:
        raise ValueError("fixture selection/order differs")
    english = native["language_subset"] == ["English"]
    gates = {
        "required_corpus_size": len(right) >= (100 if english else 25),
        "process_budget": native["peak_process_bytes"] <= BUDGET,
        "cleanup": native.get("cleanup_growth_bytes", BUDGET) <= 64 * 1024**2,
    }
    if native["mode"] == "stream":
        keys = ("kind", "text", "language", "audio_seconds", "sequence")
        gates["complete_hypothesis_events"] = all(
            [{k: e[k] for k in keys} for e in a["events"]]
            == [{k: e[k] for k in keys} for e in b["events"]]
            for a, b in zip(left, right, strict=True)
        )
    mismatches, boundary_mismatches, maximum = [], [], 0.0
    if native["task"] == "align":
        for a, b in zip(left, right, strict=True):
            if [u["text"] for u in a["units"]] != [u["text"] for u in b["units"]]:
                mismatches.append(a["id"])
                continue
            previous = 0.0
            for x, y in zip(a["units"], b["units"], strict=True):
                if not all(math.isfinite(u[k]) for u in (x, y) for k in ("start", "end")):
                    mismatches.append(a["id"])
                    continue
                difference = max(abs(x["start"] - y["start"]), abs(x["end"] - y["end"]))
                maximum = max(maximum, difference)
                if difference > 0.08 + 1e-8:
                    boundary_mismatches.append(
                        {
                            "id": a["id"],
                            "text": x["text"],
                            "reference": x,
                            "native": y,
                            "difference_seconds": difference,
                        }
                    )
                if y["start"] < previous or y["end"] < y["start"]:
                    mismatches.append(a["id"])
                previous = y["end"]
        gates.update(
            {
                "matching_units_and_order": not mismatches,
                "timestamp_tick_80ms": maximum <= 0.08 + 1e-8,
            }
        )
        metrics = {
            "maximum_boundary_difference_seconds": maximum,
            "boundary_mismatches": boundary_mismatches,
        }
    else:
        mismatches = [
            a["id"]
            for a, b in zip(left, right, strict=True)
            if a["anchor"] and a["prediction"] != b["prediction"]
        ]
        rw, nw = corpus_error(left, "prediction"), corpus_error(right, "prediction")
        rc, nc = (
            corpus_error(left, "prediction", characters=True),
            corpus_error(right, "prediction", characters=True),
        )
        gates.update(
            {
                "exact_anchors": not mismatches
                and (not english or sum(p["anchor"] for p in right) >= 16),
                "wer_degradation": nw <= rw + 0.005,
                "cer_degradation": nc <= rc + 0.005,
            }
        )
        metrics = {"reference_wer": rw, "native_wer": nw, "reference_cer": rc, "native_cer": nc}
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "numerical_candidate": native.get("numerical_candidate"),
        "gates": gates,
        "mismatches": mismatches,
        **metrics,
        "scope": "one checkpoint/task/language/mode only; all other gates remain separate",
        "reference_sha256": digest(args.reference),
        "native_sha256": digest(args.native),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    running = sub.add_parser("run")
    running.add_argument("--backend", choices=["reference", "native"], required=True)
    running.add_argument("--model", choices=PINS, required=True)
    running.add_argument("--manifest", type=Path, required=True)
    running.add_argument("--language", default="English")
    running.add_argument("--limit", type=int, default=100)
    running.add_argument("--mode", choices=["offline", "stream"], default="offline")
    running.add_argument("--chunk-samples", type=int, default=16000)
    running.add_argument(
        "--numerical-candidate",
        action="store_true",
        help="Validation-only failed candidate; production behavior is unchanged.",
    )
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--text-math-candidate", action="store_true")
    running.add_argument("--layernorm-candidate", action="store_true")
    comparing = sub.add_parser("compare")
    comparing.add_argument("--reference", type=Path, required=True)
    comparing.add_argument("--native", type=Path, required=True)
    comparing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        candidates = (args.numerical_candidate, args.text_math_candidate, args.layernorm_candidate)
        if args.backend == "reference" and any(candidates):
            parser.error(
                "numerical candidate applies only to native validation, never the reference"
            )
        if sum(candidates) > 1:
            parser.error("select one isolated numerical candidate at a time")
    if args.output.exists():
        parser.error("output must not exist")
    if args.command == "run" and (args.limit < 1 or not 1 <= args.chunk_samples <= 32000):
        parser.error("limit must be positive and chunk_samples must be in [1, 32000]")
    result = run(args) if args.command == "run" else compare(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(
        json.dumps({"output": str(args.output), "status": result.get("status", "RECORDED")}),
        flush=True,
    )
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
