"""Pinned native audio reference qualification; not a production dependency.

Run reference and native backends in separate processes. A partial fixture count
is a diagnostic only; the compare command cannot qualify fewer than 100 clips.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path

MODEL = "distil-whisper/distil-large-v3"
REVISION = "8031d2e6ce6631b7fc45629dddfc00271116d981"
REFERENCE_COMMIT = "cc832f9055ba11c8c55f918ab4bda9472b910d48"
BUDGET = 24 * 1024**3


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fixtures(path: Path, limit: int) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    records = manifest["records"][:limit]
    if not records or len({record["id"] for record in records}) != len(records):
        raise ValueError("fixtures must have distinct nonempty record IDs")
    for record in records:
        source = path.parent / record["audio"]
        if digest(source) != record["audio_sha256"]:
            raise ValueError(f"audio checksum mismatch: {record['id']}")
    return manifest, records


def run(args: argparse.Namespace) -> dict:
    manifest, records = fixtures(args.manifest, args.limit)
    from huggingface_hub import hf_hub_download

    local = Path(args.model)
    config_path = (
        local / "config.json"
        if local.is_dir()
        else Path(
            hf_hub_download(
                args.model, "config.json", revision=args.revision, local_files_only=True
            )
        )
    )
    family = json.loads(config_path.read_text())["model_type"]
    if family not in {"whisper", "moonshine"}:
        raise ValueError("qualification backend is not implemented for this audio family")
    if family != "whisper" and args.timestamp_capture:
        raise ValueError("timestamp capture is only implemented for Whisper")
    start = time.perf_counter()
    if args.backend == "reference":
        import soundfile as sf
        import torch
        from transformers import (
            AutoProcessor,
            MoonshineForConditionalGeneration,
            WhisperForConditionalGeneration,
        )

        torch.set_num_threads(4)
        model_class = (
            WhisperForConditionalGeneration
            if family == "whisper"
            else MoonshineForConditionalGeneration
        )
        model = model_class.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        processor = AutoProcessor.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=True,
        )
        dtype = str(next(model.parameters()).dtype)
    else:
        import mlx.core as mx

        from mlx_one.audio import ASRDecodeOptions, load_audio_model
        from mlx_one.audio.schemas import WhisperDecodeOptions
        from mlx_one.audio.service import transcribe
        from mlx_one.models.audio.whisper.processing import decode_audio

        model = load_audio_model(args.model, revision=args.revision, offline=True)
        mx.eval(model.model.parameters())
        dtype = str(model.model.model.encoder.conv1.weight.dtype)
        options = (
            WhisperDecodeOptions(
                temperatures=(0.0,),
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                compression_ratio_threshold=None,
                logprob_threshold=None,
                no_speech_threshold=None,
                max_tokens=128,
            )
            if family == "whisper"
            else ASRDecodeOptions(max_tokens=193)
        )
    loaded = time.perf_counter()
    print(
        json.dumps({"stage": "loaded", "backend": args.backend, "seconds": loaded - start}),
        flush=True,
    )
    predictions = []
    for record in records:
        source = args.manifest.parent / record["audio"]
        before = time.perf_counter()
        if args.backend == "reference":
            waveform, rate = sf.read(source, dtype="float32")
            if rate != 16000 or waveform.ndim != 1:
                raise ValueError("qualification fixtures must be mono 16-kHz audio")
            processed = processor(waveform, sampling_rate=rate, return_tensors="pt")
            with torch.inference_mode():
                output = (
                    model.generate(
                        processed.input_features,
                        language="en",
                        task="transcribe",
                        num_beams=1,
                        do_sample=False,
                        return_timestamps=True,
                        max_new_tokens=128,
                        condition_on_prev_tokens=False,
                    )
                    if family == "whisper"
                    else model.generate(
                        **processed,
                        num_beams=1,
                        do_sample=False,
                        max_new_tokens=193,
                    )
                )
            text = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
            duration = len(waveform) / rate
            if args.timestamp_capture:
                decoded = processor.tokenizer.decode(
                    output[0],
                    skip_special_tokens=True,
                    output_offsets=True,
                )
                segments = [
                    {
                        "text": item["text"].strip(),
                        "start": item["timestamp"][0],
                        "end": min(duration, item["timestamp"][1]),
                        "raw_end": item["timestamp"][1],
                    }
                    for item in decoded["offsets"]
                ]
        else:
            waveform = decode_audio(source)
            result = transcribe(model, waveform, language="en", options=options, sample_rate=16000)
            text, duration = result.text, result.duration
            if args.timestamp_capture:
                segments = [
                    {"text": item.text, "start": item.start, "end": item.end}
                    for item in result.segments
                ]
        prediction = {
            "id": record["id"],
            "anchor": record["anchor"],
            "reference": record["reference"],
            "prediction": text,
            "duration": duration,
            "seconds": time.perf_counter() - before,
        }
        predictions.append(prediction)
        if args.timestamp_capture:
            prediction["segments"] = segments
        if args.backend == "native":
            del waveform, result
            gc.collect()
            mx.synchronize()
            prediction["active_mlx_bytes"] = mx.get_active_memory()
            prediction["cached_mlx_bytes"] = mx.get_cache_memory()
        print(
            json.dumps(
                {"stage": "clip", "backend": args.backend, **prediction}, ensure_ascii=False
            ),
            flush=True,
        )
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise RuntimeError("24 GiB process-memory budget exceeded")
    result = {
        "backend": args.backend,
        "model": args.model,
        "revision": args.revision,
        "model_type": family,
        "manifest_sha256": digest(args.manifest),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "weight_dtype": dtype,
        "load_seconds": loaded - start,
        "wall_seconds": time.perf_counter() - start,
        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "predictions": predictions,
    }
    if args.backend == "reference":
        result["reference_source_commit"] = REFERENCE_COMMIT
        result["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("transformers", "torch", "numpy", "safetensors")
        }
        if result["versions"]["transformers"] != "5.8.1":
            raise ValueError("reference must use pinned Transformers 5.8.1")
    else:
        result["peak_mlx_bytes"] = mx.get_peak_memory()
        active = [item["active_mlx_bytes"] for item in predictions]
        result["cleanup_growth_bytes"] = max(active[-20:]) - min(active[:20])
        result["versions"] = {name: importlib.metadata.version(name) for name in ("mlx", "numpy")}
    result["audio_seconds"] = sum(item["duration"] for item in predictions)
    result["decode_seconds"] = sum(item["seconds"] for item in predictions)
    result["real_time_factor"] = result["decode_seconds"] / result["audio_seconds"]
    return result


def corpus_error(records: list[dict], field: str, *, characters: bool = False) -> float:
    from mlx_one.audio.evaluation import edit_distance, normalize_transcript

    errors, units = 0, 0
    for record in records:
        reference = normalize_transcript(record["reference"])
        prediction = normalize_transcript(record[field])
        expected = list(reference.replace(" ", "")) if characters else reference.split()
        actual = list(prediction.replace(" ", "")) if characters else prediction.split()
        errors += edit_distance(expected, actual)
        units += len(expected)
    if not units:
        raise ValueError("qualification corpus has no reference units")
    return errors / units


def compare(args: argparse.Namespace) -> dict:
    reference = json.loads(args.reference.read_text())
    native = json.loads(args.native.read_text())
    for name in ("model", "revision", "manifest_sha256"):
        if reference[name] != native[name]:
            raise ValueError(f"qualification inputs differ: {name}")
    expected, actual = reference["predictions"], native["predictions"]
    if [item["id"] for item in expected] != [item["id"] for item in actual]:
        raise ValueError("qualification record IDs or order differ")
    if any(
        a["reference"] != b["reference"] or a["anchor"] != b["anchor"]
        for a, b in zip(expected, actual, strict=True)
    ):
        raise ValueError("qualification reference text or anchor selection differs")
    mismatches = [
        a["id"]
        for a, b in zip(expected, actual, strict=True)
        if a["anchor"] and a["prediction"] != b["prediction"]
    ]
    reference_wer = corpus_error(expected, "prediction")
    native_wer = corpus_error(actual, "prediction")
    reference_cer = corpus_error(expected, "prediction", characters=True)
    native_cer = corpus_error(actual, "prediction", characters=True)
    gates = {
        "100_clips": len(actual) >= 100,
        "16_exact_anchors": sum(item["anchor"] for item in actual) >= 16 and not mismatches,
        "wer_degradation": native_wer <= reference_wer + 0.005,
        "cer_degradation": native_cer <= reference_cer + 0.005,
        "process_budget": native["peak_process_bytes"] <= BUDGET,
        "cleanup_growth": native.get("cleanup_growth_bytes", BUDGET) <= 64 * 1024**2,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "anchor_mismatches": mismatches,
        "reference_wer": reference_wer,
        "native_wer": native_wer,
        "reference_cer": reference_cer,
        "native_cer": native_cer,
        "reference": str(args.reference),
        "native": str(args.native),
        "scope": "English short-form transcript parity only; components/timestamps pending",
    }


def compare_timestamps(args: argparse.Namespace) -> dict:
    reference = json.loads(args.reference.read_text())
    native = json.loads(args.native.read_text())
    for name in ("model", "revision", "manifest_sha256"):
        if reference[name] != native[name]:
            raise ValueError(f"timestamp inputs differ: {name}")
    expected, actual = reference["predictions"], native["predictions"]
    if [item["id"] for item in expected] != [item["id"] for item in actual]:
        raise ValueError("timestamp record IDs or order differ")
    mismatches, maximum = [], 0.0
    for left, right in zip(expected, actual, strict=True):
        if len(left["segments"]) != len(right["segments"]):
            mismatches.append(left["id"])
            continue
        for a, b in zip(left["segments"], right["segments"], strict=True):
            difference = max(abs(a["start"] - b["start"]), abs(a["end"] - b["end"]))
            maximum = max(maximum, difference)
            if a["text"] != b["text"] or difference > 0.02 + 1e-8:
                mismatches.append(left["id"])
    gates = {
        "16_clips": len(actual) >= 16,
        "matched_segments": not mismatches,
        "one_timestamp_tick": maximum <= 0.02 + 1e-8,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mismatches": sorted(set(mismatches)),
        "maximum_boundary_error_seconds": maximum,
        "reference": str(args.reference),
        "native": str(args.native),
        "scope": "English short-form segment boundaries; both ends clamped to audio duration",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("reference", "native", "compare", "compare-timestamps"), required=True
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timestamp-capture", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    comparing = args.backend.startswith("compare")
    if comparing and (args.reference is None or args.native is None):
        parser.error("compare requires reference and native result files")
    if not comparing and (args.manifest is None or args.limit < 1):
        parser.error("inference requires a manifest and positive limit")
    if args.backend == "compare-timestamps":
        result = compare_timestamps(args)
    else:
        result = compare(args) if comparing else run(args)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False)
    print(
        json.dumps({"output": str(args.output), "status": result.get("status", "RECORDED")}),
        flush=True,
    )
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
