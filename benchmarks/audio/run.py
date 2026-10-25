"""Local-only audio measurements. No automatic checkpoint downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.audio.result_schema import METRICS, PROTOCOL, SCHEMA, digest_file, validate

WORKLOADS = {
    "ASR-S1": 10,
    "ASR-S2": 60,
    "ASR-S3": 600,
    "ASR-S4": 10,
    "ASR-S5": 10,
    "ALIGN-S1": 10,
    "TTS-S1": None,
    "TTS-S2": None,
    "STS-S1": 10,
    "STS-HF-Q1": 2.04,
}


def implementation_identity():
    root = Path("src/mlx_one")
    paths = sorted(
        set(
            [
                *root.glob("audio/*.py"),
                *root.glob("models/audio/**/*.py"),
                *root.glob("models/language/lfm2/*.py"),
                *root.glob("models/shared/*.py"),
                *root.glob("text/*.py"),
                *root.glob("core/*.py"),
                *root.glob("engine/**/*.py"),
                *Path("benchmarks/audio").glob("*.py"),
            ]
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def benchmark(args):
    import mlx.core as mx

    from mlx_one.audio import (
        ASRDecodeOptions,
        align_audio,
        generate_speech,
        load_audio_model,
        transcribe,
    )
    from mlx_one.audio.evaluation import character_error_rate, word_error_rate
    from mlx_one.audio.input import prepare_audio
    from mlx_one.audio.public import AudioEngine

    directory = Path(args.model).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("--model must be a downloaded local checkpoint directory")
    metadata_path = directory / "mlx-one-download.json"
    if not metadata_path.is_file():
        raise ValueError("checkpoint requires mlx-one-download.json from the pinned download tool")
    metadata = json.loads(metadata_path.read_text())
    if args.revision != metadata["revision"] or args.checkpoint != metadata["checkpoint"]:
        raise ValueError("checkpoint identity does not match download manifest")
    assets = metadata["assets"]
    for filename, digest in assets.items():
        path = directory / filename
        if directory not in path.resolve().parents or digest_file(path) != digest:
            raise ValueError("checkpoint asset checksum mismatch: " + filename)
    if args.output.exists():
        raise ValueError("benchmark output exists; refusing overwrite")
    audio_output = getattr(args, "save_first_audio", None)
    if audio_output and audio_output.exists():
        raise ValueError("audio output exists; refusing overwrite")
    duration = None
    if args.audio:
        pcm = prepare_audio(args.audio, max_seconds=600)
        duration = len(pcm) / 16000
        expected = WORKLOADS[args.workload]
        if expected is not None and abs(duration - expected) > 0.05:
            raise ValueError(f"{args.workload} requires {expected} seconds of actual audio")
    else:
        pcm = None
    if (
        args.task in {"transcribe", "stream_transcribe", "align", "speech-to-speech"}
        and pcm is None
    ):
        raise ValueError("this task requires --audio")
    text = args.text_file.read_text() if args.text_file else ""
    if args.task in {"transcribe", "stream_transcribe", "align"} and not text.strip():
        raise ValueError(
            "ASR/alignment benchmarks require a known reference transcript in --text-file"
        )
    if args.workload == "TTS-S1" and len(text.split()) != 30:
        raise ValueError("TTS-S1 requires the fixed 30-word text fixture")
    implementation_sha = implementation_identity()
    if args.workload == "TTS-S1":
        fixed = json.loads(Path("benchmarks/audio/workloads.json").read_text())["TTS-S1"]["text"]
        if " ".join(text.split()) != fixed:
            raise ValueError("TTS-S1 text differs from the pinned workload")
    started = time.perf_counter()
    engine = None
    if args.measurement_type == "python-engine":
        engine = AudioEngine(directory, revision=args.revision, offline=True)
        # Load once through the same actor and keep the model resident.
        _, lease = engine.service._task_lease(
            "audio",
            {
                "transcribe": "transcription",
                "stream_transcribe": "transcription",
                "align": "alignment",
                "synthesize": "speech-generation",
                "speech-to-speech": "speech-to-speech",
            }[args.task],
        )
        lease.release()
        bundle = None
    else:
        bundle = load_audio_model(
            directory,
            task="transcribe" if args.task == "stream_transcribe" else args.task,
            revision=args.revision,
            offline=True,
        )
    mx.synchronize()
    load_seconds = time.perf_counter() - started
    source = args.audio if args.measurement_type == "full-file-pipeline" else pcm

    from mlx_one.audio.loading import audio_model_type

    family = audio_model_type(json.loads((directory / "config.json").read_text()))
    if (
        args.workload.startswith("ASR-")
        and args.task not in {"transcribe", "stream_transcribe"}
        or args.workload.startswith("ALIGN-")
        and args.task != "align"
        or args.workload.startswith("TTS-")
        and args.task != "synthesize"
        or args.workload.startswith("STS-")
        and args.task != "speech-to-speech"
    ):
        raise ValueError("workload does not match the requested task")
    if (args.workload == "ASR-S5") != (args.task == "stream_transcribe"):
        raise ValueError("ASR-S5 is the accumulated utterance streaming protocol")
    if args.task == "stream_transcribe" and args.measurement_type != "direct-model-prepared-pcm":
        raise ValueError("ASR-S5 currently measures direct-model streaming only")
    if args.workload == "ASR-S3" and family != "whisper":
        raise ValueError("ASR-S3 long-form is implemented for Whisper only")

    def execute():
        if args.task == "stream_transcribe":
            from types import SimpleNamespace

            from mlx_one.audio import open_audio_stream

            start = time.perf_counter()
            first, peak, revised, previous = None, 0, 0, ""
            with open_audio_stream(bundle, sample_rate=16000, language=args.language) as session:
                for position in range(0, len(pcm), 32000):
                    for event in session.push(pcm[position : position + 32000]):
                        if first is None:
                            first = (time.perf_counter() - start) * 1000
                        revised += int(bool(previous) and not event.text.startswith(previous))
                        previous = event.text
                    peak = max(peak, (len(session._audio) + len(session._pending)) * 4)
                tail = time.perf_counter()
                final = session.finish()
                end = time.perf_counter()
            return SimpleNamespace(
                text=final.text,
                to_dict=final.to_dict,
                timings={
                    "first_partial_ms": first,
                    "final_result_ms": (end - start) * 1000,
                    "finalization_ms": (end - tail) * 1000,
                    "revised_partials": revised,
                    "buffer_peak_bytes": peak,
                },
            )
        options = ASRDecodeOptions(max_tokens=args.max_tokens) if args.max_tokens else None
        if args.task == "transcribe":
            if family == "whisper" and args.max_tokens:
                from mlx_one.audio import WhisperDecodeOptions

                options = WhisperDecodeOptions(max_tokens=args.max_tokens)
            fn = (
                engine.transcribe if engine else lambda audio, **kw: transcribe(bundle, audio, **kw)
            )
            return fn(source, sample_rate=16000, language=args.language, options=options)
        if args.task == "align":
            fn = (
                engine.align
                if engine
                else lambda audio, text, **kw: align_audio(bundle, audio, text, **kw)
            )
            return fn(source, text, language=args.language or "English", sample_rate=16000)
        kwargs = {"max_tokens": args.max_tokens or 1024}
        if args.task == "speech-to-speech":
            return (
                engine.speech_to_speech(source, sample_rate=16000, **kwargs)
                if engine
                else generate_speech(bundle, audio=source, sample_rate=16000, **kwargs)
            )
        return (
            engine.generate_speech(text, voice=args.voice, **kwargs)
            if engine
            else generate_speech(bundle, text, voice=args.voice, **kwargs)
        )

    runs = []
    try:
        warm = execute()
        del warm
        mx.synchronize()
        for _ in range(PROTOCOL["repetitions"]):
            mx.reset_peak_memory()
            mx.synchronize()
            started = time.perf_counter()
            result = execute()
            mx.synchronize()
            elapsed = time.perf_counter() - started
            generated = getattr(result, "waveform", None)
            output_seconds = len(generated) / result.sample_rate if generated is not None else None
            denominator = output_seconds if output_seconds is not None else duration
            timings = getattr(result, "timings", {})
            row = dict.fromkeys(METRICS)
            row.update(
                processing_seconds=elapsed,
                audio_seconds=duration,
                output_audio_seconds=output_seconds,
                real_time_factor=elapsed / denominator,
                audio_seconds_per_second=denominator / elapsed,
                peak_mlx_gib=mx.get_peak_memory() / 1024**3,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
                time_to_first_audio_ms=timings["time_to_first_audio_seconds"] * 1000
                if timings.get("time_to_first_audio_seconds") is not None
                else None,
            )
            if args.task in {"transcribe", "stream_transcribe"}:
                row.update(
                    wer=word_error_rate(text, result.text),
                    cer=character_error_rate(text, result.text),
                )
            if args.task == "stream_transcribe":
                row.update(
                    {
                        k: result.timings[k]
                        for k in (
                            "first_partial_ms",
                            "final_result_ms",
                            "finalization_ms",
                            "revised_partials",
                            "buffer_peak_bytes",
                        )
                    }
                )
            if args.task == "align":
                row["unmatched_words"] = len(result.unmatched_words)
                if args.boundaries_file:
                    expected = json.loads(args.boundaries_file.read_text())
                    if [u.text for u in result.units] != [u["text"] for u in expected]:
                        raise ValueError("alignment units do not match reference boundary units")
                    errors = sorted(
                        abs(getattr(actual, edge) - reference[edge]) * 1000
                        for actual, reference in zip(result.units, expected, strict=True)
                        for edge in ("start", "end")
                    )
                    if errors:
                        row.update(
                            alignment_mean_error_ms=statistics.mean(errors),
                            alignment_median_error_ms=statistics.median(errors),
                            alignment_p95_error_ms=errors[
                                min(len(errors) - 1, int(0.95 * (len(errors) - 1) + 0.5))
                            ],
                            alignment_max_error_ms=max(errors),
                        )
            row["output"] = result.to_dict()
            if generated is not None:
                row["waveform_sha256"] = hashlib.sha256(generated.tobytes()).hexdigest()
                if audio_output and not runs:
                    audio_output.parent.mkdir(parents=True, exist_ok=True)
                    result.save(audio_output)
                    row["audio_file"] = {
                        "name": audio_output.name,
                        "sha256": digest_file(audio_output),
                    }
            runs.append(row)
            del result, generated
    finally:
        if engine:
            engine.close()
    record = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "measured_at": datetime.now(UTC).isoformat(),
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "name": args.name or args.checkpoint.split("/")[-1],
        "precision": args.precision,
        "task": args.task,
        "workload": args.workload,
        "protocol": PROTOCOL,
        "backend": "native-mlx-metal",
        "measurement_type": args.measurement_type,
        "implementation_sha256": implementation_sha,
        "hardware": {
            "chip": subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
            "machine": platform.machine(),
        },
        "software": {
            "mlx": importlib.metadata.version("mlx"),
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
        },
        "checkpoint_assets": assets,
        "load_seconds": load_seconds,
        "input": {
            "audio_sha256": digest_file(args.audio) if args.audio else None,
            "text_sha256": digest_file(args.text_file) if args.text_file else None,
            "audio_seconds": duration,
            "sample_rate": 16000,
            "language": getattr(args, "fixture_language", None),
            "voice": args.voice if args.task == "synthesize" else None,
        },
        "decoding": {
            "language_hint": args.language,
            "max_tokens_override": args.max_tokens,
            "settings": "native family defaults; see methodology",
        },
        "runs": runs,
        "metrics": {
            key: statistics.median([row[key] for row in runs])
            if all(row[key] is not None for row in runs)
            else None
            for key in METRICS
        },
    }
    if implementation_identity() != implementation_sha:
        raise ValueError("implementation changed while benchmark was running")
    validate(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--name")
    parser.add_argument(
        "--task",
        choices=["transcribe", "stream_transcribe", "align", "synthesize", "speech-to-speech"],
        default="transcribe",
    )
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--boundaries-file", type=Path)
    parser.add_argument("--language")
    parser.add_argument(
        "--fixture-language", help="Language of the fixture, independent of model hints"
    )
    parser.add_argument("--voice", default="us_male")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--save-first-audio", type=Path, help="Save the first measured waveform after timing"
    )
    parser.add_argument(
        "--measurement-type",
        choices=["direct-model-prepared-pcm", "python-engine", "full-file-pipeline"],
        default="direct-model-prepared-pcm",
    )
    parser.add_argument("--output", type=Path, required=True)
    try:
        benchmark(parser.parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"audio benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
