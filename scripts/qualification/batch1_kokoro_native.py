"""Native Kokoro stages, real English audio, benchmarks and installed interfaces."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from batch1_interfaces import check_interfaces, package_identity

from mlx_one.audio.kokoro import load_kokoro_model, phonemize_english, synthesize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-interfaces", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    bundle = load_kokoro_model(args.model, revision="f3ff3571791e39611d31c381e3a41a3af07b4987")
    mx.eval(bundle.model.parameters(), bundle.voice)
    load_seconds = time.perf_counter() - tick
    metadata = json.loads(args.reference.with_suffix(".json").read_text())
    reference = np.load(args.reference)
    mx.random.seed(0)
    output = bundle.model(
        metadata["phonemes"], mx.array(reference["voice"]), forcing=mx.array(reference["forcing"])
    )
    traces = dict(
        albert=output.albert_hidden, durations=output.durations[0], f0=output.f0, noise=output.noise
    )
    checks = {}
    for name, value in traces.items():
        actual = np.array(value)
        expected = reference[name]
        if name in {"f0", "noise"}:
            expected = expected[:, 0]
        error = np.abs(actual - expected)
        checks[name] = dict(
            status="PASS" if np.allclose(actual, expected, atol=1e-4, rtol=1e-4) else "FAIL",
            max_abs_error=float(error.max()),
            mean_abs_error=float(error.mean()),
        )
    # Common harmonic excitation removes stochastic-source differences. This
    # diagnostic does not assert universal per-sample waveform equality.
    wave, expected = np.array(output.waveform), reference["waveform"]
    vocoder = dict(
        max_abs_error=float(np.abs(wave - expected).max()),
        mean_abs_error=float(np.abs(wave - expected).mean()),
        correlation=float(np.corrcoef(wave, expected)[0, 1]),
        samples=len(wave),
        scope="controlled publisher harmonic excitation",
    )
    text = "Hello, world. This is native speech synthesis on Apple Silicon."
    mx.random.seed(0)
    speech = synthesize(bundle, text)
    audio_path = args.output.parent / "native-english.wav"
    speech.save(audio_path)
    checks["english_audio"] = dict(
        status="PASS"
        if len(speech.waveform) > 2400
        and np.all(np.isfinite(speech.waveform))
        and np.max(np.abs(speech.waveform)) <= 1
        else "FAIL",
        **speech.to_dict(),
        text=text,
        wav_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        peak_amplitude=float(np.max(np.abs(speech.waveform))),
        rms=float(np.sqrt(np.mean(speech.waveform**2))),
    )
    rejected = {}
    for name, kwargs in {
        "voice": dict(text="hello", voice="unknown"),
        "language": dict(text="hello", language="fr"),
        "empty": dict(text=""),
        "phoneme_bound": dict(text="a" * 511, phonemes=True),
        "speed": dict(text="a", phonemes=True, speed=float("nan")),
    }.items():
        try:
            synthesize(bundle, **kwargs)
            rejected[name] = "FAIL"
        except ValueError:
            rejected[name] = "PASS"
    checks["invalid_inputs"] = dict(
        status="PASS" if set(rejected.values()) == {"PASS"} else "FAIL", cases=rejected
    )
    benchmarks = []
    phrase = (
        "The model turns this English text into speech. "
        "All neural layers run locally with the native backend. "
    )
    for characters in (100, 500):
        sample = (phrase * 6)[:characters]
        tick = time.perf_counter()
        phonemes = phonemize_english(sample)
        g2p_seconds = time.perf_counter() - tick
        row = dict(
            characters=characters,
            phonemes=len(phonemes),
            speed=1.0,
            g2p_seconds=g2p_seconds,
            text=sample,
        )
        try:
            synthesize(bundle, sample)
            samples, durations = [], []
            mx.reset_peak_memory()
            for _ in range(5):
                tick = time.perf_counter()
                result = synthesize(bundle, sample)
                mx.synchronize()
                samples.append(time.perf_counter() - tick)
                durations.append(len(result.waveform) / result.sample_rate)
            row.update(
                status="MEASURED",
                repeats=5,
                warmups=1,
                samples_seconds=samples,
                audio_seconds=durations,
                p50_seconds=float(np.median(samples)),
                p95_seconds=float(np.percentile(samples, 95)),
                p50_rtf=float(np.median(np.array(samples) / durations)),
                peak_mlx_bytes=mx.get_peak_memory(),
                scope="loaded Python pipeline including G2P",
            )
        except ValueError as exc:
            row.update(status="REJECTED_BY_INPUT_BOUND", reason=str(exc))
        benchmarks.append(row)
    identity = package_identity()
    result = dict(
        status="PASS" if all(r["status"] == "PASS" for r in checks.values()) else "FAIL",
        checks=checks,
        vocoder_diagnostic=vocoder,
        load_seconds=load_seconds,
        benchmarks=benchmarks,
        package_identity=identity,
        subjective_listening=dict(status="NOT_RUN", reason="requires human audio review"),
        environment=dict(
            python=platform.python_version(),
            macos=platform.mac_ver()[0],
            mlx=importlib.metadata.version("mlx"),
            device=mx.device_info(),
        ),
    )
    if args.check_interfaces:
        result["interfaces"] = check_interfaces(
            bundle, "speech-generation", output_dir=args.output.parent
        )
        result["installed_distribution"] = dict(
            status="PASS"
            if identity["installed_in_environment"] and not identity["forbidden_distributions"]
            else "FAIL"
        )
        if (
            any(r["status"] != "PASS" for r in result["interfaces"].values())
            or result["installed_distribution"]["status"] != "PASS"
        ):
            result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"status", "checks", "interfaces", "benchmarks"}
            }
        )
    )


if __name__ == "__main__":
    main()
