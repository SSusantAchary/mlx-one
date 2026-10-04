"""Native Qwen session/resource checks, not a substitute for reference quality gates."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

from audio_qualification import digest, fixtures
from qwen_audio_qualification import BUDGET, PINS, SOURCE


def run(model_name: str, manifest_path: Path) -> dict:
    import mlx.core as mx
    import soundfile as sf

    from mlx_one.audio import load_audio_model, open_audio_stream
    from mlx_one.audio.moonshine import AudioResourceLimitError

    _, records = fixtures(manifest_path, 2)
    if len(records) != 2:
        raise ValueError("stream stress requires two checksummed utterances")
    waves = []
    for record in records:
        wave, rate = sf.read(manifest_path.parent / record["audio"], dtype="float32")
        if wave.ndim != 1 or rate != 16000 or not 32000 <= len(wave) <= 480000:
            raise ValueError(
                "stream stress fixtures must be mono 16-kHz utterances of 2–30 seconds"
            )
        waves.append(wave)
    loaded_at = time.perf_counter()
    bundle = load_audio_model(model_name, revision=PINS[model_name], offline=True)
    mx.eval(bundle.model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - loaded_at
    baseline = mx.get_active_memory()
    active, runs = [], []

    def measure_cleanup() -> None:
        gc.collect()
        mx.synchronize()
        active.append(mx.get_active_memory())
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise AudioResourceLimitError("stream stress exceeded 24 GiB process budget")

    def events_for(session, waveform, partition):
        started = time.perf_counter()
        events = []
        first_partial = None
        for start in range(0, len(waveform), partition):
            batch = session.push(waveform[start : start + partition])
            if batch and first_partial is None:
                first_partial = time.perf_counter() - started
            events.extend(batch)
        events.append(session.finish())
        signature = [
            {
                key: getattr(event, key)
                for key in ("kind", "text", "sequence", "audio_seconds", "language")
            }
            for event in events
        ]
        return signature, first_partial, time.perf_counter() - started

    # Reuse one session across reset boundaries, then close it without unloading
    # the shared bundle. Different fragmentation must preserve the same hypotheses.
    for record, wave in zip(records, waves, strict=True):
        with open_audio_stream(bundle, sample_rate=16000, language="English") as session:
            for partition in (17, 16000, 32000):
                signature, first_partial, seconds = events_for(session, wave, partition)
                runs.append(
                    {
                        "fixture": record["id"],
                        "partition_samples": partition,
                        "events": signature,
                        "first_partial_seconds": first_partial,
                        "seconds": seconds,
                        "audio_seconds": len(wave) / 16000,
                    }
                )
                try:
                    session.push([0.0])
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("input was accepted after finish without reset")
                session.reset()
                measure_cleanup()
        if (
            session._bundle is not None
            or session._options is not None
            or session._cancelled is not None
        ):
            raise AssertionError("closed session retained model/options/callback references")
        measure_cleanup()

    # Closing/cancelling one independent session cannot change another's model
    # or input; no global allocator clearing is permitted by the implementation.
    with open_audio_stream(bundle, sample_rate=16000, language="English") as survivor:
        with open_audio_stream(bundle, sample_rate=16000, language="English") as cancelled:
            survivor.push(waves[0][:16000])
            cancelled.push(waves[1][:16000])
            cancelled.cancel()
        if survivor._bundle is not bundle or survivor._received != 16000:
            raise AssertionError("cancelling one session modified another session")
        for start in range(16000, len(waves[0]), 16000):
            survivor.push(waves[0][start : start + 16000])
        survivor.finish()
    measure_cleanup()

    # Trigger cancellation after native preflight/prefill checks, not just before
    # accepting PCM, and verify finally resets all request-owned KV state.
    cancelled_runs = 0
    for _ in range(3):
        checks = 0

        def cancel_during_decode():
            nonlocal checks
            checks += 1
            return checks >= 5

        with open_audio_stream(
            bundle, sample_rate=16000, language="English", cancelled=cancel_during_decode
        ) as session:
            try:
                session.push(waves[0][:32000])
            except RuntimeError as exc:
                if "cancelled" not in str(exc):
                    raise
                cancelled_runs += 1
            else:
                raise AssertionError("native cancellation callback was not observed")
        measure_cleanup()

    with open_audio_stream(bundle, sample_rate=16000) as session:
        try:
            session.push([0.0] * 480001)
        except AudioResourceLimitError:
            pass
        else:
            raise AssertionError("30-second resource limit did not reject excess PCM")
        if session._received or session._pending or session._audio:
            raise AssertionError("over-limit input mutated session state")
        session.reset()
        empty = session.finish()
        if empty.text or empty.language is not None or empty.audio_seconds != 0:
            raise AssertionError("empty stream invented transcript metadata")
    measure_cleanup()

    partitions_match = all(
        runs[start]["events"] == runs[start + 1]["events"] == runs[start + 2]["events"]
        for start in (0, 3)
    )
    gates = {
        "complete_hypotheses_partition_invariant": partitions_match,
        "repeated_reset_finish": True,
        "session_isolation": True,
        "native_mid_decode_cancellation": cancelled_runs == 3,
        "explicit_30_second_limit": True,
        "cleanup_growth_le_64_mib": max(active) - baseline <= 64 * 1024**2,
        "process_budget": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= BUDGET,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "two English utterances: session lifecycle/resources/fragmentation only; "
        "not ASR corpus quality or reference streaming parity",
        "model": model_name,
        "revision": PINS[model_name],
        "reference_source_commit": SOURCE,
        "dtype": str(bundle.model.audio_tower.conv2d1.weight.dtype),
        "manifest_sha256": digest(manifest_path),
        "gates": gates,
        "load_seconds": load_seconds,
        "baseline_active_bytes": baseline,
        "cleanup_active_bytes": active,
        "peak_mlx_active_bytes": mx.get_peak_memory(),
        "final_allocator_cached_bytes": mx.get_cache_memory(),
        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=[name for name in PINS if "ForcedAligner" not in name], required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    report = run(args.model, args.manifest)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    raise SystemExit(1 if report["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
