"""Pinned Parakeet offline corpus qualification; reference is validation-only.

Full transcript gates remain unchanged. Token timestamp diagnostics are recorded
separately and do not establish word-timestamp or original NeMo-reference parity.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import resource
import subprocess
import time
from pathlib import Path

from audio_qualification import BUDGET, corpus_error, digest, fixtures

SOURCE = "469230357aab0f2b303b0d638c1f8d06edb14184"
PINS = {
    "nvidia/parakeet-tdt-0.6b-v2": "ae9ad07059c7c739ffaf932226a8fe64ae2620b0",
    "nvidia/parakeet-tdt-0.6b-v3": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
}


def verify_contract(root, path, model):
    contract = json.loads(path.read_text())
    if (
        contract["model"] != model
        or contract["revision"] != PINS[model]
        or contract["reference_commit"] != SOURCE
        or contract["weight_layout"] != "canonical"
    ):
        raise ValueError("checkpoint contract differs from frozen Parakeet target")
    for name, entry in contract["assets"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError("unsafe checkpoint contract asset path")
        if digest(root / name) != entry["sha256"]:
            raise ValueError(f"checkpoint asset checksum differs: {name}")
    return contract


def run(args):
    manifest, records = fixtures(args.manifest, args.limit)
    if args.model.endswith("-v2") and args.language != "English":
        raise ValueError("original v2 qualification is English-only")
    if any(r.get("language", args.language) != args.language for r in records):
        raise ValueError("fixture language differs from requested qualification subset")
    verify_contract(args.directory, args.contract, args.model)
    import soundfile as sf

    started = time.perf_counter()
    if args.backend == "reference":
        import torch
        import transformers
        from transformers import AutoProcessor, ParakeetForTDT

        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(transformers.__file__).resolve().parents[2],
            text=True,
        ).strip()
        if commit != SOURCE or importlib.metadata.version("torch") != "2.7.1":
            raise ValueError("reference source/environment differs from frozen target")
        torch.set_num_threads(4)
        model = ParakeetForTDT.from_pretrained(
            str(args.directory),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        ).eval()
        processor = AutoProcessor.from_pretrained(str(args.directory), local_files_only=True)
    else:
        import mlx.core as mx

        from mlx_one.audio import load_audio_model, transcribe

        model = load_audio_model(args.directory, offline=True, revision=PINS[args.model])
    loaded = time.perf_counter()
    print(
        json.dumps({"stage": "loaded", "backend": args.backend, "seconds": loaded - started}),
        flush=True,
    )
    predictions = []
    for record in records:
        waveform, rate = sf.read(args.manifest.parent / record["audio"], dtype="float32")
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError("Parakeet fixtures require mono 16-kHz audio")
        before = time.perf_counter()
        if args.backend == "reference":
            inputs = processor(waveform, sampling_rate=rate, return_tensors="pt")
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, num_beams=1)
            text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
            del inputs, output
        else:
            result = transcribe(model, waveform, sample_rate=rate)
            text = result.text
            del result
        item = {
            "id": record["id"],
            "anchor": record["anchor"],
            "reference": record["reference"],
            "prediction": text,
            "duration": len(waveform) / rate,
            "seconds": time.perf_counter() - before,
        }
        del waveform
        gc.collect()
        if args.backend == "native":
            mx.synchronize()
            item["active_mlx_bytes"] = mx.get_active_memory()
            item["cached_mlx_bytes"] = mx.get_cache_memory()
        predictions.append(item)
        print(json.dumps({"stage": "clip", **item}, ensure_ascii=False), flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > BUDGET:
            raise RuntimeError("24 GiB process budget exceeded")
    result = {
        "model": args.model,
        "revision": PINS[args.model],
        "backend": args.backend,
        "reference_source_commit": SOURCE,
        "dtype": "float32",
        "mode": "offline",
        "language_subset": [args.language],
        "contract_sha256": digest(args.contract),
        "manifest_sha256": digest(args.manifest),
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["revision"],
        "predictions": predictions,
        "load_seconds": loaded - started,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    result["versions"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if args.backend == "native":
        result["peak_mlx_bytes"] = mx.get_peak_memory()
        active = [p["active_mlx_bytes"] for p in predictions]
        result["cleanup_growth_bytes"] = max(active[-20:]) - min(active[:20])
        root = Path(__file__).resolve().parents[1]
        result["native_code_sha256"] = {
            name: digest(root / "src/mlx_one" / name)
            for name in (
                "models/audio/parakeet/model.py",
                "models/audio/parakeet/processing.py",
                "audio/parakeet.py",
            )
        }
    seconds = sum(p["seconds"] for p in predictions)
    audio_seconds = sum(p["duration"] for p in predictions)
    result.update(
        {
            "audio_seconds": audio_seconds,
            "decode_seconds": seconds,
            "real_time_factor": seconds / audio_seconds,
        }
    )
    return result


def compare(args):
    ref, native = (json.loads(p.read_text()) for p in (args.reference, args.native))
    if ref["backend"] != "reference" or native["backend"] != "native":
        raise ValueError("comparison requires separate reference/native captures")
    for name in (
        "model",
        "revision",
        "reference_source_commit",
        "dtype",
        "mode",
        "language_subset",
        "contract_sha256",
        "manifest_sha256",
    ):
        if ref[name] != native[name]:
            raise ValueError(f"qualification inputs differ: {name}")
    left, right = ref["predictions"], native["predictions"]
    if [(p["id"], p["reference"], p["anchor"]) for p in left] != [
        (p["id"], p["reference"], p["anchor"]) for p in right
    ]:
        raise ValueError("fixture selection/order differs")
    english = native["language_subset"] == ["English"]
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
    gates = {
        "required_corpus_size": len(right) >= (100 if english else 25),
        "exact_anchors": not mismatches and (not english or sum(p["anchor"] for p in right) >= 16),
        "wer_degradation": nw <= rw + 0.005,
        "cer_degradation": nc <= rc + 0.005,
        "process_budget": native["peak_process_bytes"] <= BUDGET,
        "cleanup": native.get("cleanup_growth_bytes", BUDGET) <= 64 * 1024**2,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "mismatches": mismatches,
        "reference_wer": rw,
        "native_wer": nw,
        "reference_cer": rc,
        "native_cer": nc,
        "reference_sha256": digest(args.reference),
        "native_sha256": digest(args.native),
        "scope": (
            "one pinned float32 offline transcript/language corpus; components, "
            "timestamps and original NeMo parity remain separate"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    running = sub.add_parser("run")
    running.add_argument("--model", choices=PINS, required=True)
    running.add_argument("--backend", choices=("reference", "native"), required=True)
    running.add_argument("--directory", type=Path, required=True)
    running.add_argument("--contract", type=Path, required=True)
    running.add_argument("--manifest", type=Path, required=True)
    running.add_argument("--limit", type=int, default=100)
    running.add_argument("--language", choices=("English", "French", "German"), default="English")
    comparing = sub.add_parser("compare")
    comparing.add_argument("--reference", type=Path, required=True)
    comparing.add_argument("--native", type=Path, required=True)
    for cmd in (running, comparing):
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    result = run(args) if args.command == "run" else compare(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(args.output), "status": result.get("status", "CAPTURED")}))
    raise SystemExit(1 if result.get("status") == "FAIL" else 0)


if __name__ == "__main__":
    main()
