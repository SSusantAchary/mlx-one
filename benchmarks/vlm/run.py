"""Benchmark one already-downloaded, content-verified VLM per process. No downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

from benchmarks.vlm.metrics import EVIDENCE, METRICS, ROOT, file_hash, read, require, validate
from benchmarks.vlm.workloads import WORKLOAD_NAMES, workload


def identify(path, model):
    require(path.is_dir(), "checkpoint must already exist locally; this runner never downloads")
    for name, digest in model["asset_sha256"].items():
        require(
            (path / name).is_file() and file_hash(path / name) == digest,
            "pinned asset mismatch: " + name,
        )
    for name, digest in model["weight_sha256"].items():
        require(
            Path(name).name == name
            and (path / name).is_file()
            and file_hash(path / name) == digest,
            "pinned weight mismatch: " + name,
        )
    require((path / "tokenizer.json").is_file(), "tokenizer.json missing")
    if model.get("tokenizer_sha256"):
        require(
            file_hash(path / "tokenizer.json") == model["tokenizer_sha256"],
            "pinned tokenizer SHA-256 mismatch",
        )
    for name, digest in model["asset_git_blob"].items():
        data = (path / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(actual == digest, "pinned tokenizer Git blob mismatch: " + name)


def measure(bundle, job):
    import mlx.core as mx

    from mlx_one.text.schemas import TextGenerationOptions
    from mlx_one.vision.generation import _stream_vlm
    from mlx_one.vision.messages import prepare_messages

    prompt, images = prepare_messages(
        bundle,
        [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in job["images"]),
                    {"type": "text", "text": job["prompt"]},
                ],
            }
        ],
    )
    events = []
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    chunks = []
    iterator = _stream_vlm(
        bundle,
        prompt,
        images,
        TextGenerationOptions(max_tokens=128),
        lambda: False,
        observer=events.append,
    )
    try:
        chunks = list(iterator)
    finally:
        iterator.close()
        mx.synchronize()
    finished = time.perf_counter()
    prepared = next(e for e in events if e["kind"] == "prepared")
    tokens = [e for e in events if e["kind"] == "token"]
    ids = [e["token_id"] for e in tokens]
    prefill = sum(
        e["duration_ms"] for e in events if e["kind"] == "forward" and e["phase"] == "prefill"
    )
    require(prefill > 0, "missing synchronized prefill observation")
    decode_ms = (tokens[-1]["at"] - tokens[0]["at"]) * 1000 if len(tokens) > 1 else None
    cleanup = next(e for e in events if e["kind"] == "cleanup")
    prompt_tokens = len(prepared["tokens"])
    return {
        "prompt_tokens": prompt_tokens,
        "text_prompt_tokens": prompt_tokens
        - sum(m["visual_tokens"] for m in prepared["image_metadata"]),
        "image_metadata": prepared["image_metadata"],
        "output_tokens": len(ids),
        "token_ids": ids,
        "terminal_token_id": chunks[-1].terminal_token_id,
        "finish_reason": chunks[-1].finish_reason,
        "text": "".join(c.text for c in chunks),
        "preprocessing_ms": prepared["preprocessing_ms"],
        "vision_encoder_ms": None,
        "vision_projection_ms": None,
        "language_prefill_ms": None,
        "visual_prefill_ms": prefill,
        "prefill_tps": prompt_tokens * 1000 / prefill,
        "ttft_ms": (tokens[0]["at"] - started) * 1000 if tokens else None,
        "decode_ms": decode_ms,
        "decode_tps": (len(ids) - 1) * 1000 / decode_ms if decode_ms else None,
        "total_ms": (finished - started) * 1000,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 1024**3,
        "peak_process_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cleanup": cleanup["closed"] is True and cleanup["cache_bytes"] == 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="VLM-S1")
    parser.add_argument("--image-dir", type=Path, help="Prepared pinned HF image directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists; keep raw runs immutable")
    model = next(
        m
        for m in read(EVIDENCE / "benchmark_plan.json")["checkpoints"]
        if m["checkpoint"] == args.model_id
    )
    path = args.checkpoint.expanduser().resolve()
    identify(path, model)
    import statistics

    import mlx.core as mx

    from mlx_one.vision import load_vlm_model

    require(mx.metal.is_available(), "native Metal is required")
    mx.set_default_device(mx.gpu)
    started = time.perf_counter()
    bundle = load_vlm_model(path, offline=True, revision=model["revision"])
    mx.synchronize()
    load_ms = (time.perf_counter() - started) * 1000
    job = workload(args.workload, image_dir=args.image_dir)
    measure(bundle, job)
    samples = [measure(bundle, job) for _ in range(5)]
    implementation = hashlib.sha256()
    for source in sorted((ROOT / "src/mlx_one").rglob("*.py")):
        implementation.update(str(source.relative_to(ROOT)).encode())
        implementation.update(source.read_bytes())
    runner = hashlib.sha256()
    for name in ("run.py", "workloads.py", "metrics.py", "prepare_images.py"):
        source = ROOT / "benchmarks/vlm" / name
        runner.update(source.name.encode())
        runner.update(source.read_bytes())
    record = {
        "task": "vision_language",
        "checkpoint": model["checkpoint"],
        "revision": model["revision"],
        "architecture": model["architecture"],
        "precision": model["precision"],
        "workload": args.workload,
        "protocol": job["protocol"],
        "status": "MEASURED",
        "synthetic": False,
        "measurement_type": "direct-model",
        "backend": "native-mlx-metal",
        "hardware": {
            "device": mx.device_info()["device_name"],
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "mlx_version": importlib.metadata.version("mlx"),
        "metal_version": importlib.metadata.version("mlx-metal"),
        "generation_options": {
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "seed": 0,
            "stop": [],
        },
        "python_version": platform.python_version(),
        "pillow_version": importlib.metadata.version("pillow"),
        "batch_size": 1,
        "warmups": 1,
        "repetitions": 5,
        "statistic": "median",
        "max_tokens": 128,
        "prompt": job["prompt"],
        "source_sizes": job["source_sizes"],
        "image_sha256": job["image_sha256"],
        "expected_text": job["expected_text"],
        "weight_sha256": model["weight_sha256"],
        "asset_sha256": model["asset_sha256"],
        "tokenizer_sha256": file_hash(path / "tokenizer.json"),
        "asset_git_blob": model["asset_git_blob"],
        "implementation_sha256": implementation.hexdigest(),
        "runner_sha256": runner.hexdigest(),
        "model_load_ms": load_ms,
        "timing_scope": "TTFT and total include decode/RGB/resize/normalization/patch work; "
        "prefill includes vision encoder, projection and language prefill; no queue/HTTP",
        "functional_outcome": "inference_completed",
        "semantic_quality_validated": False,
        "samples": samples,
        "metrics": {
            key: statistics.median([s[key] for s in samples])
            if all(s[key] is not None for s in samples)
            else None
            for key in METRICS
        },
    }
    if "image_provenance" in job:
        record["image_provenance"] = job["image_provenance"]
    validate(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
