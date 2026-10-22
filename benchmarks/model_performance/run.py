"""Benchmark an already downloaded, pinned Hub snapshot; never download weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from report import EVIDENCE, ROOT, read, require, sha, validate_samples


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify(path, model):
    # A renamed arbitrary folder is not evidence of checkpoint origin.
    require(path.is_dir(), "checkpoint must already exist locally")
    require(
        path.parent.name == "snapshots"
        and path.parent.parent.name == "models--" + model["model_id"].replace("/", "--")
        and path.name == model["revision"],
        "use the exact pinned Hugging Face cache snapshot (see manual commands)",
    )
    require(sha(path / "config.json") == model["config_sha256"], "pinned config hash mismatch")
    index = path / "model.safetensors.index.json"
    names = (
        sorted(set(read(index)["weight_map"].values())) if index.exists() else ["model.safetensors"]
    )
    require(
        all(Path(n).name == n and (path / n).is_file() for n in names),
        "missing or unsafe safetensors shard",
    )
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--profile", choices=["stored", "affine_q4", "affine_q8"], default="stored")
    parser.add_argument("--token-ids", type=Path, help="exactly 128 token IDs; no chat templating")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output exists; choose a new run filename")
    require(
        args.repetitions >= 3 and args.warmups >= 1 and 0 < args.memory_fraction < 1,
        "need >=3 repetitions, >=1 warmup and memory fraction between zero and one",
    )
    model_info = next(
        m
        for m in read(EVIDENCE / "inventory.json")["checkpoints"]
        if m["model_id"] == args.model_id
    )
    require(args.profile in model_info["available_profiles"], "profile not inventoried")
    checkpoint = args.checkpoint.expanduser().absolute()
    names = identify(checkpoint, model_info)

    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.text.loading import load_text_model

    require(mx.metal.is_available(), "benchmark requires Apple Metal GPU")
    mx.set_default_device(mx.gpu)
    device = mx.device_info()
    budget = int(device["max_recommended_working_set_size"] * args.memory_fraction)
    require(
        sum((checkpoint / name).stat().st_size for name in names) < budget,
        "checkpoint files exceed conservative MLX memory budget",
    )
    mx.set_memory_limit(budget)
    mx.set_cache_limit(256 * 1024**2)
    hashes = {name: file_hash(checkpoint / name) for name in names}
    # Historical checkpoints additionally have independently archived weight hashes.
    cohort = read(EVIDENCE / "historical/cohort.json")
    known = next(
        (
            v["source_files"]
            for k, v in cohort["records"].items()
            if k.startswith(model_info["id"] + "-")
        ),
        None,
    )
    if known:
        require(
            all(known.get(name) == digest for name, digest in hashes.items()),
            "weight contents differ from pinned historical assets",
        )
    bundle = load_text_model(checkpoint, revision=model_info["revision"], offline=True)
    if args.profile.startswith("affine"):
        require(not bundle.quantization, "do not requantize already quantized weights")
        bits = int(args.profile[-1])
        nn.quantize(bundle.model, group_size=64, bits=bits, mode="affine")
        bundle = replace(bundle, quantization={"bits": bits, "group_size": 64, "mode": "affine"})
        configure_execution_dtype(bundle, "float16")
    model = bundle.model
    model.eval()
    mx.eval(model.parameters())
    mx.synchronize()
    if args.token_ids:
        ids = read(args.token_ids)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
    else:
        seed = bundle.tokenizer.encode("The red book was on the table. The city is calm. ")
        require(bool(seed), "empty prompt seed")
        ids = (seed * ((128 + len(seed) - 1) // len(seed)))[:128]
    vocab_size = getattr(model.config, "vocab_size", None)
    require(
        isinstance(ids, list)
        and len(ids) == 128
        and all(type(t) is int and t >= 0 and (vocab_size is None or t < vocab_size) for t in ids),
        "prompt must contain exactly 128 valid token IDs",
    )
    require(bundle.context_length >= 256, "checkpoint context too small for protocol")

    def request():
        cache = configure_kv_caches(model.make_cache(), "f16", "f16")
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        start = time.perf_counter()
        logits = model(mx.array([ids]), cache=cache, logits_mode="last_only").logits
        token = mx.argmax(logits[0, -1])
        mx.eval(token)
        generated = [int(token.item())]
        del logits, token
        mx.synchronize()
        prefill = time.perf_counter() - start
        start_decode = time.perf_counter()
        while len(generated) < 128 and generated[-1] not in bundle.eos_token_ids:
            logits = model(mx.array([[generated[-1]]]), cache=cache, logits_mode="last_only").logits
            token = mx.argmax(logits[0, -1])
            mx.eval(token)
            generated.append(int(token.item()))
            del logits, token
        mx.synchronize()
        decode = time.perf_counter() - start_decode
        peak = int(mx.get_peak_memory())
        for layer in cache:
            layer.reset()
        clean = all(layer.offset == 0 and layer.nbytes == 0 for layer in cache)
        require(clean, "state cleanup failed")
        return dict(
            input_tokens=len(ids),
            generated_tokens=len(generated),
            token_ids=generated,
            ttft_ms=prefill * 1000,
            prefill_tokens_per_second=len(ids) / prefill,
            decode_seconds=decode,
            decode_tokens_per_second=(len(generated) - 1) / decode,
            peak_inference_mlx_bytes=peak,
            cleanup=clean,
            finish_reason="stop" if generated[-1] in bundle.eos_token_ids else "length",
        )

    for _ in range(args.warmups):
        request()
    samples = [request() for _ in range(args.repetitions)]

    # Save first-token EOS too; its actual count excludes it from 128-output coverage.
    def sysctl(name):
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()

    report = dict(
        schema_version="mlx-one-performance-v1",
        checkpoint=args.model_id,
        revision=model_info["revision"],
        profile=args.profile,
        output_budget=128,
        timestamp=time.time(),
        config_sha256=sha(checkpoint / "config.json"),
        weight_sha256=hashes,
        prompt_token_ids=ids,
        prompt_source=(
            "custom-ids:" + sha(args.token_ids)
            if args.token_ids
            else "repeated-seed:The red book was on the table. The city is calm. "
        ),
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        runner_sha256=sha(Path(__file__)),
        source_sha256={
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src/mlx_one").rglob("*.py"))
        },
        environment=dict(
            protocol="standard-128-128",
            measurement_type="direct-model",
            backend="native MLX direct dense; last_only logits; production_fp16 KV",
            hardware=dict(
                chip_type=device["device_name"],
                physical_memory=f"{int(sysctl('hw.memsize')) / 1024**3:g} GB",
                machine_model=sysctl("hw.model"),
            ),
            platform=platform.platform(),
            python=platform.python_version(),
            packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-metal")},
            warmups_per_shape=args.warmups,
            repetitions=args.repetitions,
            statistic="per-column median",
            metal_device_info=device,
            greedy=True,
            eos_policy="respect all checkpoint EOS IDs",
        ),
        samples=samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    validate_samples(samples, 128, 128, args.repetitions)
    print(f"Saved {args.output}; actual outputs: {[s['generated_tokens'] for s in samples]}")


if __name__ == "__main__":
    main()
