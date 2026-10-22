#!/usr/bin/env python3
"""Local-only full-checkpoint benchmark. Each backend runs in a fresh process."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import resource
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx


def cache_bytes(cache):
    total = 0
    for layer in cache:
        if hasattr(layer, "allocated_nbytes"):
            total += layer.nbytes  # Native API reports logical active state.
            continue
        for value in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if value is not None:
                active = min(layer.offset, getattr(layer, "max_size", layer.offset), value.shape[2])
                total += value.nbytes * active // value.shape[2]
    return total


def cache_allocated_bytes(cache):
    return sum(getattr(layer, "allocated_nbytes", layer.nbytes) for layer in cache)


def sample(logits, temperature, top_p, rng):
    # Deliberately common sampling code for both direct backends.
    if temperature == 0:
        return int(mx.argmax(logits).item())
    probs = mx.softmax(logits / temperature).tolist()
    ranked = sorted(enumerate(probs), key=lambda pair: pair[1], reverse=True)
    kept = []
    mass = 0
    for token, p in ranked:
        kept.append((token, p))
        mass += p
        if mass >= top_p:
            break
    threshold = rng.random() * mass
    for token, p in kept:
        threshold -= p
        if threshold <= 0:
            return token
    return kept[-1][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backend", choices=["native", "reference"], required=True)
    p.add_argument("--checkpoint-revision", default="20ba1835e625264e07366d215933fe31a9940a6a")
    p.add_argument("--interface", choices=["direct", "python"], default="direct")
    p.add_argument(
        "--token-ids", type=Path, required=True, help="JSON list, or object with input_ids"
    )
    p.add_argument("--input-lengths", default="128,512")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kv-dtype", choices=["float16"], default="float16")
    p.add_argument("--skip-lengths", default="")
    p.add_argument("--memory-fraction", type=float, default=0.8)
    p.add_argument("--profile-layers", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.checkpoint.is_dir():
        p.error("checkpoint must be an existing local directory; downloads are disabled")
    if a.interface == "python" and a.backend != "native":
        p.error("reference comparison uses direct interface; Python Engine is measured separately")
    if (
        a.repetitions < 1
        or a.warmups < 1
        or not 1 <= a.max_tokens <= 4096
        or not 0 < a.memory_fraction < 1
    ):
        p.error("invalid repetitions/warmups/token limit/memory fraction")
    if a.temperature < 0 or not 0 < a.top_p <= 1:
        p.error("invalid sampling settings")
    config = json.loads((a.checkpoint / "config.json").read_text())
    generation_path = a.checkpoint / "generation_config.json"
    gen = json.loads(generation_path.read_text()) if generation_path.exists() else {}
    eos = gen.get("eos_token_id", config.get("eos_token_id", []))
    eos = {eos} if isinstance(eos, int) else set(eos)
    ids = json.loads(a.token_ids.read_text())
    ids = ids.get("input_ids") if isinstance(ids, dict) else ids
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or x < 0 or x >= config["vocab_size"] for x in ids)
    ):
        p.error("input_ids must be a flat nonempty list of valid vocabulary IDs")
    index_path = a.checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    # Header sizes are negligible; packed file bytes are a lower memory bound.
    files = set(index.get("weight_map", {}).values()) or {"model.safetensors"}
    packed_bytes = sum((a.checkpoint / name).stat().st_size for name in files)
    device = mx.device_info()
    host_available = None
    try:
        vm = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True, text=True).stdout
        page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
        pages = sum(
            int(re.search(label + r":\s+(\d+)", vm).group(1))
            for label in ("Pages free", "Pages inactive", "Pages speculative")
        )
        host_available = pages * page_size
    except (OSError, subprocess.CalledProcessError, AttributeError):
        pass
    budget = int(device["max_recommended_working_set_size"] * a.memory_fraction)
    if host_available is not None:
        budget = min(budget, int(host_available * a.memory_fraction))
    if packed_bytes > budget:
        p.error(
            f"packed checkpoint {packed_bytes} bytes exceeds chosen working-set budget {budget}"
        )
    weight_hashes = {}
    for name in sorted(files):
        if Path(name).name != name:
            p.error("unsafe shard path")
        digest = hashlib.sha256()
        with (a.checkpoint / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        weight_hashes[name] = digest.hexdigest()
    started = time.perf_counter()
    if a.backend == "native":
        from mlx_one.text.loading import load_text_model

        bundle = load_text_model(a.checkpoint, offline=True)
        model, tokenizer = bundle.model, bundle.tokenizer

        def forward(tokens, cache):
            return model(tokens, cache=cache, logits_mode="last_only").logits

        make_cache = model.make_cache
    else:
        from mlx_lm.models.cache import make_prompt_cache
        from turboquant_mlx.generate import load_turboquant

        model, tokenizer = load_turboquant(str(a.checkpoint), lazy=False)
        import mlx.nn as nn

        class LastTokenHead(nn.Module):
            def __init__(self, head):
                super().__init__()
                self.head = head

            def __call__(self, x):
                return self.head(x[:, -1:, :])

        model.lm_head = LastTokenHead(model.lm_head)

        def forward(tokens, cache):
            return model(tokens, cache=cache)

        def make_cache():
            cache = make_prompt_cache(model)
            for layer in cache:
                original = layer.update_and_fetch
                layer.update_and_fetch = lambda k, v, original=original: original(
                    k.astype(mx.float16), v.astype(mx.float16)
                )
            return cache

    mx.eval(model.parameters())
    mx.synchronize()
    load_seconds = time.perf_counter() - started
    a.output.parent.mkdir(parents=True, exist_ok=True)
    engine = None
    if a.interface == "python":
        from mlx_one import CacheConfig, Engine

        engine = Engine(bundle, cache_config=CacheConfig(prefix_cache_enabled=False))
    report = dict(
        schema_version="tq3-full-model-v1",
        backend=a.backend,
        interface=a.interface,
        checkpoint=str(a.checkpoint.resolve()),
        checkpoint_revision=a.checkpoint_revision,
        weight_sha256=weight_hashes,
        config_sha256=hashlib.sha256((a.checkpoint / "config.json").read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest()
        if index_path.exists()
        else None,
        token_file_sha256=hashlib.sha256(a.token_ids.read_bytes()).hexdigest(),
        mlx=importlib.metadata.version("mlx"),
        platform=platform.platform(),
        device=device,
        load_seconds=load_seconds,
        preflight_available_memory_bytes=host_available,
        preflight_memory_budget_bytes=budget,
        packed_file_bytes=packed_bytes,
        settings=vars(a).copy(),
        full_checkpoint_performance_status=(
            "MEASURED_LOCAL_CHECKPOINT"
            if config.get("num_hidden_layers") == 24
            and config.get("num_local_experts") == 32
            and config.get("hidden_size") == 2880
            and a.checkpoint_revision != "synthetic"
            else "SYNTHETIC_DRIVER_TEST"
        ),
        results=[],
        skipped=[],
    )
    report["settings"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in report["settings"].items()
    }
    report["comparability_notes"] = [
        "Direct backends use identical Python sampling and FP16 KV storage.",
        (
            "Both direct backends project only the final prompt position; "
            "reference uses a benchmark-only last-token head wrapper."
        ),
        "Instrumented layer samples are a separate run and excluded from throughput.",
        (
            "Native honors checkpoint YaRN truncate=false; pinned MLX-LM 0.32.2 "
            "rounds correction bounds with floor/ceil. Outputs/EOS may differ."
        ),
        (
            "No simultaneous backend/model residency. "
            "Report actual output counts including terminal EOS."
        ),
    ]

    def run(prompt):
        cache = make_cache()
        rng = random.Random(a.seed)
        out = []
        mx.synchronize()
        start = time.perf_counter()
        logits = forward(mx.array([prompt], mx.int32), cache)
        mx.eval(logits)
        mx.synchronize()
        prefill_end = time.perf_counter()
        token = sample(logits[0, -1], a.temperature, a.top_p, rng)
        out.append(token)
        first = time.perf_counter()
        decode_times = []
        while len(out) < a.max_tokens and token not in eos:
            tick = time.perf_counter()
            logits = forward(mx.array([[token]], mx.int32), cache)
            mx.eval(logits)
            mx.synchronize()
            token = sample(logits[0, -1], a.temperature, a.top_p, rng)
            out.append(token)
            decode_times.append(time.perf_counter() - tick)
        end = time.perf_counter()
        decode_seconds = sum(decode_times)
        result = dict(
            ttft_seconds=first - start,
            prefill_seconds=prefill_end - start,
            prefill_tokens_per_second=len(prompt) / (prefill_end - start),
            decode_seconds=decode_seconds,
            decode_tokens_per_second=len(decode_times) / decode_seconds if decode_seconds else None,
            generated_tokens=len(out),
            non_eos_generated_tokens=sum(t not in eos for t in out),
            decode_steps=len(decode_times),
            requested_output_tokens=a.max_tokens,
            finish_reason="stop" if out[-1] in eos else "length",
            output_token_ids=out,
            total_seconds=end - start,
            cache_state_bytes=cache_bytes(cache),
            cache_allocated_bytes=cache_allocated_bytes(cache),
            peak_mlx_active_bytes=mx.get_peak_memory(),
            process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        return result

    try:
        for length in [int(x) for x in a.input_lengths.split(",")]:
            if length <= 0:
                p.error("positive input lengths required")
            if str(length) in a.skip_lengths.split(","):
                report["skipped"].append(dict(input_tokens=length, reason="user skipped"))
                continue
            # Conservative upper bound: full KV at every layer, QKV and logits scratch.
            kv = (
                2
                * config["num_hidden_layers"]
                * (length + a.max_tokens)
                * config["num_key_value_heads"]
                * config["head_dim"]
                * 2
            )
            scratch = length * config["vocab_size"] * 4 + 256 * 1024**2
            if (
                packed_bytes + kv + scratch > budget
                or length + a.max_tokens > config["max_position_embeddings"]
            ):
                report["skipped"].append(
                    dict(
                        input_tokens=length,
                        reason="conservative memory/context limit",
                        estimated_bytes=packed_bytes + kv + scratch,
                    )
                )
                continue
            if len(ids) < length:
                report["skipped"].append(
                    dict(input_tokens=length, reason="insufficient saved input IDs")
                )
                continue
            prompt = ids[:length]
            prompt_hash = hashlib.sha256(
                json.dumps(prompt, separators=(",", ":")).encode()
            ).hexdigest()

            def execute(prompt=prompt):
                if engine is None:
                    return run(prompt)
                text = tokenizer.decode(prompt, skip_special_tokens=False)
                if tokenizer.encode(text) != prompt:
                    raise ValueError(
                        "Python prompt decode/encode does not preserve saved IDs; "
                        "use direct interface"
                    )
                start = time.perf_counter()
                result = engine.generate(
                    text,
                    max_tokens=a.max_tokens,
                    temperature=a.temperature,
                    top_p=a.top_p,
                    seed=a.seed,
                )
                return dict(
                    total_seconds=time.perf_counter() - start,
                    generated_tokens=result.generation_tokens,
                    finish_reason=result.finish_reason,
                    timings=result.timings,
                    runtime=engine.runtime_stats(),
                )

            mx.reset_peak_memory()
            cold = execute()
            warmups = []
            for _ in range(a.warmups):
                t = time.perf_counter()
                execute()
                warmups.append(time.perf_counter() - t)
            samples = []
            with dispatch_context(a.backend) as dispatch:
                for _ in range(a.repetitions):
                    mx.reset_peak_memory()
                    samples.append(execute())
            entry = dict(
                input_tokens=length,
                prompt_token_sha256=prompt_hash,
                cold=cold,
                warmup_seconds=warmups,
                samples=samples,
                dispatch_statistics=dict(dispatch),
            )
            if engine is None:
                entry["warm_ttft_median_seconds"] = statistics.median(
                    x["ttft_seconds"] for x in samples
                )
                tps = [
                    x["decode_tokens_per_second"]
                    for x in samples
                    if x["decode_tokens_per_second"] is not None
                ]
                entry["warm_decode_tps_median"] = statistics.median(tps) if tps else None
            if a.profile_layers and engine is None:
                with profile_layers(model) as events:
                    run(prompt)
                entry["instrumented_layer_samples"] = events
                entry["instrumented_moe_seconds"] = sum(
                    e["seconds"] for e in events if e["kind"] == "moe"
                )
            report["results"].append(entry)
            a.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    finally:
        if engine:
            engine.close()
        report["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["wired_memory_bytes"] = None  # MLX does not expose process wired memory.
        a.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            dict(
                output=str(a.output),
                workloads=len(report["results"]),
                skipped=len(report["skipped"]),
            )
        )
    )


@contextlib.contextmanager
def dispatch_context(backend):
    if backend == "native":
        from mlx_one.quantization.turboquant.dispatch import collect_dispatch

        with collect_dispatch() as stats:
            yield stats
    else:
        yield {"availability": "reference does not expose native telemetry"}


@contextlib.contextmanager
def profile_layers(model):
    # Instrument class dispatch temporarily; keep wrappers out of timed runs.
    events = []
    originals = {}
    names = {}
    for i, layer in enumerate(model.model.layers):
        names[id(layer)] = (i, "decoder")
        names[id(layer.mlp)] = (i, "moe")
    try:
        for module in [obj for layer in model.model.layers for obj in (layer, layer.mlp)]:
            cls = type(module)
            if cls in originals:
                continue
            original = cls.__call__
            originals[cls] = original

            def wrapped(self, *args, original=original, **kwargs):
                mx.synchronize()
                start = time.perf_counter()
                out = original(self, *args, **kwargs)
                mx.eval(out)
                mx.synchronize()
                layer, kind = names[id(self)]
                events.append(dict(layer=layer, kind=kind, seconds=time.perf_counter() - start))
                return out

            cls.__call__ = wrapped
        yield events
    finally:
        for cls, original in originals.items():
            cls.__call__ = original


if __name__ == "__main__":
    main()
