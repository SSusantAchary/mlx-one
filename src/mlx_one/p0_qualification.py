"""Offline P0 readiness inventory and native dense restoration oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from mlx_one.engine.pressure import sample_telemetry
from mlx_one.inference_baseline import TOLERANCES, profiles, source_provenance
from mlx_one.provenance import collect_software_provenance


def inventory(root, *, checkpoints=(), test_log=None, probe=False):
    """Inspect local assets. Missing assets and Metal are independent blockers."""
    root = Path(root)
    assets, blockers = [], []
    for name, location in checkpoints:
        path = Path(location)
        files = {}
        if not path.is_dir():
            blockers.append({"checkpoint": name, "reason": "local checkpoint missing"})
        else:
            for item in sorted(path.iterdir()):
                if item.is_file() and item.suffix in {".json", ".jinja", ".safetensors"}:
                    digest = hashlib.sha256()
                    with item.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    files[item.name] = digest.hexdigest()
            if not any(key.endswith(".safetensors") for key in files):
                blockers.append({"checkpoint": name, "reason": "local weights missing"})
        assets.append({"checkpoint": name, "files": files})
    metal = {"status": "pending", "reason": "native probe not requested"}
    if probe:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import mlx.core as m; "
                 "assert m.metal.is_available(); print(m.default_device())"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            metal = {"status": "available" if result.returncode == 0 else "blocked",
                     "output": (result.stdout or result.stderr)[-2000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            metal = {"status": "blocked", "reason": type(exc).__name__}
        if metal["status"] == "blocked":
            blockers.append({"reason": "native Metal probe unavailable"})
    return {"schema_version": "native-p0-readiness-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_provenance(root),
            "software": asdict(collect_software_provenance()),
            "host": {"platform": platform.platform(), "telemetry": asdict(sample_telemetry())},
            "assets": assets, "metal": metal, "blockers": blockers,
            "test_output": Path(test_log).read_text() if test_log else None,
            "hardware_qualification": "pending", "qualification_promoted": False}


def write_inventory(output, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
    (output / "readiness.json").write_bytes(raw)
    (output / "checksums.json").write_text(json.dumps({
        "readiness.json": hashlib.sha256(raw).hexdigest()}, indent=2))


def p0_profiles(*, include_16k=False, include_32k=False):
    """Initial safe contexts; long-context diagnostics require explicit opt-in."""
    for mode in ("reference_dense", "experimental_block"):
        for profile in profiles(mode, models=("qwen",)):
            if profile.slots == 1:
                for workload in ("baseline", "p0_prefix", "parity"):
                    yield replace(profile, cache_type="f16", execution_dtype="float16",
                                  workload=workload)
        if include_16k or include_32k:
            seed = next(profiles(mode, models=("qwen",)))
            for context in ([16384] if include_16k else []) + ([32000] if include_32k else []):
                yield replace(seed, context_tokens=context, slots=1, workload="long",
                              cache_type="f16", execution_dtype="float16")


def dense_snapshot_parity(bundle, prompt_ids, boundary, *, steps=3, cache_type="f16"):
    """Compare full recomputation, cached decode, and a trimmed isolated snapshot."""
    import mlx.core as mx
    import numpy as np

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.engine.cache import CacheBundle
    from mlx_one.engine.reporting import tensor_dtypes

    prompt_ids = tuple(prompt_ids)
    if not 0 < boundary < len(prompt_ids) or steps < 1:
        raise ValueError("parity needs an interior boundary and positive steps")
    def fresh():
        return CacheBundle(tuple(configure_kv_caches(bundle.model.make_cache(),
                                                    cache_type, cache_type)))
    def forward(cache, ids):
        output = bundle.model(mx.array([ids]), cache=cache.entries)
        mx.eval(output.logits)
        return np.asarray(output.logits[0, -1].astype(mx.float32))
    cached, donor = fresh(), fresh()
    restored = None
    rows, ids = [], list(prompt_ids)
    try:
        forward(donor, prompt_ids)
        restored = donor.snapshot()
        if not restored.trim(len(prompt_ids) - boundary):
            raise ValueError("dense snapshot cannot trim to boundary")
        for index in range(steps):
            oracle = fresh()
            try:
                expected = forward(oracle, tuple(ids))
            finally:
                oracle.release()
            ordinary = forward(cached, prompt_ids if index == 0 else (ids[-1],))
            resumed = forward(restored, prompt_ids[boundary:] if index == 0 else (ids[-1],))
            observed = tensor_dtypes(bundle.model.parameters())
            atol, rtol = TOLERANCES[observed[0]]
            token_ids = [int(np.argmax(v)) for v in (expected, ordinary, resumed)]
            rows.append({"position": len(ids), "token_ids": token_ids,
                         "logits": [v.tolist() for v in (expected, ordinary, resumed)],
                         "max_error": max(float(np.max(np.abs(expected - v)))
                                          for v in (ordinary, resumed)),
                         "passed": len(set(token_ids)) == 1 and all(
                             np.allclose(expected, v, atol=atol, rtol=rtol)
                             for v in (ordinary, resumed)),
                         "offsets": [cached.offset, restored.offset]})
            ids.append(token_ids[0])
        return {"schema_version": "native-dense-restoration-parity-v1", "rows": rows,
                "passed": all(row["passed"] for row in rows),
                "restored_tokens": boundary, "donor_position": donor.offset}
    finally:
        for cache in (cached, donor, restored):
            if cache is not None:
                cache.release()


def prefix_gate(rows, common_prefix, *, threshold=.1):
    """A cache metric or text-only match cannot establish the P0 warm-prefix gate."""
    import statistics

    cold = [r for r in rows if r.get("temperature_phase") == "cold"]
    warm = [r for r in rows if r.get("temperature_phase") == "warm"]
    enough = len(cold) >= 3 and len(warm) >= 3
    checks = {"three_paired_repetitions": enough, "exact_prefix": common_prefix > 0,
              "completed": all(r.get("status") == "completed" for r in rows)}
    checks["token_id_parity"] = enough and all(
        a.get("output_token_ids") and a.get("output_token_ids") == b.get("output_token_ids")
        for a, b in zip(cold, warm, strict=False))
    checks["actual_skipped_prefill"] = enough and all(
        b["metrics"].get("reused_prompt_tokens", 0) > 0
        and a["metrics"].get("actual_prefill_tokens", 0)
        - b["metrics"].get("actual_prefill_tokens", 0)
        == b["metrics"].get("reused_prompt_tokens", 0)
        <= common_prefix for a, b in zip(cold, warm, strict=False))
    latency = [r.get("client_ttft_ms") for r in cold + warm]
    improvement = None
    if enough and all(v is not None and v > 0 for v in latency):
        improvement = 1 - statistics.median(r["client_ttft_ms"] for r in warm) / (
            statistics.median(r["client_ttft_ms"] for r in cold))
    checks["warm_p50_ttft"] = (improvement is not None and improvement >= threshold
                               if threshold is not None else True)
    return {"status": "passed" if all(checks.values()) else "failed",
            "checks": checks, "required_ttft_improvement": threshold,
            "observed_ttft_improvement": improvement, "qualification_promoted": False}


async def prefix_checks(client, bundle, engine, scheduler, profile):
    from mlx_one.inference_baseline import deterministic_prompt
    from mlx_one.inference_baseline_worker import stream_request

    content = deterministic_prompt(bundle.tokenizer, max(profile.context_tokens - 160, 32))
    donor = [{"role": "system", "content": content}, {"role": "user", "content": "suffix A"}]
    borrower = [donor[0], {"role": "user", "content": "suffix B"}]
    ids = [tuple(bundle.tokenizer.encode(bundle.chat_template.render(m)))
           for m in (donor, borrower)]
    common = 0
    for a, b in zip(*ids, strict=False):
        if a != b:
            break
        common += 1
    if max(map(len, ids)) + profile.output_tokens > bundle.context_length:
        raise ValueError("rendered P0 prefix exceeds checkpoint context")
    traces, rows = [], []
    previous = getattr(engine, "_qualification_token_sink", None)
    engine._qualification_token_sink = lambda request, tokens: traces.append(tokens)
    async def request(messages, phase, index):
        start = len(traces)
        result = await stream_request(client, profile, messages, phase=phase,
                                      group=f"p0-{index}-{phase}", index=index)
        result["output_token_ids"] = list(traces[-1]) if len(traces) > start else None
        rows.append(result)
    try:
        await request(borrower, "warmup", -1)
        for index in range(profile.repeats):
            scheduler.execute(engine.clear_caches)
            await request(borrower, "cold", index)
            scheduler.execute(engine.clear_caches)
            await request(donor, "donor", index)
            await request(borrower, "warm", index)
        gate = prefix_gate(rows, common, threshold=.1 if profile.context_tokens >= 8192 else None)
        return {**gate, "requests": rows, "exact_shared_tokens": common,
                "rendered_token_ids": [list(tokens) for tokens in ids],
                "namespace_class": "authenticated_principal",
                "threshold_scope": "named rendered-prefix workload"}
    finally:
        engine._qualification_token_sink = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--include-16k", action="store_true")
    parser.add_argument("--include-32k", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=LOCAL_PATH")
    parser.add_argument("--tests-log", type=Path)
    parser.add_argument("--probe-metal", action="store_true")
    args = parser.parse_args(argv)
    items = list(p0_profiles(include_16k=args.include_16k, include_32k=args.include_32k))
    if args.list_profiles:
        print(json.dumps([asdict(p) for p in items], indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required unless listing profiles")
    if args.run:
        from mlx_one.hardware import detect_hardware
        from mlx_one.inference_baseline import run_profile

        root = Path(__file__).resolve().parents[2]
        host = detect_hardware().to_dict()
        provenance = source_provenance(root)
        outcomes = [run_profile(p, args.output / p.model_id.replace("/", "--") / p.profile_id,
                                hardware=host, provenance=provenance) for p in items]
        return int(any(row["status"] != "completed" for row in outcomes))
    checkpoints = []
    for value in args.checkpoint:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--checkpoint requires NAME=LOCAL_PATH")
        checkpoints.append((name, path))
    manifest = inventory(Path(__file__).resolve().parents[2], checkpoints=checkpoints,
                         test_log=args.tests_log, probe=args.probe_metal)
    write_inventory(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
