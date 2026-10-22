"""Standalone comparator workers; external engines never enter the native runtime."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_one.inference_baseline import process_rss, source_provenance
from mlx_one.inference_qualification import validate_integrity
from mlx_one.native_qualification import preflight
from mlx_one.p0_engine_qualification import atomic_json, digest, utc, validate_assets
from qualification.p01_comparison import read_worker
from qualification.p01_experiments import VERSION, validate_controlled_result, validate_sources
from qualification.p01_math import run_task, write_integrity

OMLX_REVISION = "03bc3dd7118a5f1bbf9cb9ac119f551c77477b09"


def comparator_inventory(get_version=version):
    packages = {}
    for name in ("mlx", "mlx-lm", "omlx", "numpy"):
        try:
            packages[name] = get_version(name)
        except PackageNotFoundError:
            packages[name] = None
    blockers = []
    if packages["omlx"] is None:
        blockers.append("pinned oMLX is not installed")
    if packages["mlx"] != "0.32.4":
        blockers.append("pinned oMLX requires MLX 0.32.4; preserve the qualification environment")
    if packages["numpy"] is not None:
        from packaging.version import Version

        if Version(packages["numpy"]) >= Version("2.4"):
            blockers.append("pinned oMLX requires NumPy below 2.4")
    return {"packages": packages, "omlx_revision": OMLX_REVISION,
        "mlx_lm": "AVAILABLE" if packages["mlx-lm"] else "BLOCKED",
        "omlx": "BLOCKED" if blockers else "NOT_RUN",
        "omlx_blockers": blockers, "qualification_promoted": False}


def validate_external(result, task, metadata, source):
    if (result.get("source") != source or result.get("backend") != "mlx_lm"
            or result.get("identity") != task["peer_identity"]):
        raise ValueError("external checkpoint, precision, source or workload substitution")
    validate_controlled_result(result, task, require_native_ownership=False)
    if result.get("live_comparator_cache_states") != 0:
        raise ValueError("external comparator cache state remains owned")
    if result.get("packed_payload_preserved") is not True:
        raise ValueError("external floating conversion changed packed payload")
    if result.get("native_parity", {}).get("status") != "PASS":
        raise ValueError("standalone comparator needs matching native logits parity")


def validate_peer_pair(peer, parity):
    """A short parity receipt cannot qualify a different decode continuation."""
    identity = peer["identity"]
    if (identity.get("measurement_kind") != "synchronized_prepared_history_forwards"
            or identity.get("profile") is not False
            or peer.get("correctness_passed") is not True
            or parity.get("correctness_passed") is not True
            or parity.get("native_parity", {}).get("status") != "PASS"):
        raise ValueError("standalone comparison requires unprofiled prepared-history and parity")
    for key in ("checkpoint", "revision", "assets_sha256", "prompt_ids", "source_sha256",
                "execution_dtype", "kv_dtype", "environment", "tokenizer_identity",
                "context_limit"):
        if key not in identity or identity[key] != parity.get("identity", {}).get(key):
            raise ValueError("standalone timing and native parity workloads differ: " + key)
    fixture = identity.get("output_fixture")
    if (not isinstance(fixture, list) or len(fixture) not in {128, 256}
            or parity["native_parity"].get("forced_tokens") != fixture):
        raise ValueError("native parity must cover the exact controlled continuation")
    return identity


def worker(manifest_path, destination):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from mlx_one.engine.pressure import activation_forecast
    from mlx_one.p0_engine_worker import error_metrics

    manifest = json.loads(Path(manifest_path).read_text())
    task, native, source = manifest["task"], manifest["native"], manifest["source"]
    validate_sources(manifest["tool_sources"], source)
    if (source_provenance(Path.cwd()) != source
            or digest(__file__) != manifest["tool_sha256"]):
        raise ValueError("standalone comparator source changed")
    peer, proof = Path(task["peer"]), Path(task["parity_peer"])
    if digest(peer) != task["peer_sha256"] or digest(proof) != task["parity_sha256"]:
        raise ValueError("matched native artifacts changed")
    validate_peer_pair(read_worker(peer), read_worker(proof))
    metadata = native["metadata"]
    actual = validate_assets(metadata["model_id"], metadata["snapshot_path"], metadata["revision"])
    if actual["assets"] != metadata["assets"]:
        raise ValueError("immutable comparator checkpoint changed")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"
    def beat(phase, seconds=600):
        atomic_json(heartbeat, {"status": "running", "phase": phase,
                                "started": time.monotonic(), "timeout": seconds})
    beat("load", 120)
    mx.set_default_device(mx.gpu)
    model, tokenizer, config = load(metadata["snapshot_path"], return_config=True)
    if config.get("model_type") not in {"qwen2", "qwen3"}:
        raise ValueError("standalone Phase-A adapter supports validated Qwen2/Qwen3 dense state")
    packed = [(m, m.weight) for _, m in model.named_modules() if hasattr(m, "bits")]
    model.set_dtype(mx.float16, predicate=lambda dtype:
                    str(dtype).rsplit(".", 1)[-1] in {"float16", "bfloat16", "float32"})
    mx.eval(model.parameters())
    if any(module.weight is not payload for module, payload in packed):
        raise ValueError("comparator conversion modified packed weight identity")
    prompt = json.loads((peer.parent / "prompt.json").read_text())
    if tokenizer.encode(prompt["text"]) != task["peer_identity"]["prompt_ids"]:
        raise ValueError("comparator tokenization differs from exact native input")
    ids, continuation = prompt["token_ids"], task["forced_tokens"]
    ceiling = native["environment"]["preflight"]["process_ceiling_bytes"]
    live_caches = set()
    def execute(tokens, *, numerical=False):
        cache = make_prompt_cache(model)
        identity = id(cache)
        live_caches.add(identity)
        try:
            width = task["chunk"] or len(ids)
            prompt_parts = [mx.array([ids[start:start + width]])
                            for start in range(0, len(ids), width)]
            following = [mx.array([[token]]) for token in tokens]
            mx.eval(*prompt_parts, *following)
            estimate = activation_forecast(SimpleNamespace(**config), batch=1,
                                           chunk=min(width, len(ids)), history=0)["bytes"]
            observed = max(process_rss(os.getpid()), mx.get_active_memory())
            if observed + estimate >= ceiling:
                raise MemoryError("standalone analytical forecast exceeds unchanged host ceiling")
            vectors, gaps = [], []
            mx.synchronize()
            started = time.perf_counter()
            for values in prompt_parts:
                output = model(values, cache=cache)
                mx.eval(output, [entry.state for entry in cache])
                mx.synchronize()
            prefill = time.perf_counter() - started
            if numerical:
                vectors.append(np.asarray(output[0, -1].astype(mx.float32)).copy())
            decode_started = time.perf_counter()
            for values in following:
                before = time.perf_counter()
                output = model(values, cache=cache)
                mx.eval(output, [entry.state for entry in cache])
                mx.synchronize()
                gaps.append(time.perf_counter() - before)
                if numerical:
                    vectors.append(np.asarray(output[0, -1].astype(mx.float32)).copy())
            decode = time.perf_counter() - decode_started
            for entry in cache:
                keys = getattr(entry, "keys", None)
                if keys is not None and keys.dtype != mx.float16:
                    raise ValueError("standalone observed attention KV is not FP16")
            return vectors if numerical else {"prefill_tps": len(ids) / prefill,
                "decode_tps": len(tokens) / decode, "prefill_seconds": prefill,
                "decode_seconds": decode, "forced_tokens": tokens,
                "output_kind": "forced_not_generated", "prepared_history_position": len(ids),
                "committed_position": len(ids) + len(tokens), "actual_batch_sizes": [1],
                "actual_prefill_tokens": len(ids), "synchronized_decode_forward_seconds": gaps}
        finally:
            del cache
            live_caches.remove(identity)
            gc.collect()
    result = {"schema_version": VERSION, "backend": "mlx_lm",
        "backend_version": version("mlx-lm"), "source": source,
        "identity": task["peer_identity"], "measurements": [], "packed_payload_preserved": True,
        "correctness_passed": False, "status": "running", "qualification_promoted": False}
    try:
        beat("parity")
        with np.load(proof.parent / "logits.npz") as oracle:
            if oracle["forced_tokens"].tolist() != continuation:
                raise ValueError("numerical payload does not cover the controlled continuation")
            values = execute(oracle["forced_tokens"].tolist(), numerical=True)
            metrics = [error_metrics(left, right) for left, right in
                       zip(oracle["reference"], values, strict=True)]
        passed = all(m["within_tolerance"] and m["greedy_agreement"] for m in metrics)
        result["native_parity"] = {"status": "PASS" if passed else "FAIL", "logits": metrics}
        if not passed:
            raise AssertionError("standalone same-checkpoint logits parity failed")
        result["correctness_passed"] = True
        beat("warmup")
        execute(continuation)
        for repetition in range(task["repetitions"]):
            beat("measurement")
            mx.synchronize()
            mx.reset_peak_memory()
            started = time.monotonic()
            value = execute(continuation)
            result["measurements"].append({**value, "repetition": repetition,
                "started": started, "finished": time.monotonic(), "instrumented": False,
                "peak_mlx_bytes": mx.get_peak_memory(), "active_mlx_bytes": mx.get_active_memory(),
                "cached_mlx_bytes": mx.get_cache_memory(), "rss_peak_bytes": None})
            atomic_json(root / "experiment.json", result)
        result.update(status="PASS", live_comparator_cache_states=len(live_caches),
                      native_cache_manager_counters="NOT_APPLICABLE")
    except BaseException as error:
        result.update(status="FAIL", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        atomic_json(root / "experiment.json", result)
        atomic_json(heartbeat, {"status": "finished"})
        write_integrity(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--parity-peer", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("qualification/results/p01_math/external"))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    inventory = comparator_inventory()
    if args.inspect:
        print(json.dumps(inventory, indent=2))
        return 0
    if not args.peer or not args.parity_peer or args.repetitions < 1:
        parser.error("--peer controlled native result and --parity-peer receipt are required")
    peer = read_worker(args.peer)
    parity = read_worker(args.parity_peer)
    try:
        identity = validate_peer_pair(peer, parity)
    except ValueError as error:
        parser.error(str(error))
    validate_integrity(args.parity_peer.parent, required={"logits.npz"})
    manifest = json.loads((args.peer.parent.parent / "manifest.json").read_text())
    source = source_provenance(Path.cwd())
    if source != manifest["source"]:
        parser.error("current source/environment differs from native peer; rerun matched peers")
    task = {**manifest["task"], "id": str(time.time_ns()), "route": "mlx_lm",
        "peer": str(args.peer.resolve()), "peer_sha256": digest(args.peer),
        "parity_peer": str(args.parity_peer.resolve()), "parity_sha256": digest(args.parity_peer),
        "peer_identity": identity, "repetitions": args.repetitions, "status": "NOT_RUN"}
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    report = {"inventory": inventory, "status": "NOT_RUN", "timestamp": utc(),
              "qualification_promoted": False}
    if args.run:
        if inventory["mlx_lm"] != "AVAILABLE":
            report.update(status="BLOCKED", reason="standalone MLX-LM unavailable")
        else:
            observed = preflight(root / "preflight")
            native = manifest["native"]
            tools = {str(p.resolve()): digest(p) for p in Path(__file__).parent.glob("p01*.py")}
            report["task"] = run_task(root, task, native["metadata"], native["environment"],
                source, observed, worker_entry=__file__, validator=validate_external,
                tool_sources=tools, schema_version=VERSION)
            report["status"] = report["task"]["status"]
    atomic_json(root / "manifest.json", report)
    write_integrity(root)
    print(root)
    return int(report["status"] not in {"PASS", "NOT_RUN"})


if __name__ == "__main__":
    raise SystemExit(main())
