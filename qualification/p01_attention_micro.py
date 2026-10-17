"""Bounded stock-SDPA probes; results never enable a runtime attention route."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_one.inference_baseline import source_provenance
from mlx_one.inference_qualification import write_integrity
from mlx_one.native_qualification import preflight
from mlx_one.p0_engine_qualification import atomic_json, digest, utc, validate_assets
from qualification.p01_math import canonical_digest, run_task, summarize


def attention_layouts(model):
    """Discover actual attention modules; convolution/recurrent layers are excluded."""
    layouts = {}
    for name, module in model.named_modules():
        if not all(hasattr(module, field) for field in ("q_proj", "k_proj", "v_proj")):
            continue
        config = getattr(module, "config", None)
        heads = getattr(module, "num_heads", getattr(config, "num_attention_heads", None))
        kv_heads = getattr(module, "num_kv_heads", getattr(config, "num_key_value_heads", None))
        dim = getattr(module, "head_dim", getattr(config, "head_dim", None))
        if any(type(value) is not int or value < 1 for value in (heads, kv_heads, dim)):
            raise ValueError("attention probe needs explicit native head dimensions")
        if heads % kv_heads:
            raise ValueError("invalid native GQA layout")
        key = heads, kv_heads, dim
        layouts.setdefault(key, {"heads": heads, "kv_heads": kv_heads, "head_dim": dim,
                                 "modules": []})["modules"].append(name)
    return list(layouts.values())


def attention_bound(layout, query_tokens, key_tokens):
    h, hk, d = (layout[key] for key in ("heads", "kv_heads", "head_dim"))
    if (any(type(n) is not int or n < 1 for n in (h, hk, d, query_tokens, key_tokens))
            or h % hk or query_tokens > key_tokens):
        raise ValueError("positive lower-right causal GQA dimensions required")
    # Conservative reference scores/probabilities, repeats, both outputs and FP32 casts.
    # No peak-memory benefit is inferred from this paired numerical probe.
    return h * query_tokens * key_tokens * 24 + (h * query_tokens + 2 * h * key_tokens) * d * 16


def validate_result(result, task, metadata, source):
    if (result.get("source") != source or result.get("checkpoint") != metadata["model_id"]
            or result.get("revision") != metadata["revision"] or not result.get("cases")
            or result.get("diagnostic") is not True or result.get("qualification_promoted")):
        raise ValueError("attention probe identity or diagnostic scope mismatch")
    if any(result.get("ownership", {}).get(key) != 0 for key in
           ("active_handles", "reserved_bytes", "allocated_bytes")):
        raise ValueError("attention probe cleanup incomplete")
    if any(row["status"] == "FAIL" for row in result["cases"]):
        raise ValueError("attention probe failed unchanged numerical tolerance")


def worker(manifest_path, destination):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.attention import attention_reference
    from mlx_one.p0_engine_worker import Cases, error_metrics

    manifest = json.loads(Path(manifest_path).read_text())
    source, task, native = manifest["source"], manifest["task"], manifest["native"]
    if source_provenance(Path.cwd()) != source or digest(__file__) != manifest["tool_sha256"]:
        raise ValueError("attention probe source/environment changed")
    metadata = native["metadata"]
    actual = validate_assets(metadata["model_id"], metadata["snapshot_path"], metadata["revision"])
    if actual["assets"] != metadata["assets"]:
        raise ValueError("attention probe immutable assets changed")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"
    def beat(phase):
        atomic_json(heartbeat, {"status": "running", "phase": phase,
                                "started": time.monotonic(), "timeout": 600})
    beat("load")
    cases = Cases(native, root)
    runtime = cases.engine._cache_manager
    result = {"source": source, "checkpoint": metadata["model_id"],
        "revision": metadata["revision"], "cases": [], "status": "running", "diagnostic": True,
        "qualification_promoted": False, "gpu_kernel_count": None,
        "scope": "random operands in native layouts; no checkpoint or end-to-end promotion",
        "peak_scope": "paired reference-output overlap; not a peak-memory comparison"}
    try:
        for layout in attention_layouts(cases.bundle.model):
            for query_tokens in task["query_tokens"]:
                for offset in task["offsets"]:
                    key_tokens = query_tokens + offset
                    row = {**layout, "query_tokens": query_tokens, "key_tokens": key_tokens,
                        "logical_query_start": offset, "mask": "causal", "status": "NOT_RUN",
                        "measurements": {}}
                    result["cases"].append(row)
                    bound = attention_bound(layout, query_tokens, key_tokens)
                    if bound > runtime.config.memory_budget_bytes:
                        row.update(status="BLOCKED_CAPACITY", reason="conservative paired scratch")
                        continue
                    beat("attention")
                    def execute(bound=bound, layout=layout, query_tokens=query_tokens,
                                key_tokens=key_tokens, row=row):
                        with runtime.transient_reservation(bound):
                            mx.random.seed(0)
                            h, hk, d = (layout[k] for k in ("heads", "kv_heads", "head_dim"))
                            q = mx.random.normal((1, h, query_tokens, d)).astype(mx.float16)
                            k = mx.random.normal((1, hk, key_tokens, d)).astype(mx.float16)
                            v = mx.random.normal(k.shape).astype(mx.float16)
                            mx.eval(q, k, v)
                            def forward(route):
                                if route == "reference":
                                    return attention_reference(q, k, v, scale=d**-.5,
                                        mask="causal", stable=True, promote=True)
                                dtype = mx.float16 if route == "stock_fp16" else mx.float32
                                return mx.fast.scaled_dot_product_attention(
                                    q.astype(dtype), k.astype(dtype), v.astype(dtype),
                                    scale=d**-.5, mask="causal", force_fused=True)
                            expected = forward("reference")
                            mx.eval(expected)
                            reference_host = np.asarray(expected.astype(mx.float32)).copy()
                            parity = {}
                            for route in ("reference", "stock_fp16", "stock_fp32"):
                                value = forward(route)
                                mx.eval(value)
                                timings = []
                                for _ in range(task["repetitions"]):
                                    value = None
                                    mx.synchronize()
                                    mx.reset_peak_memory()
                                    started = time.perf_counter()
                                    value = forward(route)
                                    mx.eval(value)
                                    mx.synchronize()
                                    timings.append({"seconds": time.perf_counter() - started,
                                        "peak_mlx_bytes": mx.get_peak_memory(),
                                        "retained_reference_output_bytes": expected.nbytes})
                                row["measurements"][route] = timings
                                metrics = error_metrics(reference_host,
                                    np.asarray(value.astype(mx.float32)), "float16")
                                metrics.pop("greedy_agreement")  # Attention outputs are not logits.
                                parity[route] = metrics
                                row.setdefault("seconds", {})[route] = summarize(
                                    item["seconds"] for item in timings)
                            return parity
                    row["parity"] = cases.engine._device.call(id(cases.engine), execute)
                    row["status"] = ("PASS" if all(m["within_tolerance"] for m in
                                                   row["parity"].values()) else "FAIL")
                    atomic_json(root / "experiment.json", result)
                    if row["status"] == "FAIL":
                        raise AssertionError("stock attention probe failed; stop shape expansion")
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        cases.close()
        result["ownership"] = runtime.stats().to_dict()
        atomic_json(root / "experiment.json", result)
        atomic_json(heartbeat, {"status": "finished"})
        write_integrity(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--query-tokens", type=int, nargs="+", default=[1, 2, 16, 127, 128, 512])
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 512])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("qualification/results/p01_math/attention-micro"))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if (not args.baseline or args.repetitions < 1 or not args.query_tokens or not args.offsets
            or any(q < 1 or q > 8192 for q in args.query_tokens)
            or any(o < 0 or o > 8192 for o in args.offsets)):
        parser.error("immutable baseline, positive repetitions, queries 1..8192, offsets 0..8192")
    assets = json.loads((args.baseline / "download_manifest.json").read_text())
    environment = json.loads((args.baseline / "environment.json").read_text())
    models = [v["metadata"] for v in assets.values() if v["status"] == "PASS"
              and (args.model is None or v["metadata"]["model_id"] in args.model)]
    if not models or args.model and set(args.model) - {m["model_id"] for m in models}:
        parser.error("requested pinned assets unavailable; no substitution")
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    source = source_provenance(Path.cwd())
    tasks = [{"id": canonical_digest([m["model_id"], args.query_tokens, args.offsets])[:24],
        "checkpoint": m["model_id"], "query_tokens": args.query_tokens, "offsets": args.offsets,
        "repetitions": args.repetitions, "status": "NOT_RUN"} for m in models]
    report = {"tasks": tasks, "source": source, "timestamp": utc(), "diagnostic": True,
              "qualification_promoted": False}
    try:
        if args.run:
            observed = preflight(root / "preflight")
            for index, (task, model) in enumerate(zip(tasks, models, strict=True)):
                tasks[index] = run_task(root, task, model, environment, source, observed,
                    worker_entry=__file__, validator=validate_result)
                if tasks[index]["status"] != "PASS":
                    break
    finally:
        atomic_json(root / "manifest.json", report)
        write_integrity(root)
    print(root)
    return int(any(t["status"] not in {"PASS", "NOT_RUN"} for t in tasks))


if __name__ == "__main__":
    raise SystemExit(main())
