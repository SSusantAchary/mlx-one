"""Supervised real-checkpoint packed-projection microbenchmarks (diagnostic only)."""
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
from qualification.p01_math import canonical_digest, run_task


def projection_shapes(model):
    """Every distinct packed layout, with all represented source module names."""
    groups = {}
    for name, module in model.named_modules():
        if not hasattr(module, "bits"):
            continue
        key = (tuple(module.weight.shape), module.bits, module.group_size,
               str(module.weight.dtype), str(module.scales.dtype), type(module).__name__)
        if key not in groups:
            groups[key] = {"module": module, "names": [], "bits": module.bits,
                "group_size": module.group_size, "packed_shape": list(module.weight.shape),
                "width": module.weight.shape[-1] * (32 // module.bits),
                "outputs": module.weight.shape[0], "as_linear": hasattr(module, "as_linear")}
        groups[key]["names"].append(name)
    return list(groups.values())


def projection_bound(rows, width, outputs):
    if any(type(value) is not int or value < 1 for value in (rows, width, outputs)):
        raise ValueError("positive actual projection dimensions required")
    # Both evaluated results, FP32 numerical reductions, input casts and overlap.
    return rows * (width * 8 + outputs * 16)


def validate_result(result, task, metadata, source):
    if (result.get("source") != source or result.get("checkpoint") != metadata["model_id"]
            or result.get("revision") != metadata["revision"] or not result.get("cases")
            or result.get("packed_payload_preserved") is not True):
        raise ValueError("microbenchmark source, checkpoint, cases or payload proof missing")
    if any(result.get("ownership", {}).get(key) != 0 for key in
           ("active_handles", "reserved_bytes", "allocated_bytes")):
        raise ValueError("microbenchmark reservation/handle cleanup incomplete")
    if any(case["status"] == "FAIL" for case in result["cases"]):
        raise ValueError("projection numerical regression")


def worker(manifest_path, destination):
    import mlx.core as mx

    from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
    from mlx_one.models.shared.execution import project
    from mlx_one.p0_engine_worker import Cases

    manifest = json.loads(Path(manifest_path).read_text())
    source, task, native = manifest["source"], manifest["task"], manifest["native"]
    if (source_provenance(Path.cwd()) != source
            or digest(__file__) != manifest["tool_sha256"]):
        raise ValueError("frozen microbenchmark source/environment changed")
    metadata = native["metadata"]
    actual = validate_assets(metadata["model_id"], metadata["snapshot_path"], metadata["revision"])
    if actual["assets"] != metadata["assets"]:
        raise ValueError("microbenchmark checkpoint assets changed")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "heartbeat.json"
    def beat(phase):
        atomic_json(heartbeat, {"status": "running", "phase": phase,
                                "started": time.monotonic(), "timeout": 600})
    beat("load")
    cases = Cases(native, root)
    runtime, model = cases.engine._cache_manager, cases.bundle.model
    def probe(module, value, route, as_linear):
        if route == "reference":
            return project(module, value, execution_dtype="float16", as_linear=as_linear)
        # Deliberately test the stock primitive even where runtime dispatch
        # refuses it. This diagnostic cannot enable an unqualified matrix route.
        output = mx.quantized_matmul(value, module.weight, scales=module.scales,
            biases=module.biases, transpose=True, bits=module.bits,
            group_size=module.group_size, mode="affine")
        return output + module.bias if not as_linear and "bias" in module else output
    result = {"source": source, "checkpoint": metadata["model_id"],
        "revision": metadata["revision"], "cases": [], "status": "running",
        "diagnostic": True, "qualification_promoted": False, "gpu_kernel_count": None,
        "peak_scope": "paired-diagnostic; reference output overlap reported, not a memory win"}
    before = [(module, module.weight) for _, module in model.named_modules()
              if hasattr(module, "bits")]
    try:
        for shape in projection_shapes(model):
            module = shape["module"]
            for rows in task["rows"]:
                case = {k: v for k, v in shape.items() if k != "module"}
                case.update(rows=rows, status="NOT_RUN", measurements={})
                case["candidate_scope"] = "direct stock primitive probe; not runtime selection"
                result["cases"].append(case)
                beat("projection")
                bound = projection_bound(rows, shape["width"], shape["outputs"])
                if bound > runtime.config.memory_budget_bytes:
                    case.update(status="BLOCKED_CAPACITY", reason="declared transient bound")
                    continue
                def execute(bound=bound, rows=rows, shape=shape, module=module, case=case):
                    with runtime.transient_reservation(bound):
                        mx.random.seed(0)
                        value = mx.random.normal((1, rows, shape["width"])).astype(mx.float16)
                        mx.eval(value)
                        outputs = []
                        for route in ("reference", "mlx_qmm"):
                            policy = NativeMathPolicy(projection=route)
                            with math_execution(policy, family=cases.bundle.architecture,
                                                phase="decode" if rows == 1 else "prefill"):
                                output = probe(module, value, route, shape["as_linear"])
                                mx.eval(output)
                                mx.synchronize()
                                timings = []
                                for _ in range(task["repetitions"]):
                                    output = None
                                    mx.synchronize()
                                    mx.reset_peak_memory()
                                    started = time.perf_counter()
                                    output = probe(module, value, route, shape["as_linear"])
                                    mx.eval(output)
                                    mx.synchronize()
                                    timings.append({"seconds": time.perf_counter() - started,
                                        "peak_mlx_bytes": mx.get_peak_memory(),
                                        "retained_reference_output_bytes": sum(
                                            item.nbytes for item in outputs),
                                        "active_mlx_bytes": mx.get_active_memory(),
                                        "cached_mlx_bytes": mx.get_cache_memory()})
                                case["measurements"][route] = timings
                                outputs.append(output)
                        expected, actual = (output.astype(mx.float32) for output in outputs)
                        difference = mx.abs(expected - actual)
                        within = mx.all(difference <= .01 + .01 * mx.abs(expected))
                        maximum, mean = mx.max(difference), mx.mean(difference)
                        mx.eval(within, maximum, mean)
                        return {"within_tolerance": bool(within.item()), "atol": .01, "rtol": .01,
                                "max_absolute_error": float(maximum.item()),
                                "mean_absolute_error": float(mean.item()),
                                "actual_dtype": str(outputs[1].dtype),
                                "actual_output_shape": list(outputs[1].shape)}
                case["parity"] = cases.engine._device.call(id(cases.engine), execute)
                case["status"] = "PASS" if case["parity"]["within_tolerance"] else "FAIL"
                atomic_json(root / "experiment.json", result)
                if case["status"] == "FAIL":
                    raise AssertionError("real packed-projection parity failed; stop expansion")
        result["packed_payload_preserved"] = all(m.weight is payload for m, payload in before)
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
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 64, 128, 512])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("qualification/results/p01_math/micro"))
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if not args.baseline or args.repetitions < 1 or any(r < 1 or r > 8192 for r in args.rows):
        parser.error("pinned --baseline, positive repetitions and rows 1..8192 required")
    assets = json.loads((args.baseline / "download_manifest.json").read_text())
    environment = json.loads((args.baseline / "environment.json").read_text())
    models = [value["metadata"] for value in assets.values() if value["status"] == "PASS"
              and (args.model is None or value["metadata"]["model_id"] in args.model)]
    if not models or args.model and set(args.model) - {m["model_id"] for m in models}:
        parser.error("selected pinned assets unavailable; no model substitution")
    source = source_provenance(Path.cwd())
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    tasks = [{"id": canonical_digest([m["model_id"], args.rows])[:24],
        "checkpoint": m["model_id"], "rows": args.rows, "repetitions": args.repetitions,
        "status": "NOT_RUN"} for m in models]
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
    return int(any(task["status"] not in {"PASS", "NOT_RUN"} for task in tasks))


if __name__ == "__main__":
    raise SystemExit(main())
