"""Offline, isolated P0.1 experiments; importing this module never initializes MLX.

Keep this supervisor outside the runtime source tree so a baseline can be
recorded before any mathematical implementation changes. Every worker verifies
both the runtime digest and this tool's digest before loading a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mlx_one.inference_baseline import process_rss, source_provenance, supervise
from mlx_one.inference_qualification import write_integrity as write_result_integrity
from mlx_one.native_qualification import preflight
from mlx_one.p0_engine_qualification import atomic_json, digest, exact_prompt, utc, validate_assets

VERSION = "native-math-experiment-v1"
WIDTHS = (128, 512, 2048, 4096, 8192)
PHASES = ("load", "tokenization", "attention_qkv", "attention_core", "attention_output",
          "mlp", "logits", "hybrid", "kv_append", "sampling", "synchronization", "host")


def write_integrity(output):
    """Extend existing evidence checksums to the native numerical payloads."""
    root = Path(output)
    write_result_integrity(root)
    path = root / "result-integrity.json"
    checksums = json.loads(path.read_text())
    checksums.update({p.name: digest(p) for p in sorted(root.glob("*.npz")) if p.is_file()})
    atomic_json(path, checksums)


def await_prerequisites(paths, *, timeout=12 * 3600, clock=time.monotonic, sleep=time.sleep):
    until = clock() + timeout
    for path in paths:
        while True:
            state = json.loads(Path(path).read_text())
            if state.get("status") in {"PASS", "completed"}:
                break
            if state.get("status") not in {"running", "QUEUED"}:
                raise RuntimeError(f"prerequisite stopped: {path}: {state.get('status')}")
            if clock() >= until:
                raise TimeoutError("prerequisite execution deadline; no native experiment started")
            sleep(5)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def summarize(values):
    values = list(values)
    if not values:
        return {key: None for key in ("p50", "p95", "min", "max", "stddev")}
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("measurements must be finite and nonnegative")
    ordered = sorted(values)
    return {"p50": statistics.median(values), "p95": ordered[math.ceil(.95 * len(values)) - 1],
            "min": min(values), "max": max(values),
            "stddev": statistics.pstdev(values)}


def abba(pairs):
    """Independent processes, balanced A/B order without pooling profiled timings."""
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ValueError("positive ABBA cycle count required")
    return [route for _ in range(pairs) for route in ("reference", "optimized",
                                                   "optimized", "reference")]


def validate_pair(left, right):
    keys = ("checkpoint", "revision", "assets_sha256", "tokenizer_identity", "prompt_ids",
            "output_fixture", "execution_dtype", "kv_dtype", "source_sha256", "environment",
            "context_limit", "seed", "prefix_reuse", "batch", "profile", "measurement_kind")
    missing = [key for key in keys if key not in left or key not in right]
    if missing:
        raise ValueError("incomplete comparison identity: " + ", ".join(missing))
    changed = [key for key in keys if left[key] != right[key]]
    if changed:
        raise ValueError("unmatched comparison: " + ", ".join(changed))
    if left["profile"]:
        raise ValueError("instrumented runs cannot establish throughput qualification")


def compare(left, right, *, minimum_repetitions=10):
    validate_pair(left["identity"], right["identity"])
    a, b = left["measurements"], right["measurements"]
    pending = (len(a) < minimum_repetitions or len(b) < minimum_repetitions
               or not left.get("correctness_passed") or not right.get("correctness_passed"))
    metrics = {}
    for key in ("prefill_tps", "decode_tps", "peak_mlx_bytes", "rss_peak_bytes"):
        av, bv = [r.get(key) for r in a], [r.get(key) for r in b]
        if not av or not bv or any(v is None for v in av + bv):
            pending = True
            metrics[key] = None
        else:
            am, bm = summarize(av)["p50"], summarize(bv)["p50"]
            metrics[key] = bm / am if am else None
            pending |= am == 0
    return {"status": "PENDING" if pending else "MEASURED", "ratios": metrics,
            "qualification_promoted": False}


def validate_worker_result(result, task, metadata, source):
    identity = result.get("identity", {})
    required = {"checkpoint": metadata["model_id"], "revision": metadata["revision"],
                "source_sha256": source["source_sha256"], "execution_dtype": "float16",
                "kv_dtype": "float16", "batch": 1, "prefix_reuse": False, "profile": False,
                "assets_sha256": canonical_digest(metadata["assets"])}
    if any(identity.get(key) != expected for key, expected in required.items()):
        raise ValueError("worker execution identity mismatch")
    if len(identity.get("prompt_ids", [])) != task["prompt_tokens"]:
        raise ValueError("worker rendered-token count mismatch")
    fixture = identity.get("output_fixture")
    if not isinstance(fixture, list) or len(fixture) > task["output_tokens"]:
        raise ValueError("worker output fixture missing or exceeds budget")
    rows = result.get("measurements", [])
    if len(rows) != task["repetitions"]:
        raise ValueError("incomplete measured repetitions")
    for row in rows:
        timing = row.get("timing", {})
        if (timing.get("token_ids") != fixture or row.get("generated_tokens") != len(fixture)
                or timing.get("actual_prefill_tokens") != task["prompt_tokens"]
                or set(timing.get("batch_sizes", [])) != {1}):
            raise ValueError("worker tokens, skipped work or actual batch changed")
    ownership = result.get("ownership", {})
    if any(ownership.get(key) != 0 for key in ("active_handles", "reserved_bytes",
                                             "allocated_bytes")):
        raise ValueError("worker request cleanup incomplete")


def worker_status(result, observation):
    if observation["status"] == "completed":
        return result.get("status", "BLOCKED")
    if observation["status"] == "worker_failed" and result.get("status") == "FAIL":
        return "FAIL"
    return "BLOCKED"


def experiment_inventory(metadata, widths=WIDTHS, *, independent_runs=2, repetitions=2,
                         routes=("reference",), profile=False, chunk=None, allowed_widths=WIDTHS):
    if independent_runs < 2 or repetitions < 1:
        raise ValueError("at least two independent processes and one repetition required")
    if set(widths) - set(allowed_widths) or not widths:
        raise ValueError("unsupported prompt widths")
    if chunk not in {None, 256, 512, 1024, 2048}:
        raise ValueError("unsupported prefill chunk")
    if set(routes) - {"reference", "optimized"} or not routes:
        raise ValueError("explicit reference/optimized routes required")
    result = []
    for model in metadata:
        for width in widths:
            order = (abba(math.ceil(independent_runs / 2)) if len(routes) == 2 else
                     [routes[0]] * independent_runs)
            for index, route in enumerate(order):
                legal = width + 128 <= model["configured_context"]
                record = {"checkpoint": model["model_id"], "revision": model["revision"],
                          "prompt_tokens": width, "output_tokens": 128, "route": route,
                          "process_replica": index, "repetitions": repetitions,
                          "warmups": 1, "profile": profile, "chunk": chunk,
                          "status": "NOT_RUN" if legal else "NOT_SUPPORTED",
                          "reason": None if legal else "prompt plus output exceeds context"}
                record["id"] = canonical_digest(record)[:24]
                result.append(record)
    return result


def worker(manifest_path, destination):
    # Import native implementations only after parent GPU/RSS/memory preflight.
    from mlx_one.engine.observation import observe_request
    from mlx_one.p0_engine_worker import Cases, Observer, component_state, deadline

    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "heartbeat.json", {"status": "running", "started": time.monotonic(),
                                          "timeout": 120, "phase": "load"})
    if source_provenance(Path.cwd())["source_sha256"] != manifest["source"]["source_sha256"]:
        raise RuntimeError("frozen runtime source changed")
    if digest(__file__) != manifest["tool_sha256"]:
        raise RuntimeError("experiment tooling changed")
    task, native = manifest["task"], manifest["native"]
    metadata = native["metadata"]
    actual = validate_assets(metadata["model_id"], metadata["snapshot_path"], metadata["revision"])
    if actual["assets"] != metadata["assets"]:
        raise ValueError("pinned checkpoint assets changed")
    # The baseline uses the existing Engine verbatim. Optimized construction is
    # supplied by its own implementation only after the baseline has completed.
    if task["route"] != "reference" or task["chunk"] is not None or task["profile"]:
        raise NotImplementedError("optimized/profile routes require the frozen PR-01 runtime")
    cases = Cases(native, root)
    atomic_json(root / "heartbeat.json", {"status": "finished", "phase": "load"})
    result = {"status": "running", "measurements": [], "phase_coverage": {
        phase: "unavailable" for phase in PHASES}, "correctness_passed": False,
        "qualification_promoted": False, "gpu_kernel_count": None}
    try:
        prompt = exact_prompt(cases.bundle.tokenizer, task["prompt_tokens"])
        prompt_ids = list(cases.bundle.tokenizer.encode(prompt))
        identity = {"checkpoint": metadata["model_id"], "revision": metadata["revision"],
            "assets_sha256": canonical_digest(metadata["assets"]),
            "tokenizer_identity": cases.bundle.tokenizer_identity, "prompt_ids": prompt_ids,
            "output_fixture": None, "execution_dtype": "float16", "kv_dtype": "float16",
            "source_sha256": manifest["source"]["source_sha256"],
            "environment": {"hardware": native["environment"]["preflight"]["hardware"],
                            "packages": manifest["source"]["packages"],
                            "software": manifest["source"]["software"]},
            "context_limit": cases.bundle.context_length, "seed": 0,
            "prefix_reuse": False, "batch": 1, "profile": False,
            "measurement_kind": "python_engine_model_token_intervals"}
        result.update(identity=identity, load_seconds=cases.load_seconds)
        atomic_json(root / "prompt.json", {"text": prompt, "token_ids": prompt_ids,
                                          "identity": identity})
        atomic_json(root / "heartbeat.json", {"status": "running",
            "started": time.monotonic(), "timeout": 600, "phase": "warmup"})
        class WarmObserver(Observer):
            def __init__(self):
                super().__init__()
                self.capture_state = True
                self.components = None

            def __call__(self, event):
                if event["kind"] == "state":
                    self.components = component_state(event["cache"])
                    self.capture_state = False
                    return
                super().__call__(event)
        warm = WarmObserver()
        with observe_request(warm), deadline(600):
            reference_value = cases.engine.generate(prompt, max_tokens=128, seed=0)
        reference = {**reference_value.to_dict(), "observer": warm.summary()}
        if not warm.components:
            raise AssertionError("evaluated component telemetry unavailable")
        for component in warm.components:
            if component["position"] != len(prompt_ids):
                raise AssertionError("prefill component position does not equal real inputs")
            for name, tensor in component["fields"].items():
                if name in {"keys", "values", "_keys", "_values"} and len(tensor["shape"]) == 4:
                    if tensor["dtype"] != "float16":
                        raise AssertionError("observed attention KV is not FP16")
        result["state_components"] = warm.components
        atomic_json(root / "heartbeat.json", {"status": "finished", "phase": "warmup"})
        identity["output_fixture"] = reference["observer"]["token_ids"]
        atomic_json(root / "prompt.json", {"text": prompt, "token_ids": prompt_ids,
                                          "identity": identity})
        result["workload_sha256"] = canonical_digest(identity)
        result["phase_coverage"].update(load="observed", tokenization="observed",
                                        synchronization="observed")
        for repetition in range(task["repetitions"]):
            heartbeat = {"status": "running", "started": time.monotonic(), "timeout": 600}
            atomic_json(root / "heartbeat.json", heartbeat)
            cases.mx.synchronize()
            cases.mx.reset_peak_memory()
            started = time.monotonic()
            value, observer = cases.generate(prompt, max_tokens=128, seed=0)
            sync_started = time.perf_counter()
            cases.mx.synchronize()
            sync_ms = (time.perf_counter() - sync_started) * 1000
            finished = time.monotonic()
            timing = observer.summary()
            if timing["token_ids"] != identity["output_fixture"]:
                raise AssertionError("independent fresh-state greedy tokens changed")
            if timing["actual_prefill_tokens"] != len(prompt_ids):
                raise AssertionError("baseline unexpectedly reused a prefix")
            assert set(timing["batch_sizes"]) == {1}
            row = {"repetition": repetition, "started": started, "finished": finished,
                   "generated_tokens": value["generation_tokens"],
                   "prefill_tps": len(prompt_ids) / (timing["forward_ms"]["prefill"] / 1000),
                   "decode_tps": timing["decode_tokens_per_second"],
                   "decode_tps_definition": "accepted intervals / first-to-last model token time",
                   "synchronized_decode_forward_ms": timing["forward_ms"]["decode"],
                   "engine_wall_seconds": value["engine_wall_seconds"],
                   "synchronization_ms": sync_ms, "peak_mlx_bytes": cases.mx.get_peak_memory(),
                   "active_mlx_bytes": cases.mx.get_active_memory(),
                   "cached_mlx_bytes": cases.mx.get_cache_memory(), "rss_peak_bytes": None,
                   "gpu_kernel_count": None, "timing": timing}
            result["measurements"].append(row)
            stats = cases.engine._cache_manager.stats()
            assert stats.active_handles == stats.reserved_bytes == stats.allocated_bytes == 0
            result["ownership"] = stats.to_dict()
            heartbeat.update(status="finished", finished=finished)
            atomic_json(root / "heartbeat.json", heartbeat)
            atomic_json(root / "experiment.json", result)
        # State parity belongs to independently reconstructed native execution;
        # repeated token agreement is deliberately not labeled correctness_passed.
        result.update(status="PASS",
                      correctness_reason="independent numerical gate not run in timing worker")
    except BaseException as error:
        result.update(status="FAIL", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        cases.close()
        atomic_json(root / "experiment.json", result)
        write_integrity(root)


def run_task(root, task, model, environment, source, observed, *, worker_entry=None,
             validator=validate_worker_result, tool_sources=None, schema_version=VERSION):
    directory = root / task["id"]
    directory.mkdir()
    entry = Path(worker_entry or __file__).resolve()
    manifest = {"schema_version": schema_version, "task": task, "source": source,
                "tool_sha256": digest(entry), "tool_sources": tool_sources,
                "native": {"metadata": model,
                "environment": {**environment, "source": source, "preflight": observed}}}
    atomic_json(directory / "manifest.json", manifest)
    command = [sys.executable, str(entry), "--worker",
               str(directory / "manifest.json"), str(directory / "data")]
    atomic_json(directory / "command.json", {"argv": command, "timestamp": utc()})
    timed_out, origin = [], []
    def clock():
        value = time.monotonic()
        if not origin:
            origin.append(value)
        return value
    with (directory / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                 "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()),
                                                os.environ.get("PYTHONPATH", ""))),
                 **({"MTL_CAPTURE_ENABLED": "1"} if task.get("capture") else {})})
        def rss(pid):
            value = process_rss(pid)
            heartbeat = directory / "data/heartbeat.json"
            if heartbeat.exists():
                state = json.loads(heartbeat.read_text())
                if state["status"] == "running" and time.monotonic() - state["started"] > min(
                        state.get("timeout", 120), 600):
                    timed_out.append(state)
                    process.terminate()
            return value
        try:
            timeout = task.get("supervision_timeout_seconds",
                               120 + 600 * (task["repetitions"] + 1))
            if type(timeout) not in {int, float} or not 0 < timeout <= 8 * 3600:
                raise ValueError("invalid bounded worker supervision deadline")
            observation = supervise(process, timeout=timeout,
                                    memory_limit=observed["process_ceiling_bytes"], rss=rss,
                                    clock=clock)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if timed_out:
        observation["status"] = "request_timeout"
    observation["monotonic_origin"] = origin[0] if origin else None
    observation["monotonic_finished"] = time.monotonic()
    atomic_json(directory / "supervision.json", observation)
    path = directory / "data/experiment.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for row in result.get("measurements", []):
        samples = [s["rss_bytes"] for s in observation["rss_samples"] if origin
                   and row["started"] <= origin[0] + s["elapsed_seconds"] <= row["finished"]
                   and s.get("rss_bytes") is not None]
        row["rss_peak_bytes"] = max(samples) if samples else None
        row["rss_peak_scope"] = "supervised-request-samples" if samples else "unavailable"
    if path.exists():
        atomic_json(path, result)
        write_integrity(path.parent)
    status = worker_status(result, observation)
    if status == "PASS":
        try:
            validator(result, task, model, source)
        except ValueError as error:
            status = "BLOCKED"
            result.update(status=status, reason=str(error))
            atomic_json(path, result)
            write_integrity(path.parent)
    write_integrity(directory)
    return {**task, "status": status, "artifact": str(directory),
            "supervision": observation["status"],
            "reason": result.get("reason", task.get("reason"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, default=Path("qualification/results/p01_math"))
    parser.add_argument("--model", action="append")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", choices=WIDTHS, default=WIDTHS)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--independent-runs", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--after", type=Path, action="append", default=[],
                        help="Wait for these existing native cohorts before starting GPU work")
    parser.add_argument("--worker", nargs=2)
    args = parser.parse_args(argv)
    if args.worker:
        worker(*args.worker)
        return 0
    if not args.baseline:
        parser.error("--baseline with immutable pinned checkpoint inventory is required")
    environment = json.loads((args.baseline / "environment.json").read_text())
    downloads = json.loads((args.baseline / "download_manifest.json").read_text())
    models = [r["metadata"] for r in downloads.values() if r["status"] == "PASS"
              and (args.model is None or r["metadata"]["model_id"] in args.model)]
    if not models or args.model and set(args.model) - {m["model_id"] for m in models}:
        parser.error("requested model lacks a complete pinned offline checkpoint")
    tasks = experiment_inventory(models, args.prompt_tokens, repetitions=args.repetitions,
                                 independent_runs=args.independent_runs)
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True)
    source = source_provenance(Path.cwd())
    manifest = {"schema_version": VERSION, "timestamp": utc(), "source": source,
                "tool_sha256": digest(__file__), "tasks": tasks, "qualification_promoted": False,
                "status": "QUEUED" if args.run else "NOT_RUN",
                "baseline_noise_limit_fraction": .05,
                "performance_qualification_repetitions": 10,
                "prerequisites": [str(path.resolve()) for path in args.after]}
    (root / "runner-source.txt").write_text(Path(__file__).read_text())
    atomic_json(root / "manifest.json", manifest)
    write_integrity(root)
    print(root, flush=True)
    if not args.run:
        return 0
    try:
        await_prerequisites(args.after)
        if source_provenance(Path.cwd())["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("frozen experiment source changed")
        observed = preflight(root / "preflight")
        manifest.update(status="running", preflight=observed)
        for index, task in enumerate(tasks):
            if task["status"] != "NOT_RUN":
                continue
            if source_provenance(Path.cwd())["source_sha256"] != source["source_sha256"]:
                raise RuntimeError("frozen experiment source changed")
            model = next(m for m in models if m["model_id"] == task["checkpoint"])
            tasks[index] = run_task(root, task, model, environment, source, observed)
            atomic_json(root / "manifest.json", manifest)
            write_integrity(root)
            print(json.dumps(tasks[index]), flush=True)
            if tasks[index]["supervision"] != "completed":
                raise RuntimeError("native supervision failed; subsequent experiments stopped")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="BLOCKED", reason=str(error), error_type=type(error).__name__)
        raise
    finally:
        atomic_json(root / "manifest.json", manifest)
        write_integrity(root)
    return int(any(t["status"] not in {"PASS", "NOT_SUPPORTED"} for t in tasks))


if __name__ == "__main__":
    raise SystemExit(main())
