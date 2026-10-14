"""Staged native P1 qualification using existing offline workers and evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from mlx_one.inference_baseline import run_profile, source_provenance, supervise
from mlx_one.inference_qualification import export_matrix, validate_integrity, write_integrity
from mlx_one.p1_profiles import CHECKPOINTS, profile

BASE_SHA = "b698f88afc1729cd7d2dea53fd5a69f11b60246e"
PLAN_VERSION = "native-p1-command-manifest-v1"
CORRECTNESS_VERSION = "native-p1-correctness-v1"


def qualification_source():
    source = source_provenance(Path.cwd())
    source["native_test_sha256"] = hashlib.sha256(
        Path("tests/test_p1_checkpoint_parity.py").read_bytes()).hexdigest()
    return source


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class NativeCase:
    checkpoint: str
    mode: str
    prompt_tokens: int
    output_tokens: int
    slots: int
    chunk_tokens: int
    page_tokens: int = 32
    prompt_class: str = "literal"

    @property
    def key(self):
        payload = json.dumps(vars(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    def reference(self, *, cooperative=False):
        return replace(self, mode="reference_dense", slots=self.slots if cooperative else 1,
                       page_tokens=32)

    def optimized_b1(self):
        return replace(self, slots=1)


def cases(checkpoints=tuple(CHECKPOINTS)):
    """Full requested legal cross-product; SmolLM2 uses a labeled legal substitute."""
    for key in checkpoints:
        asset = CHECKPOINTS[key]
        dense = "experimental_hybrid_batch" if asset.architecture == "lfm2" else (
            "experimental_dense_batch")
        for prompt, output, slots, chunk in itertools.product(
                (128, 512, 2048, 8192), (32, 128, 256), (1, 2, 4), (128, 256, 512, 1024)):
            label = "literal"
            if prompt + output > asset.context_tokens:
                prompt, label = asset.context_tokens - output, "maximum_legal_8k"
            for mode, page in ((dense, 32), ("paged", 16), ("paged", 32)):
                yield NativeCase(key, mode, prompt, output, slots, chunk, page, label)


def settings(case):
    return {"mode": case.mode, "prompt_tokens": case.prompt_tokens,
        "output_tokens": case.output_tokens, "slots": case.slots,
        "chunk_tokens": case.chunk_tokens, "page_tokens": case.page_tokens,
        "queue_size": 32, "max_scheduled_tokens": 2048, "max_prefill_tokens": 1024,
        "ttft_budget_ms": 2000 if case.prompt_tokens <= 2048 else 10000,
        "decode_gap_budget_ms": 100, "max_queue_wait_ms": 5000, "max_service_gap_ms": 250}


def comparison_traffic(case):
    """Identical twelve requests for serial, cooperative and optimized workers."""
    from mlx_one.traffic_replay import validate_traffic

    return validate_traffic({"schema_version": "native-traffic-v1", "seed": 0,
        "requests": [{"request_id": f"request-{index:04}", "arrival_ms": index * 20,
            "prompt_tokens": case.prompt_tokens, "output_tokens": case.output_tokens,
            "principal_label": "principal-a", "consumer": {}, "shared_prefix_fraction": 0}
            for index in range(12)]})


def command_manifest(checkpoints=tuple(CHECKPOINTS)):
    optimized = list(cases(checkpoints))
    controls = {}
    for case in optimized:
        for control in (case.reference(), case.reference(cooperative=True)):
            controls[control.key] = control
    records = []
    for case in [*controls.values(), *optimized]:
        dependencies = []
        if case.mode != "reference_dense":
            dependencies = [case.reference().key, case.reference(cooperative=True).key]
            if case.slots > 1:
                dependencies.append(case.optimized_b1().key)
        records.append({"id": case.key, "case": vars(case), "settings": settings(case),
                        "dependencies": list(dict.fromkeys(dependencies)), "status": "NOT_RUN"})
    rejected = [{"checkpoint": key, "prompt_tokens": 8192, "output_tokens": output,
                 "status": "REJECTED", "reason": "prompt plus output exceeds 8192 tokens"}
                for key in checkpoints if CHECKPOINTS[key].context_tokens == 8192
                for output in (32, 128, 256)]
    return {"schema_version": PLAN_VERSION, "base_sha": BASE_SHA,
        "optimized_configurations": len(optimized), "tasks": records,
        "rejected_literal_contexts": rejected,
        "repetitions": 5, "warmup_requests": 1, "workers_at_once": 1,
        "qualification_promoted": False}


def supervised_command(command, output, *, ceiling, timeout=600, env=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "command.json", {"argv": command, "source": source_provenance(Path.cwd())})
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})})
        result = supervise(process, timeout=timeout, memory_limit=ceiling)
    result["worker_log_sha256"] = hashlib.sha256((output / "worker.log").read_bytes()).hexdigest()
    write(output / "supervision.json", result)
    write_integrity(output)
    return result


def preflight(output, *, configured_ceiling=None):
    """Actually execute GPU work and supervise a live child before loading weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        write(output / "preflight.json", {"status": "BLOCKED", "reason": str(exc)})
        write_integrity(output)
        raise RuntimeError("native host telemetry is unavailable") from exc
    if chip != "Apple M4" or ram != 32 * 1024**3:
        write(output / "preflight.json", {"status": "BLOCKED",
            "hardware": {"chip": chip, "memory_bytes": ram}, "reason": "declared host mismatch"})
        write_integrity(output)
        raise RuntimeError("qualification requires the declared Apple M4 / 32 GiB host")
    if configured_ceiling is not None and (
            not isinstance(configured_ceiling, int) or isinstance(configured_ceiling, bool)
            or configured_ceiling < 1):
        raise ValueError("configured process ceiling must be a positive integer")
    smoke_ceiling = min(int(ram * .75), configured_ceiling or int(ram * .75))
    command = [sys.executable, "-c", '''import json,time
import mlx.core as mx
mx.set_default_device(mx.gpu)
x=mx.arange(16,dtype=mx.float32); y=(x*x).sum()
mx.eval(y); mx.synchronize()
print(json.dumps({"device":str(mx.default_device()),"metal":mx.metal.is_available(),
 "result":y.item(),"active_bytes":mx.get_active_memory(),"cached_bytes":mx.get_cache_memory(),
 "peak_bytes":mx.get_peak_memory(),"device_info":mx.device_info()}),flush=True)
time.sleep(.5)''']
    result = supervised_command(command, output / "gpu-smoke", ceiling=smoke_ceiling)
    observed = {}
    for line in (output / "gpu-smoke/worker.log").read_text().splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "result" in value:
                observed = value
        except ValueError:
            continue
    good = (result["status"] == "completed" and observed.get("metal") is True
            and observed.get("result") == 1240 and "gpu" in observed.get("device", "")
            and all(isinstance(observed.get(key), int) and observed[key] >= 0
                    for key in ("active_bytes", "cached_bytes", "peak_bytes"))
            and bool(result["rss_samples"])
            and all(s["rss_bytes"] is not None for s in result["rss_samples"]))
    recommended = observed.get("device_info", {}).get("max_recommended_working_set_size")
    ceiling = min(smoke_ceiling, recommended) if isinstance(recommended, int) and (
        recommended > 0) else smoke_ceiling
    report = {"status": "PASS" if good else "BLOCKED", "hardware": {"chip": chip,
        "memory_bytes": ram}, "gpu_smoke": observed, "source": qualification_source(),
        "disk_free_bytes": shutil.disk_usage(output).free, "process_ceiling_bytes": ceiling}
    write(output / "preflight.json", report)
    write_integrity(output)
    if not good:
        raise RuntimeError("native Metal/RSS preflight failed")
    return report


def validate_correctness(path, checkpoints, *, source, hardware):
    """Do not accept a hand-written PASS, stale source, skipped tests or changed logs."""
    path = Path(path).resolve()
    validate_integrity(path.parent, required={path.name})
    proof = json.loads(path.read_text())
    if (proof.get("schema_version") != CORRECTNESS_VERSION
            or proof.get("source", {}).get("source_sha256") != source["source_sha256"]
            or proof.get("source", {}).get("native_test_sha256") != source["native_test_sha256"]
            or proof.get("hardware") != hardware):
        raise ValueError("correctness evidence schema, source or hardware mismatch")
    for checkpoint in checkpoints:
        for paged in (False, True):
            row = proof.get("outcomes", {}).get(f"{checkpoint}:{paged}", {})
            if row.get("status") != "PASS":
                raise ValueError("checkpoint correctness is incomplete")
            directory = Path(row["path"])
            validate_integrity(directory, required={"supervision.json", "command.json"})
            supervision = json.loads((directory / "supervision.json").read_text())
            log = (directory / "worker.log").read_bytes()
            if (supervision.get("status") != "completed"
                    or supervision.get("worker_log_sha256") != hashlib.sha256(log).hexdigest()
                    or not re.search(rb"6 passed", log) or re.search(rb"\d+ skipped", log)):
                raise ValueError("native correctness log is missing, changed or incomplete")
            required = {f"parity-{checkpoint}-{paged}-{page}-{size}.json"
                        for page in (16, 32) for size in (1, 2, 4)}
            validate_integrity(directory, required=required)
            for name in required:
                parity = json.loads((directory / name).read_text())
                if (parity.get("status") != "passed"
                        or parity.get("greedy_agreement_fraction") != 1
                        or parity.get("cleanup", {}).get("active_handles") != 0
                        or parity.get("cleanup", {}).get("reserved_bytes") != 0):
                    raise ValueError("native parity/state artifacts are incomplete")
    return proof


def correctness_matrix(outcomes, source):
    rows = []
    for checkpoint, asset in CHECKPOINTS.items():
        dense = "experimental_hybrid_batch" if asset.architecture == "lfm2" else (
            "experimental_dense_batch")
        for paged, mode in ((False, dense), (True, "paged")):
            row = outcomes.get(f"{checkpoint}:{paged}")
            prerequisite = outcomes.get(f"{checkpoint}:False", {})
            status = row["status"] if row else "BLOCKED" if paged and (
                prerequisite.get("status") == "FAIL") else "NOT_RUN"
            rows.append({"checkpoint": checkpoint, "mode": mode, "status": status,
                "source_sha256": source["source_sha256"],
                "artifact": row.get("path") if row else None,
                "production_qualified": False, "reason": "dense correctness failed" if (
                    status == "BLOCKED") else "native measurements required"})
    return rows


def execute_manifest(manifest, output, hardware, *, checkpoint_paths=None, runner=run_profile,
                     artifact_root=None, memory_ceiling=None):
    """Dependency ordering and fail-closed correctness admission, with durable outcomes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = checkpoint_paths or {}
    source = source_provenance(Path.cwd())
    source_digest = source["source_sha256"]
    outcomes = {}
    tasks = {task["id"]: task for task in manifest["tasks"]}
    visiting = set()
    def execute(identity):
        if identity in outcomes:
            return outcomes[identity]
        if identity in visiting:
            raise ValueError("qualification dependency cycle")
        task = tasks[identity]
        visiting.add(identity)
        prerequisites = [execute(dep) for dep in task["dependencies"]]
        visiting.remove(identity)
        if any(item["status"] != "PASS" for item in prerequisites):
            outcomes[identity] = {"status": "BLOCKED", "reason": "comparison run unavailable"}
        elif source_provenance(Path.cwd())["source_sha256"] != source_digest:
            raise RuntimeError("source changed during the measurement cohort")
        else:
            case = NativeCase(**task["case"])
            options = {**task["settings"], "checkpoint_path": checkpoint_paths.get(case.checkpoint)}
            if memory_ceiling is not None:
                options["memory_fraction"] = min(.75, memory_ceiling / hardware["memory_bytes"])
            from mlx_one.traffic_replay import traffic_identity

            traffic = comparison_traffic(case)
            traffic_path = output / "traffic" / (traffic_identity(traffic) + ".json")
            write(traffic_path, traffic)
            options.update(traffic_manifest=str(traffic_path.resolve()),
                           traffic_identity=traffic_identity(traffic))
            if case.mode != "reference_dense":
                options["serial_run"] = outcomes[case.reference().key]["path"]
                options["reference_run"] = outcomes[case.reference(cooperative=True).key]["path"]
                if case.slots > 1:
                    options["b1_run"] = outcomes[case.optimized_b1().key]["path"]
            directory = (Path(artifact_root) / case.checkpoint / output.name / identity
                         if artifact_root else output / case.checkpoint / identity)
            try:
                item = profile(case.checkpoint, **options)
                result = runner(item, directory, hardware=hardware, provenance=source)
            except (OSError, ValueError, RuntimeError) as exc:
                directory.mkdir(parents=True, exist_ok=True)
                result = {"status": "worker_failed", "reason": f"{type(exc).__name__}: {exc}"}
                write(directory / "failure.json", result)
                write_integrity(directory)
            outcomes[identity] = {"status": "PASS" if result["status"] == "completed" else "FAIL",
                                  "path": str(directory.resolve()),
                                  "worker_status": result["status"]}
            # Completed workers are measurements, not qualification passes.
            gate = directory / "gate-p1.json"
            if gate.exists():
                outcomes[identity]["qualification_status"] = json.loads(gate.read_text())["status"]
            if result["status"] in {"memory_watchdog_unavailable", "memory_watchdog_terminated"}:
                write(output / "outcomes.json", outcomes)
                raise RuntimeError("native supervision failed; qualification stopped")
        write(output / "outcomes.json", outcomes)
        return outcomes[identity]
    for task in manifest["tasks"]:
        execute(task["id"])
    export_matrix(output, output / "matrix")
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "manifest", "preflight", "correctness", "performance", "report",
        "inventory", "prefix-diagnostic"))
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/p1"))
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument("--checkpoint-paths", type=Path)
    parser.add_argument("--process-ceiling-bytes", type=int,
                        help="Lower the PR-00 native diagnostic watchdog ceiling")
    args = parser.parse_args(argv)
    stamp = str(time.time_ns())
    if args.stage in {"inventory", "prefix-diagnostic"}:
        from mlx_one.runtime_baseline import run_baseline

        if args.process_ceiling_bytes is not None and args.process_ceiling_bytes < 1:
            parser.error("process ceiling must be positive")
        root = args.output / socket.gethostname() / "runtime-baseline" / stamp
        result = run_baseline(root, native=args.stage == "prefix-diagnostic",
            configured_ceiling=args.process_ceiling_bytes,
            worker_output=args.output / socket.gethostname() / "qwen3-small" / stamp
                / "prefix-diagnostic")
        print(root)
        return result
    root = args.output / socket.gethostname() / "qualification" / stamp
    if args.stage == "report":
        export_matrix(args.output, args.output / "matrix")
        return 0
    root.mkdir(parents=True)
    manifest = command_manifest(args.checkpoints)
    write(root / "command-manifest.json", manifest)
    if args.stage == "manifest":
        print(root)
        return 0
    observed = preflight(root / "preflight")
    from mlx_one.p1_profiles import asset_inventory

    write(root / "assets.json", asset_inventory())
    paths = json.loads(args.checkpoint_paths.read_text()) if args.checkpoint_paths else {}
    if args.stage == "correctness":
        outcomes = {}
        proof = {"schema_version": CORRECTNESS_VERSION, "source": observed["source"],
                 "hardware": observed["hardware"], "outcomes": outcomes,
                 "qualification_promoted": False}
        for checkpoint in args.checkpoints:
            for paged in (False, True):
                directory = (args.output / socket.gethostname() / checkpoint / stamp
                             / ("paged" if paged else "dense"))
                env = {"MLX_ONE_RUN_P1_CHECKPOINTS": "1", "HF_HUB_OFFLINE": "1",
                       "MLX_ONE_P1_RESULTS_DIR": str(directory.resolve())}
                if checkpoint in paths:
                    env["MLX_ONE_P1_" + checkpoint.upper().replace("-", "_") + "_PATH"] = paths[
                        checkpoint]
                command = [sys.executable, "-m", "pytest", "--no-cov", "-q",
                    "tests/test_p1_checkpoint_parity.py", "-k", f"{checkpoint} and {paged}", "-x"]
                result = supervised_command(command, directory,
                    ceiling=observed["process_ceiling_bytes"], env=env)
                log = (directory / "worker.log").read_text()
                passed = result["status"] == "completed" and re.search(r"6 passed", log)
                passed = bool(passed and not re.search(r"\d+ skipped", log))
                outcomes[f"{checkpoint}:{paged}"] = {"status": "PASS" if passed else "FAIL",
                                                     "path": str(directory)}
                write(root / "correctness.json", proof)
                write(root / "cohort-matrix.json", correctness_matrix(outcomes, observed["source"]))
                write_integrity(root)
                if not passed:
                    print(root)
                    return 1
        print(root)
        return 0
    if args.stage == "performance":
        if args.correctness_report is None:
            parser.error("performance requires --correctness-report from native tests")
        validate_correctness(args.correctness_report, args.checkpoints,
                             source=observed["source"], hardware=observed["hardware"])
        execute_manifest(manifest, root, observed["hardware"], checkpoint_paths=paths,
                         artifact_root=args.output / socket.gethostname(),
                         memory_ceiling=observed["process_ceiling_bytes"])
    print(root)
    return 0
