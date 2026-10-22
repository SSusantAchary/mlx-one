#!/usr/bin/env python3
"""Offline M4/32GB handoff, reusing the native P1 workers and admission gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlx_one.engine.kernel_dispatch import sdpa_capability  # noqa: E402
from mlx_one.inference_qualification import validate_integrity, write_integrity  # noqa: E402
from mlx_one.native_correctness import checkpoint_proof  # noqa: E402
from mlx_one.native_qualification import (  # noqa: E402
    preflight,
    qualification_source,
    supervised_command,
    write,
)
from mlx_one.p1_profiles import CHECKPOINTS  # noqa: E402


def collect_results(root):
    """Export observed worker summaries; absent metrics remain null, never inferred."""
    rows = []
    for path in sorted(root.rglob("summary.json")):
        data = json.loads(path.read_text())
        profile = data.get("profile", {})
        requests_path = path.parent / "requests.jsonl"
        requests = [json.loads(line) for line in requests_path.read_text().splitlines()] if (
            requests_path.exists()) else []
        warm = [row for row in requests if row.get("temperature_phase") == "warm"
                and row.get("status") == "completed"]
        def median_metric(key, warm=warm):
            values = [row.get("metrics", {}).get(key) for row in warm]
            return statistics.median(values) if values and all(
                isinstance(value, (int, float)) for value in values) else None
        supervision_path = path.parent / "supervision.json"
        supervision = json.loads(supervision_path.read_text()) if supervision_path.exists() else {}
        trace_path = path.parent / "forward-traces.jsonl"
        traces = [json.loads(line) for line in trace_path.read_text().splitlines()] if (
            trace_path.exists()) else []
        dispatch_path = path.parent / "math-dispatch.jsonl"
        dispatches = [json.loads(line) for line in dispatch_path.read_text().splitlines()] if (
            dispatch_path.exists()) else []
        attention = [event for event in dispatches if event.get("operation") == "attention"]
        rows.append({"artifact": str(path.resolve()), "model": profile.get("model_id"),
            "backend": profile.get("inference_mode"), "batch": profile.get("slots"),
            "actual_attention_dispatch": {selected: sum(event["selected"] == selected
                for event in attention) for selected in sorted({e["selected"] for e in attention})},
            "completed_requests": data.get("completed"),
            "actual_forward_batch_sizes": sorted({size for request in requests for size in
                request.get("metrics", {}).get("model_forward_batch_sizes", [])}),
            "kv_head_repeat_bytes": sum(e["shape"]["bytes"] for e in dispatches
                if e.get("operation") == "kv_head_repeat") if dispatches else None,
            "context": profile.get("context_tokens"),
            "actual_prompt_max_tokens": data.get("observed_prompt_max_tokens"),
            "ttft_p95_ms": data.get("warm_client_ttft_p95_ms"),
            "prefill_tps_median": median_metric("prompt_tokens_per_second"),
            "per_request_decode_tps_median": median_metric("decode_tokens_per_second"),
            "aggregate_tps_by_group": data.get("aggregate_output_tokens_per_second_by_group"),
            "tpot_p50_ms": data.get("model_token_gap_p50_ms"),
            "tpot_p95_ms": data.get("model_token_gap_p95_ms"),
            "peak_mlx_bytes": data.get("measured_peak_active_bytes"),
            "peak_rss_bytes": supervision.get("process_peak_rss_bytes"),
            "peak_cache_bytes": data.get("peak_cache_bytes"),
            "peak_physical_kv_bytes": data.get("peak_physical_bytes"),
            "history_pack_bytes": sum(t["history_pack_bytes"] for t in traces) if traces and all(
                "history_pack_bytes" in t for t in traces) else None,
            "history_scatter_bytes": sum(t["history_scatter_bytes"] for t in traces) if (
                traces and all("history_scatter_bytes" in t for t in traces)) else None,
            "copy_scope": "all recorded forward traces",
            "failed_requests": data.get("failed"),
            "decode_rates": data.get("controlled_decode", {}).get("rates")})
    for path in sorted(root.rglob("fused-results.json")):
        data = json.loads(path.read_text())
        for case in data.get("cases", []):
            rows.append({"artifact": str(path.resolve()), "test": case.get("test"),
                "model": case.get("checkpoint", {}).get("model_id"),
                "backend": "forced_fused_diagnostic", "status": case.get("status"),
                "max_error": case.get("max_error"), "mean_error": case.get("mean_error"),
                "reference_seconds": case.get("reference_seconds"),
                "fused_seconds": case.get("fused_seconds"),
                "latency_speedup": case.get("latency_speedup")})
    write(root / "measurements.json", {"rows": rows, "qualification_promoted": False})
    with (root / "measurements.csv").open("w", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["artifact"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
    return rows


def validate_fused_report(path, *, source, scope, paths):
    path = Path(path)
    validate_integrity(path.parent, required={
        path.name, "junit.xml", "command.json", "supervision.json", "worker.log"})
    report = json.loads(path.read_text())
    cases = report.get("cases", [])
    tests = list(ET.parse(path.parent / "junit.xml").getroot().iter("testcase"))
    checkpoint_only = scope.get("checkpoint_only", False)
    expected = fused_test_names(scope["checkpoints"], checkpoint_only=checkpoint_only)
    command = json.loads((path.parent / "command.json").read_text())
    supervision = json.loads((path.parent / "supervision.json").read_text())
    argv = command.get("argv", [])
    command_source = {key: value for key, value in source.items() if key != "native_test_sha256"}
    expected_argv = fused_command(path.parent, argv[0], kernel_only=not scope["checkpoints"],
                                  checkpoint_only=checkpoint_only) if (
        argv) else []
    if (not argv or argv != expected_argv
            or command.get("source") != command_source
            or supervision.get("status") != "completed"
            or supervision.get("worker_log_sha256") != hashlib.sha256(
                (path.parent / "worker.log").read_bytes()).hexdigest()):
        raise ValueError("fused native command, source or supervision mismatch")
    if (report.get("schema_version") != "native-fused-diagnostic-v2"
            or report.get("source") != source or report.get("scope") != scope
            or report.get("status") not in {"PASS", "COMPLETE_WITH_UNSUPPORTED"}
            or report.get("force_fused") is not True
            or report.get("qualification_promoted") is not False
            or len(cases) != len(expected) or len(tests) != len(expected)
            or any(row.get("status") not in {"PASS", "UNSUPPORTED"} for row in cases)
            or any(list(test) for test in tests)
            or {row["test"].split("::")[-1] for row in cases} != expected
            or {test.get("name") for test in tests} != expected):
        raise ValueError("fused correctness source/scope/test coverage mismatch or failure")
    for row in cases:
        if row["status"] != "UNSUPPORTED":
            continue
        decision = sdpa_capability(q_shape=tuple(row.get("q_shape", ())),
            k_shape=tuple(row.get("k_shape", ())), v_shape=tuple(row.get("k_shape", ())),
            mlx_version=tuple(int(p) for p in
                              source.get("packages", {}).get("mlx", "0").split(".")))
        results = row.get("results", {})
        if (decision.selected != "reference" or row.get("reason") != decision.reason
                or row.get("selected_fallback") != "reference"
                or row.get("fallback_result", {}).get("within_tolerance") is not True
                or set(results) != {"native", "float32"}
                or any(r.get("status") != "UNSUPPORTED" or r.get("actual_dispatch") !=
                       "rejected_before_kernel" or "force_fused=True but no fused kernel"
                       not in r.get("exception", "") for r in results.values())):
            raise ValueError("unsupported shape lacks capability/rejection/reference evidence")
    for checkpoint in scope["checkpoints"]:
        asset = checkpoint_proof(checkpoint, paths[checkpoint])
        rows = [row for row in cases if row.get("checkpoint", {}).get("checkpoint") == checkpoint]
        if (len(rows) != 1 or rows[0]["checkpoint"] != asset
                or rows[0].get("fused_calls", 0) < 1
                or rows[0].get("checkpoint_unchanged") is not True):
            raise ValueError("fused checkpoint files or actual dispatch changed")
    return report


def fused_test_names(checkpoints, *, checkpoint_only=False):
    import itertools

    names = {"test_fused_kernel_smoke"}
    for mask, (queries, history), dtype, (heads, kv), batch in itertools.product(
            ("causal", "boolean", "additive"), ((1, 0), (1, 17), (7, 19), (32, 0)),
            ("float16", "bfloat16"), ((8, 8), (8, 2), (8, 1)), (1, 2, 4)):
        names.add("test_fused_kernel_identical_operands"
                  f"[{mask}-{queries}-{history}-{dtype}-{heads}-{kv}-{batch}]")
    names.update(f"test_fused_kernel_long_prefill[{dim}-{context}]"
                 for dim, context in itertools.product((64, 128), (512, 2048, 4096)))
    names.update(f"test_fused_checkpoint_logits_state_and_continuation[{key}]"
                 for key in checkpoints)
    if checkpoint_only:
        names = {name for name in names if "checkpoint" in name}
    return names


def fused_command(worker, executable, *, kernel_only=False, checkpoint_only=False):
    command = [executable, "-m", "pytest", "--no-cov", "-q",
               f"--junitxml={Path(worker) / 'junit.xml'}",
               "tests/optimization/test_fused_attention_native.py"]
    return command + (["-k", "not checkpoint"] if kernel_only else
                      ["-k", "checkpoint"] if checkpoint_only else [])


def fused_stage(stage, output, args, *, source, environment, correctness):
    """Diagnostic track is deliberately separate from production correctness receipts."""
    output.mkdir(parents=True, exist_ok=True)
    observed = preflight(output / "preflight", configured_ceiling=args.process_ceiling_bytes)
    scope = {"checkpoints": [] if args.kernel_only else args.checkpoints,
             "hardware": observed["hardware"], "attention_backend": "forced_fused_diagnostic"}
    if args.checkpoint_only:
        scope["checkpoint_only"] = True
    paths = json.loads(args.checkpoint_paths.read_text()) if args.checkpoint_paths else {}
    if stage == "performance":
        validate_fused_report(correctness, source=source, scope=scope, paths=paths)
    worker = output / "worker"
    command = fused_command(worker, sys.executable,
                            kernel_only=args.kernel_only and stage != "smoke",
                            checkpoint_only=args.checkpoint_only and stage != "smoke")
    if stage == "smoke":
        command += ["-k", "test_fused_kernel_smoke"]
    env = {**environment, "MLX_ONE_RUN_FUSED_TESTS": "1",
           "MLX_ONE_FUSED_RESULTS_DIR": str(worker), "MLX_ONE_FUSED_PATHS": json.dumps(paths),
           "MLX_ONE_FUSED_KEYS": json.dumps(scope["checkpoints"]),
           "MLX_ONE_FUSED_SCOPE": json.dumps(scope),
           "MLX_ONE_FUSED_PERFORMANCE": "1" if stage == "performance" else "0"}
    result = supervised_command(command, worker, ceiling=observed["process_ceiling_bytes"],
                                timeout=args.timeout_seconds, env=env)
    report_path = worker / "fused-results.json"
    if result["status"] != "completed":
        raise RuntimeError(f"fused {stage} failed: {result['status']}; inspect {worker}")
    if stage != "smoke":
        validate_fused_report(report_path, source=source, scope=scope, paths=paths)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",
                        choices=("identity", "smoke", "correctness", "performance", "full"))
    parser.add_argument("--checkpoint-paths", type=Path,
                        help="JSON object mapping each selected P1 key to an existing local path")
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "qualification/results/inference_optimization")
    parser.add_argument("--process-ceiling-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--include-paged", action="store_true",
                        help="Run experimental paged checks separately; never promote failures")
    parser.add_argument("--phase1", action="store_true", help="Run forced-fused diagnostics")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Phase 1 operator diagnostics only; never checkpoint qualification")
    parser.add_argument("--checkpoint-only", action="store_true",
                        help="Execute whole-model parity independently of operator failures")
    parser.add_argument("--reference-smoke", action="store_true",
                        help="Bounded reference benchmark; full five-case baseline remains pending")
    args = parser.parse_args(argv)
    if not 0 < args.process_ceiling_bytes <= 24 * 1024**3 or args.timeout_seconds < 1:
        parser.error("use a positive ceiling <=24 GiB and a positive timeout")
    if args.kernel_only and not args.phase1:
        parser.error("--kernel-only is limited to Phase 1 diagnostics")
    if args.checkpoint_only and (not args.phase1 or args.kernel_only):
        parser.error("--checkpoint-only requires Phase 1 and excludes --kernel-only")
    if args.reference_smoke and (args.phase1 or args.stage not in {"performance", "full"}):
        parser.error("--reference-smoke requires Phase 0 performance/full")
    if args.stage != "smoke" and not args.kernel_only:
        if args.checkpoint_paths is None:
            parser.error("model stages require explicit --checkpoint-paths; no downloads")
        paths = json.loads(args.checkpoint_paths.read_text())
        for key in args.checkpoints:
            if not isinstance(paths.get(key), str) or not Path(paths[key]).is_dir():
                parser.error(f"missing existing local checkpoint path: {key}")
    if args.stage == "performance" and args.correctness_report is None:
        parser.error("performance requires a current --correctness-report")
    root = args.output.resolve() / str(time.time_ns())
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    source = qualification_source()
    report = {"schema_version": "inference-optimization-handoff-v1", "source": source,
              "status": "NOT_RUN", "stages": [], "qualification_promoted": False,
              "phase": 1 if args.phase1 else 0, "kernel_only": args.kernel_only}
    report["reference_smoke"] = args.reference_smoke
    write(root / "handoff.json", report)
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    stages = ["smoke", "correctness", "performance"] if args.stage == "full" else [args.stage]
    correctness = args.correctness_report.resolve() if args.correctness_report else None
    try:
        for stage in stages:
            if qualification_source() != source:
                raise RuntimeError("source, tests or runtime changed during handoff")
            output = root / stage
            if stage == "identity":
                observed = preflight(output / "preflight",
                                     configured_ceiling=args.process_ceiling_bytes)
                worker = output / "worker"
                result = supervised_command([sys.executable,
                    "scripts/qualification/verify_http_identity.py", str(worker)], worker,
                    ceiling=observed["process_ceiling_bytes"], timeout=args.timeout_seconds,
                    env={**environment, "MLX_ONE_FUSED_PATHS": args.checkpoint_paths.read_text(),
                         "MLX_ONE_FUSED_KEYS": json.dumps(args.checkpoints)})
                report["stages"].append({"stage": stage, "status": result["status"]})
                if result["status"] != "completed":
                    raise RuntimeError(f"HTTP identity failed; inspect {worker}")
                continue
            if args.phase1:
                result_path = fused_stage(stage, output, args, source=source,
                                         environment=environment, correctness=correctness)
                report["stages"].append({"stage": stage, "status": "PASS",
                                          "artifact": str(result_path)})
                if stage == "correctness":
                    correctness = result_path
                continue
            command = [sys.executable, "benchmarks/p1_native_qualification.py",
                       "preflight" if stage == "smoke" else stage, "--output", str(output),
                       "--process-ceiling-bytes", str(args.process_ceiling_bytes),
                       "--checkpoints", *args.checkpoints]
            if stage != "smoke":
                command += ["--checkpoint-paths", str(args.checkpoint_paths.resolve())]
                if stage == "correctness":
                    command += ["--backends", "reference_dense"]
                    if args.include_paged:
                        command += ["paged"]
                else:
                    command += ["--phase0-baseline", "--correctness-report", str(correctness)]
                    if args.reference_smoke:
                        command += ["--reference-smoke"]
            write(root / f"{stage}-command.json", {"argv": command, "source": source})
            with (root / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.timeout_seconds)
                finally:
                    # Kill the isolated process group on timeout/interruption, including workers.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            report["stages"].append({"stage": stage, "returncode": returncode,
                                      "status": "PASS" if returncode == 0 else "FAIL"})
            if returncode:
                raise RuntimeError(f"{stage} failed; inspect {root / (stage + '.log')}")
            if stage == "correctness":
                reports = list(output.rglob("correctness.json"))
                if len(reports) != 1:
                    raise RuntimeError("correctness report missing or ambiguous")
                correctness = reports[0].resolve()
            # Worker completion establishes measurement availability, not performance qualification.
        report["status"] = "PASS"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        report.update(status="FAIL", reason="interrupted; isolated worker group terminated")
    finally:
        try:
            rows = collect_results(root)
        except (OSError, ValueError) as exc:
            rows = []
            report.update(status="FAIL", export_error=str(exc))
        write(root / "handoff.json", report)
        lines = [f"Status: {report['status']}", "", f"Source: {source['source_sha256']}",
                 "", "No backend was promoted.", "",
                 *[f"- {row['stage']}: {row['status']}" for row in report["stages"]]]
        if report.get("reason"):
            lines += ["", report["reason"]]
        lines += ["", f"Indexed {len(rows)} measurement records; see measurements.json/CSV."]
        (root / "summary.md").write_text("\n".join(lines) + "\n")
        write_integrity(root)
    print(root)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
