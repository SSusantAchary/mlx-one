"""Backend-free profiles, validation, and subprocess supervision for native baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mlx_one.engine.modes import resolve_backend
from mlx_one.p1_profiles import CHECKPOINTS as P1_CHECKPOINTS

CHECKPOINTS = {
    "qwen": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    "lfm": ("mlx-community/LFM2-350M-4bit", "18dc72abf3b2337f9123cfd6eeeb58dfa7947066"),
    "qwen15": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    "qwen3": ("Qwen/Qwen2.5-3B", "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"),
}
CHECKPOINT_CONTEXTS = {"Qwen/Qwen2.5-0.5B": 32768, "Qwen/Qwen2.5-1.5B": 131072,
                       "Qwen/Qwen2.5-3B": 32768, "mlx-community/LFM2-350M-4bit": 128000}
CHECKPOINT_CONTEXTS.update({c.model_id: c.context_tokens for c in P1_CHECKPOINTS.values()})
TOLERANCES = {"float32": (1e-4, 1e-4), "float16": (1e-2, 1e-2), "bfloat16": (5e-2, 1e-2)}


class BaselineError(ValueError):
    pass


@dataclass(frozen=True)
class BaselineProfile:
    model_id: str
    revision: str
    inference_mode: str
    context_tokens: int = 2048
    slots: int = 1
    output_tokens: int = 128
    repeats: int = 3
    seed: int = 0
    workload: str = "baseline"
    timeout_seconds: float = 600
    memory_fraction: float = 0.75
    cache_type: str = "bf16"
    atol: float | None = None
    rtol: float | None = None
    execution_dtype: str | None = None
    chunk_tokens: int = 512
    max_scheduled_tokens: int = 2048
    max_prefill_tokens: int = 1024
    ttft_budget_ms: float | None = None
    decode_gap_budget_ms: float | None = None
    reference_run: str | None = None
    page_tokens: int = 32
    kv_bits: int | None = None
    duration_seconds: int | None = None
    shared_prefix_fraction: float | None = None
    admission_policy: str = "deadline"
    quality_limits: dict[str, float] | None = None
    leak_budget_bytes: int | None = None
    min_decode_tokens_per_second: float | None = None
    quality_run: str | None = None
    pressure_allocation_bytes: int | None = None
    pressure_test_ceiling_bytes: int | None = None
    memory_tolerances: dict | None = None
    calibration_phase: str = "decode"
    queue_size: int | None = None
    output_buffer_events: int = 32
    output_stall_timeout_s: float = 30.0
    prefix_cache_budget_bytes: int = 512 * 1024**2
    cache_capacity_bytes: int | None = None
    pressure_config: dict | None = None
    traffic_manifest: str | None = None
    traffic_identity: str | None = None
    max_queue_wait_ms: float | None = None
    max_service_gap_ms: float | None = None
    quality_manifest: str | None = None
    quality_dataset_identity: dict | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision):
            raise BaselineError("baseline requires a model and immutable checkpoint revision")
        if not self.inference_mode:
            raise BaselineError("baseline requires explicit inference mode")
        if self.inference_mode == "auto":
            raise BaselineError("benchmarks require an explicit resolved execution mode")
        resolve_backend(self.inference_mode, None)
        if min(self.context_tokens, self.slots, self.output_tokens, self.repeats) < 1:
            raise BaselineError("profile counts must be positive")
        if not 0 < self.memory_fraction <= 0.75 or self.timeout_seconds <= 0:
            raise BaselineError("invalid watchdog limits")
        if self.cache_type not in {"bf16", "f16"} and not (
            self.inference_mode == "paged" and self.kv_bits in {4, 8}
            and self.cache_type == f"q{self.kv_bits}_0"
        ):
            raise BaselineError("baseline uses unquantized KV only")
        if self.page_tokens not in {16, 32, 64, 128} or self.kv_bits not in {None, 4, 8}:
            raise BaselineError("unsupported page or quantization profile")
        if self.kv_bits and self.inference_mode != "paged":
            raise BaselineError("quantized page profiles require paged mode")
        if self.duration_seconds is not None and self.duration_seconds not in {1800, 3600}:
            raise BaselineError("soak profiles require 30 or 60 minutes")
        if (self.workload == "soak") != (self.duration_seconds is not None):
            raise BaselineError("soak workload requires an explicit 30/60 minute duration")
        for bound in (self.pressure_allocation_bytes, self.pressure_test_ceiling_bytes,
                      self.cache_capacity_bytes, self.queue_size):
            if bound is not None and (not isinstance(bound, int) or bound < 1):
                raise BaselineError("memory and queue bounds must be positive integers")
        if self.output_buffer_events < 1 or self.output_stall_timeout_s <= 0:
            raise BaselineError("invalid output buffer bounds")
        if self.calibration_phase not in {"decode", "prefill"}:
            raise BaselineError("invalid calibration phase")
        if self.workload == "peak_calibration" and self.inference_mode not in {
                "paged", "experimental_hybrid_batch", "experimental_dense_batch"}:
            raise BaselineError("peak calibration requires explicit optimized execution")
        if self.workload == "lifecycle" and self.inference_mode != "experimental_hybrid_batch":
            raise BaselineError("native lifecycle profiles require explicit hybrid mode")
        if self.workload == "microbench" and self.inference_mode != "paged":
            raise BaselineError("direct attention microbenchmarks require explicit paged mode")
        if self.workload == "quality" and self.inference_mode not in {
            "paged", "experimental_hybrid_batch"
        }:
            raise BaselineError("quality profiles require an explicit paged or hybrid target")
        if self.shared_prefix_fraction not in {None, 0, .25, .5, .8}:
            raise BaselineError("unsupported shared prefix fraction")
        if self.admission_policy not in {"deadline", "cache_aware"}:
            raise BaselineError("unsupported admission policy")
        if self.leak_budget_bytes is not None and self.leak_budget_bytes < 0:
            raise BaselineError("leak budget must be nonnegative")
        for limit in (self.max_queue_wait_ms, self.max_service_gap_ms):
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise BaselineError("fairness limits must be finite and positive")
        if self.min_decode_tokens_per_second is not None and (
            not math.isfinite(self.min_decode_tokens_per_second)
            or self.min_decode_tokens_per_second <= 0
        ):
            raise BaselineError("declared decode floor must be finite and positive")
        if self.quality_limits is not None and (
            set(self.quality_limits) != {"max_logit_error", "max_mean_nll_delta",
                                         "max_retrieval_accuracy_drop"}
            or any(not math.isfinite(v) or v < 0 for v in self.quality_limits.values())
        ):
            raise BaselineError("declare all finite nonnegative quality limits")
        if self.workload == "parity" and self.output_tokens < 3:
            raise BaselineError("parity requires at least three forced steps")
        if self.execution_dtype not in {None, "float16", "bfloat16"}:
            raise BaselineError("unsupported execution dtype")
        if self.inference_mode == "experimental_dense_batch" and (
            self.cache_type != "f16" or self.execution_dtype != "float16"
        ):
            raise BaselineError("batched profiles require explicit FP16 execution and KV")
        if min(self.chunk_tokens, self.max_scheduled_tokens, self.max_prefill_tokens) < 1:
            raise BaselineError("invalid scheduling budgets")
        if (self.ttft_budget_ms is None) != (self.decode_gap_budget_ms is None):
            raise BaselineError("declare both latency budgets")
        if self.ttft_budget_ms is not None and any(not math.isfinite(value) or value <= 0
            for value in (self.ttft_budget_ms, self.decode_gap_budget_ms)):
            raise BaselineError("latency budgets must be finite and positive")
        if self.workload not in {"baseline", "prefix", "long", "parity", "batching", "turnover",
                                 "mixed", "cancellation", "slow_consumer", "microbench",
                                 "quality", "pressure", "soak", "multi_turn",
                                 "lifecycle", "peak_calibration", "p0_prefix"}:
            raise BaselineError("unknown baseline workload")
        if (self.atol is None) != (self.rtol is None):
            raise BaselineError("override both atol and rtol")
        if self.atol is not None and (not math.isfinite(self.atol) or self.atol < 0
                                     or not math.isfinite(self.rtol) or self.rtol < 0):
            raise BaselineError("tolerances must be finite and nonnegative")

    @property
    def profile_id(self) -> str:
        payload = asdict(self)
        new = {"checkpoint_path": None, "page_tokens": 32, "kv_bits": None,
               "duration_seconds": None,
               "shared_prefix_fraction": None, "admission_policy": "deadline",
               "quality_limits": None,
               "leak_budget_bytes": None, "min_decode_tokens_per_second": None, "quality_run": None,
               "quality_manifest": None, "quality_dataset_identity": None,
               "traffic_manifest": None, "traffic_identity": None,
               "max_queue_wait_ms": None, "max_service_gap_ms": None,
               "queue_size": None, "output_buffer_events": 32, "output_stall_timeout_s": 30.,
               "cache_capacity_bytes": None, "prefix_cache_budget_bytes": 512 * 1024**2,
               "pressure_config": None, "calibration_phase": "decode",
               "pressure_allocation_bytes": None,
               "pressure_test_ceiling_bytes": None, "memory_tolerances": None}
        advanced = (self.inference_mode in {"paged", "experimental_hybrid_batch"}
                    or self.workload in {"microbench", "quality", "pressure", "soak",
                                        "multi_turn", "lifecycle", "peak_calibration", "p0_prefix"}
                    or any(payload[k] != v for k, v in new.items()))
        if not advanced:
            for key in new:
                payload.pop(key)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return f"native-baseline-v{2 if advanced else 1}-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "profile_id": self.profile_id, "offline": True,
                "temperature": 0, "speculation": False,
                "prefix_reuse": self.workload in {"prefix", "multi_turn", "soak", "p0_prefix"},
                "tolerances": TOLERANCES}


def deterministic_prompt(tokenizer: Any, target_tokens: int, *, suffix: str = "") -> str:
    """Create locally reproducible, numbered retrieval text near a token target."""

    sections: list[str] = []
    index = 1
    while len(tokenizer.encode("".join(sections))) < target_tokens:
        sections.append(
            f"SECTION {index:05d}\n"
            f"The project identifier is PROJECT-{index:05d}.\n"
            f"The owner is OWNER-{index:05d}.\n"
            f"The checksum is VALUE-{index:05d}.\n\n"
        )
        index += 1
    prompt = "".join(sections)
    # Keep the final retrieval request intact while approaching the requested size.
    while len(tokenizer.encode(prompt + suffix)) > target_tokens and sections:
        sections.pop()
        prompt = "".join(sections)
    return prompt + suffix


def profiles(mode: str, *, models=("qwen", "lfm"), workload="baseline"):
    if mode in {"paged", "experimental_hybrid_batch"} or workload in {
        "microbench", "quality", "pressure", "soak", "multi_turn"
    }:
        from mlx_one.inference_qualification import qualification_profiles

        yield from qualification_profiles(mode, models=models, workload=workload)
        return
    batching = mode == "experimental_dense_batch" or workload == "batching"
    for key in models:
        if (batching or mode == "experimental_block") and not key.startswith("qwen"):
            continue
        model, revision = CHECKPOINTS[key]
        contexts = (2048, 8192) if workload in {"baseline", "batching"} else (
            (16384, 32768) if workload == "long" else (2048,))
        slots = (1, 2, 4, 8) if batching else ((1, 2, 4) if workload == "baseline" else (1,))
        if workload in {"turnover", "mixed", "cancellation", "slow_consumer"}:
            slots = (2, 4, 8)
        for context in contexts:
            for size in slots:
                for chunk in ((512, 1024, 2048, 4096) if batching else (512,)):
                    fp16 = batching or mode == "experimental_block" or workload in {
                        "turnover", "mixed", "cancellation", "slow_consumer"}
                    yield BaselineProfile(model, revision, mode, context, size, workload=workload,
                                          output_tokens=32 if workload == "parity" else 128,
                                          cache_type="f16" if fp16 else "bf16",
                                          execution_dtype="float16" if fp16 else None,
                                          chunk_tokens=chunk, max_prefill_tokens=max(1024, chunk),
                                          max_scheduled_tokens=max(2048, chunk * 2))


def source_provenance(root: Path) -> dict[str, Any]:
    """Hash working source as well as HEAD, since a baseline may precede a commit."""
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    digest = hashlib.sha256()
    for parent in (root / "src" / "mlx_one", root / "benchmarks"):
        for path in sorted(parent.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    packages = {}
    for name in ("mlx", "mlx-one", "tokenizers", "huggingface-hub", "safetensors", "uvicorn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    from mlx_one.provenance import collect_software_provenance

    return {"software": collect_software_provenance().to_dict(),
            "git_revision": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": digest.hexdigest(), "packages": packages}


def validate_identity(profile: BaselineProfile, runtime: dict[str, Any]) -> None:
    report = runtime.get("inference", runtime)
    scope = report.get("scope", {})
    if (scope.get("model_id") != profile.model_id or scope.get("revision") != profile.revision
            or report.get("engine_mode") != profile.inference_mode):
        raise BaselineError("backend substitution or checkpoint identity mismatch")
    for field in ("topology_digest", "weight_precision", "cache_precision", "mlx_version",
                  "hardware", "source_revision"):
        if not scope.get(field):
            raise BaselineError(f"incomplete identity metadata: {field}")
    expected = ("float16,uint8" if profile.kv_bits else
                ("bfloat16" if profile.cache_type == "bf16" else "float16"))
    if scope["cache_precision"] != expected:
        raise BaselineError("actual KV precision does not match requested baseline precision")
    if runtime.get("scheduler", {}).get("parallel_slots", profile.slots) != profile.slots:
        raise BaselineError("client fanout does not match configured server slots")
    caps = report.get("capabilities", {})
    if (bool(caps.get("continuous_batching"))
            != (profile.inference_mode in {"experimental_dense_batch", "paged",
                                          "experimental_hybrid_batch"})):
        raise BaselineError("continuous batching capability differs from explicit mode")
    if (bool(caps.get("paged_attention")) != (profile.inference_mode == "paged")
            or bool(caps.get("hybrid_prefix_restore"))
            != (profile.inference_mode == "experimental_hybrid_batch")):
        raise BaselineError("capability differs from selected explicit execution path")
    if profile.execution_dtype and scope["weight_precision"] != profile.execution_dtype:
        raise BaselineError("actual weight precision differs from explicit execution dtype")


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "completed"
             and row.get("temperature_phase") != "warmup"]
    warm = [row for row in valid if row.get("temperature_phase") == "warm"]
    gaps = [gap for row in warm for gap in row.get("sse_delivery_gaps_ms", [])]
    model_gaps = [gap for row in warm for gap in row.get("model_token_gaps_ms", [])]
    wall_groups = {}
    for row in warm:
        wall_groups.setdefault(row["group_id"], []).append(row)
    aggregate = []
    for group in wall_groups.values():
        elapsed = max(r["ended"] for r in group) - min(r["started"] for r in group)
        aggregate.append(sum(r["generated_tokens"] for r in group) / elapsed if elapsed else None)
    return {
        "completed": len(valid), "failed": sum(
            r.get("status") not in {"completed", "cancelled_expected",
                                    "deadline_expected", "stalled_expected"}
            for r in rows),
        "cold_client_ttft_ms": [r.get("client_ttft_ms") for r in valid
                                if r.get("temperature_phase") == "cold"],
        "warm_client_ttft_p50_ms": percentile([r["client_ttft_ms"] for r in warm
                                               if r.get("client_ttft_ms") is not None], .5),
        "warm_client_ttft_p95_ms": percentile([r["client_ttft_ms"] for r in warm
                                               if r.get("client_ttft_ms") is not None], .95),
        "sse_delivery_gap_p50_ms": percentile(gaps, .5),
        "sse_delivery_gap_p95_ms": percentile(gaps, .95),
        "model_token_gap_p50_ms": percentile(model_gaps, .5),
        "model_token_gap_p95_ms": percentile(model_gaps, .95),
        "aggregate_output_tokens_per_second_by_group": aggregate,
        "qualification_promoted": False,
    }


def process_rss(pid: int) -> int | None:
    try:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(raw.strip()) * 1024
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def supervise(process, *, timeout, memory_limit, rss=process_rss, clock=time.monotonic,
              sleep=time.sleep) -> dict[str, Any]:
    started = clock()
    samples = []
    failure = None
    while process.poll() is None:
        current = rss(process.pid)
        samples.append({"elapsed_seconds": clock() - started, "rss_bytes": current})
        if current is None:
            failure = "memory_watchdog_unavailable"
        elif current > memory_limit:
            failure = "memory_watchdog_terminated"
        elif clock() - started >= timeout:
            failure = "timeout"
        if failure:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        sleep(.1)
    code = process.wait()
    return {"status": failure or ("completed" if code == 0 else "worker_failed"),
            "exit_code": code, "rss_samples": samples,
            "process_peak_rss_bytes": max((s["rss_bytes"] or 0 for s in samples), default=0)}


def run_profile(profile: BaselineProfile, output: Path, *, hardware: dict[str, Any],
                provenance: dict[str, Any]) -> dict[str, Any]:
    if not hardware.get("memory_bytes") or not hardware.get("chip"):
        raise BaselineError("detected hardware identity and RAM are required for the watchdog")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": "native-baseline-v1", "profile": profile.to_dict(),
                "hardware": hardware, "source": provenance, "output": str(output.resolve())}
    request_path = output / "manifest.json"
    request_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for bound in (profile.pressure_allocation_bytes, profile.pressure_test_ceiling_bytes):
        if bound is not None and bound > hardware["memory_bytes"] * profile.memory_fraction:
            outcome = {"status": "rejected_pressure", "reason": "pressure bound exceeds watchdog"}
            (output / "failure.json").write_text(json.dumps(outcome) + "\n")
            (output / "supervision.json").write_text(json.dumps(outcome) + "\n")
            return outcome
    maximum = CHECKPOINT_CONTEXTS.get(profile.model_id)
    if maximum is not None and profile.context_tokens + profile.output_tokens > maximum:
        outcome = {"status": "rejected_context", "checkpoint_limit": maximum,
                   "requested_total_tokens": profile.context_tokens + profile.output_tokens}
        (output / "failure.json").write_text(json.dumps(outcome, indent=2) + "\n")
        (output / "supervision.json").write_text(json.dumps(outcome, indent=2) + "\n")
        return outcome
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlx_one.inference_baseline_worker",
             str(request_path.resolve())],
            stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
        )
        outcome = supervise(process, timeout=profile.timeout_seconds,
                            memory_limit=int(hardware["memory_bytes"] * profile.memory_fraction))
    (output / "supervision.json").write_text(json.dumps(outcome, indent=2) + "\n")
    # Every abnormal exit gets a durable failure artifact even if the worker crashed at import.
    if outcome["status"] != "completed":
        failure_path = output / "failure.json"
        if failure_path.exists():
            outcome["worker_failure"] = json.loads(failure_path.read_text())
        failure_path.write_text(json.dumps(outcome, indent=2) + "\n")
    from mlx_one.inference_qualification import write_integrity

    write_integrity(output)
    return outcome


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-mode", required=True,
                        choices=("reference_dense", "experimental_block",
                                 "experimental_dense_batch", "paged", "experimental_hybrid_batch"))
    parser.add_argument("--models", nargs="+", choices=tuple(CHECKPOINTS),
                        default=["qwen", "lfm"])
    parser.add_argument("--workload", choices=("baseline", "prefix", "long", "parity",
                                             "batching", "turnover",
                                 "mixed", "cancellation", "slow_consumer", "microbench",
                                 "quality", "pressure", "soak", "multi_turn",
                                 "lifecycle", "peak_calibration"),
                        default="baseline")
    parser.add_argument("--output", type=Path, default=Path("qualification/results"))
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--qualification-profiles", action="store_true",
                        help="Use the PR-07--12 matrix also for matched dense reference runs")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--ttft-budget-ms", type=float)
    parser.add_argument("--decode-gap-budget-ms", type=float)
    parser.add_argument("--reference-run", type=str)
    parser.add_argument("--quality-run", type=str,
                        help="Checksummed quality run required for 128K admission evidence")
    parser.add_argument("--execution-dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--slots", type=int)
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--page-tokens", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--kv-bits", type=int, choices=(4, 8))
    parser.add_argument("--soak-minutes", type=int, choices=(30, 60))
    parser.add_argument("--shared-prefix-fraction", type=float, choices=(0, .25, .5, .8))
    parser.add_argument("--admission-policy", choices=("deadline", "cache_aware"))
    parser.add_argument("--quality-limits", type=Path)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument("--traffic-manifest", type=Path)
    parser.add_argument("--max-queue-wait-ms", type=float)
    parser.add_argument("--max-service-gap-ms", type=float)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--pressure-allocation-bytes", type=int)
    parser.add_argument("--pressure-test-ceiling-bytes", type=int)
    parser.add_argument("--memory-tolerances", type=Path)
    parser.add_argument("--calibration-phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--output-buffer-events", type=int, default=32)
    parser.add_argument("--output-stall-timeout-s", type=float, default=30.)
    parser.add_argument("--cache-memory-budget-bytes", type=int)
    parser.add_argument("--prefix-cache-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--pressure-config", type=Path)
    parser.add_argument("--leak-budget-bytes", type=int)
    parser.add_argument("--min-decode-tok-s", type=float)
    args = parser.parse_args(argv)
    if args.qualification_profiles:
        from mlx_one.inference_qualification import qualification_profiles

        candidates = qualification_profiles(args.inference_mode, models=args.models,
                                            workload=args.workload)
    else:
        candidates = profiles(args.inference_mode, models=args.models, workload=args.workload)
    replay = None
    if args.traffic_manifest:
        from mlx_one.traffic_replay import load_traffic_manifest

        replay = load_traffic_manifest(args.traffic_manifest)
    dataset = None
    if args.quality_manifest:
        from mlx_one.quality_dataset import load_quality_manifest

        dataset = load_quality_manifest(args.quality_manifest)
    selected = [replace(p, atol=args.atol, rtol=args.rtol,
                        ttft_budget_ms=args.ttft_budget_ms,
                        decode_gap_budget_ms=args.decode_gap_budget_ms,
                        reference_run=args.reference_run,
                        quality_run=args.quality_run,
                        traffic_manifest=str(args.traffic_manifest.resolve())
                        if args.traffic_manifest else None,
                        traffic_identity=replay["sha256"] if replay else None,
                        context_tokens=args.context_tokens or p.context_tokens,
                        output_tokens=args.output_tokens or p.output_tokens,
                        pressure_allocation_bytes=args.pressure_allocation_bytes,
                        pressure_test_ceiling_bytes=args.pressure_test_ceiling_bytes,
                        memory_tolerances=json.loads(args.memory_tolerances.read_text())
                        if args.memory_tolerances else None,
                        calibration_phase=args.calibration_phase,
                        queue_size=args.queue_size,
                        output_buffer_events=args.output_buffer_events,
                        output_stall_timeout_s=args.output_stall_timeout_s,
                        cache_capacity_bytes=args.cache_memory_budget_bytes,
                        prefix_cache_budget_bytes=args.prefix_cache_bytes,
                        pressure_config=json.loads(args.pressure_config.read_text())
                        if args.pressure_config else None,
                        max_queue_wait_ms=args.max_queue_wait_ms,
                        max_service_gap_ms=args.max_service_gap_ms,
                        quality_manifest=str(args.quality_manifest.resolve())
                        if args.quality_manifest else None,
                        quality_dataset_identity=dataset["identity"] if dataset else None,
                        kv_bits=args.kv_bits,
                        cache_type=f"q{args.kv_bits}_0" if args.kv_bits else p.cache_type,
                        quality_limits=(json.loads(args.quality_limits.read_text())
                                        if args.quality_limits else None),
                        leak_budget_bytes=args.leak_budget_bytes,
                        min_decode_tokens_per_second=args.min_decode_tok_s,
                        execution_dtype=args.execution_dtype or p.execution_dtype)
                for p in candidates]
    selected = [profile for profile in selected if
                (args.slots is None or profile.slots == args.slots)
                and (args.context_tokens is None or profile.context_tokens == args.context_tokens)
                and (args.chunk_tokens is None or profile.chunk_tokens == args.chunk_tokens)
                and (args.page_tokens is None or profile.page_tokens == args.page_tokens)
                and (args.soak_minutes is None
                     or profile.duration_seconds == args.soak_minutes * 60)
                and (args.shared_prefix_fraction is None
                     or profile.shared_prefix_fraction == args.shared_prefix_fraction)
                and (args.admission_policy is None
                     or profile.admission_policy == args.admission_policy)]
    selected = list({profile.profile_id: profile for profile in selected}.values())
    if not selected:
        parser.error("no profiles match the selected model, workload, or dimensions")
    if args.list_profiles:
        print(json.dumps([p.to_dict() for p in selected], indent=2))
        return 0
    from mlx_one.hardware import detect_hardware

    hardware = detect_hardware().to_dict()
    root = Path(__file__).resolve().parents[2]
    provenance = source_provenance(root)
    failed = False
    for profile in selected:
        output = (args.output / hardware["profile_id"] / profile.model_id.replace("/", "--")
                  / f"{profile.profile_id}-{time.time_ns()}")
        outcome = run_profile(profile, output, hardware=hardware, provenance=provenance)
        print(json.dumps({"profile": profile.profile_id, "status": outcome["status"],
                          "artifacts": str(output)}), flush=True)
        failed |= outcome["status"] != "completed"
    return int(failed)
