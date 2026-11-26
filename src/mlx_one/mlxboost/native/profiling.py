"""Synchronized wall time, never mislabeled GPU-only time. Explicit developer API."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .capabilities import binary_sha256, distribution_version, file_sha256, source_sha256
from .registry import write_immutable


def _command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment(mx=None):
    """No host names, serial numbers, model paths or prompt text."""
    info = mx.device_info() if mx is not None else {}
    python_sources = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        python_sources.update(path.name.encode() + b"\0" + path.read_bytes())
    harness = Path(sys.argv[0])
    return dict(
        git_sha=_command("git", "rev-parse", "HEAD"),
        python=sys.version.split()[0],
        macos=platform.mac_ver()[0],
        os=platform.system(),
        machine=platform.machine(),
        mlx_version=distribution_version("mlx"),
        mlx_metal_version=distribution_version("mlx-metal"),
        device=info,
        implementation_sha256=source_sha256(),
        python_sources_sha256=python_sources.hexdigest(),
        harness_sha256=file_sha256(harness) if harness.is_file() else None,
        working_tree_dirty=bool(_command("git", "status", "--porcelain")),
        binary_sha256=binary_sha256(),
        compiler=_command("clang++", "--version"),
        sdk=_command("xcrun", "--show-sdk-version"),
        xcode=_command("xcodebuild", "-version"),
        metal_compiler=_command("xcrun", "--find", "metal"),
        load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        thermal_state="uncontrolled; not measured",
        threads=os.cpu_count(),
    )


def summary(samples):
    ordered = sorted(samples)
    if not ordered or any(not math.isfinite(v) or v <= 0 for v in ordered):
        raise ValueError("positive timing samples required")

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return dict(
        samples_ms=samples,
        sample_count=len(samples),
        median_ms=statistics.median(samples),
        p50_ms=percentile(0.5),
        p90_ms=percentile(0.9),
        p99_ms=percentile(0.99),
        stdev_ms=statistics.pstdev(samples),
        iqr_ms=percentile(0.75) - percentile(0.25),
    )


def benchmark(functions, *, warmup=5, trials=30, seed=7, mx=None, budget_seconds=60):
    """Profile evaluated MLX-pytree-returning callables; None means already blocking.

    Interleave backends in a seeded randomized order each round. Caller must
    materialize identical inputs before entry. Cold first use includes pipeline
    compilation if needed; compilation is not separately observable here.
    """
    if not functions or not 0 <= warmup <= 100 or not 3 <= trials <= 1000:
        raise ValueError("require functions, warmup 0..100, trials 3..1000")
    if not 0 < budget_seconds <= 600:
        raise ValueError("time budget must be in (0,600]")
    if mx is None:
        import mlx.core as mx
    rng = random.Random(seed)
    deadline = time.monotonic() + budget_seconds

    def timed(fn):
        if time.monotonic() >= deadline:
            raise TimeoutError("profiling time budget exceeded")
        mx.synchronize()
        started = time.perf_counter_ns()
        out = fn()
        if out is not None:
            mx.eval(out)
        mx.synchronize()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return elapsed

    names = list(functions)
    rng.shuffle(names)
    cold = {name: timed(functions[name]) for name in names}
    warm = {name: [] for name in names}
    for _ in range(warmup):
        rng.shuffle(names)
        for name in names:
            warm[name].append(timed(functions[name]))
    # The MLX counter is process-global: benchmark sessions must run exclusively.
    mx.reset_peak_memory()
    samples = {name: [] for name in names}
    order = []
    for _ in range(trials):
        rng.shuffle(names)
        order.append(list(names))
        for name in names:
            samples[name].append(timed(functions[name]))
    memory = dict(
        mlx_peak_bytes=mx.get_peak_memory(),
        mlx_active_bytes=mx.get_active_memory(),
        mlx_cache_bytes=mx.get_cache_memory(),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        process_physical_footprint_bytes=None,
        notes="MLX counters and process high-water RSS have distinct scopes",
    )
    return {
        name: dict(
            status="measured",
            timing_scope="synchronized_wall",
            cold_first_use_ms=cold[name],
            warmup_samples_ms=warm[name],
            **summary(samples[name]),
            memory=memory,
            methodology=dict(
                warmup=warmup,
                trials=trials,
                seed=seed,
                synchronized=True,
                randomized_order=order,
                compilation="included in first profiled call only if still needed",
                first_use="first timed call after caller preparation; not guaranteed process-cold",
                cooldown="none",
                concurrency="exclusive process required",
            ),
        )
        for name in functions
    }


@contextmanager
def metal_capture(path, *, mx=None):
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("capture requires MTL_CAPTURE_ENABLED=1")
    if not _command("xcodebuild", "-version"):
        raise RuntimeError("capture requires active full Xcode")
    if mx is None:
        import mlx.core as mx
    mx.synchronize()
    mx.metal.start_capture(str(path))
    try:
        yield
        mx.synchronize()
    finally:
        mx.metal.stop_capture()


def export_report(directory, record):
    path = write_immutable(directory, record)
    rows = [
        "# MLXBoost Phase 0 measurement",
        "",
        "Status: EXPERIMENTAL. No model speedup claim.",
        "",
        "Times include dispatch, evaluation and GPU completion.",
        "",
        "| Backend | Samples | Median ms | p90 ms | First profiled ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in sorted(record.get("profiles", {}).items()):
        rows.append(
            f"| {name} | {result['sample_count']} | {result['median_ms']:.6f} | "
            f"{result['p90_ms']:.6f} | {result['cold_first_use_ms']:.6f} |"
        )
    for skip in record.get("skipped", []):
        rows.extend(["", "Skipped: " + skip])
    summary_path = Path(directory) / (path.stem + ".md")
    payload = "\n".join(rows) + "\n"
    try:
        with summary_path.open("x") as handle:
            handle.write(payload)
    except FileExistsError:
        if summary_path.read_text() != payload:
            raise ValueError("immutable summary mismatch") from None
    return path
