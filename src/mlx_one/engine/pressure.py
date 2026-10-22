"""Live pressure policy, with independent physical observations and owner accounting."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

from mlx_one.engine.cache_errors import CacheCapacityError


@dataclass(frozen=True)
class MemoryPressureConfig:
    enabled: bool = True
    process_ceiling_bytes: int | None = None
    memory_fraction: float = .75
    warning_fraction: float = .9
    sample_interval_s: float = 1.0

    def __post_init__(self):
        if (not 0 < self.memory_fraction <= .75 or not 0 < self.warning_fraction < 1
                or self.sample_interval_s <= 0
                or (self.process_ceiling_bytes is not None and self.process_ceiling_bytes < 1)):
            raise ValueError("invalid memory pressure configuration")


@dataclass(frozen=True)
class PressureTelemetry:
    rss_bytes: int | None = None
    installed_bytes: int | None = None
    mlx_active_bytes: int | None = None
    mlx_cached_bytes: int | None = None
    mlx_peak_bytes: int | None = None
    recommended_bytes: int | None = None
    os_pressure: str | None = None
    swap_bytes: int | None = None


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=.5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def sample_telemetry():
    """Read-only host queries; do not initialize MLX to collect diagnostics."""
    rss = _command(["ps", "-o", "rss=", "-p", str(os.getpid())])
    total = pressure = swap = None
    if platform.system() == "Darwin":
        total = _command(["sysctl", "-n", "hw.memsize"])
        level = _command(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        pressure = {"1": "normal", "2": "warning", "4": "critical"}.get(level)
        usage = _command(["sysctl", "-n", "vm.swapusage"])
        match = re.search(r"used\s*=\s*([\d.]+)([MG])", usage or "")
        if match:
            swap = int(float(match[1]) * (1024**2 if match[2] == "M" else 1024**3))
    elif hasattr(os, "sysconf"):
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            pass
    mx = sys.modules.get("mlx.core")
    def observed(name):
        try:
            return int(getattr(mx, name)()) if mx is not None else None
        except (AttributeError, RuntimeError):
            return None
    recommended = None
    try:
        info = mx.device_info() if mx is not None else {}
        recommended = info.get("max_recommended_working_set_size")
    except (AttributeError, RuntimeError):
        pass
    return PressureTelemetry(int(rss) * 1024 if rss and rss.isdigit() else None,
                             int(total) if total is not None else None,
                             observed("get_active_memory"), observed("get_cache_memory"),
                             observed("get_peak_memory"), recommended, pressure, swap)


class MemoryPressureController:
    def __init__(self, config=None, *, telemetry=sample_telemetry, clock=time.monotonic):
        self.config = config or MemoryPressureConfig()
        self.telemetry, self.clock = telemetry, clock
        self.last = PressureTelemetry()
        self.last_sample = -float("inf")
        self.state = "unknown"
        self.ceiling = None
        self._swap_streak = self._recovery = 0
        self.eviction_actions = self.queued_admissions = self.rejected_forwards = 0
        self._stop = threading.Event()
        self._sampler = None
        self._latest = None
        self._last_shed_sample = None

    def start(self):
        """Sample off the inference actor; lifecycle owners start/stop this with the model."""
        if not self.config.enabled or self._sampler is not None:
            return
        self._stop.clear()
        self._latest = (self.telemetry(), self.clock())
        self.last_sample = -float("inf")
        def sample():
            while not self._stop.wait(self.config.sample_interval_s):
                try:
                    snapshot = self.telemetry()
                except (OSError, RuntimeError, ValueError):
                    snapshot = PressureTelemetry()
                self._latest = (snapshot, self.clock())
        self._sampler = threading.Thread(target=sample, name="mlx-one-pressure", daemon=True)
        self._sampler.start()
        self.poll()

    def close(self):
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=3)
            self._sampler = None

    def poll(self):
        if not self.config.enabled:
            self.state = "disabled"
            return self.state
        now = self.clock()
        if now - self.last_sample < self.config.sample_interval_s:
            return self.state
        current, sampled_at = (self._latest if self._sampler is not None
                               else (self.telemetry(), now))
        if self._sampler is not None and sampled_at <= self.last_sample:
            return self.state
        self._swap_streak = (self._swap_streak + 1 if self.last.swap_bytes is not None
                             and current.swap_bytes is not None
                             and current.swap_bytes > self.last.swap_bytes else 0)
        self.last, self.last_sample = current, sampled_at
        ceilings = [v for v in (self.config.process_ceiling_bytes, current.recommended_bytes,
                                int(current.installed_bytes * self.config.memory_fraction)
                                if current.installed_bytes else None) if v is not None]
        self.ceiling = min(ceilings) if ceilings else None
        observed = self.working_set()
        if (current.os_pressure == "critical" or self._swap_streak >= 2
                or (self.ceiling is not None and observed >= self.ceiling)):
            self.state, self._recovery = "critical", 0
        elif (current.os_pressure == "warning" or (self.ceiling is not None
                and observed >= self.ceiling * self.config.warning_fraction)):
            self.state, self._recovery = "warning", 0
        elif self.ceiling is None and current.os_pressure is None:
            self.state = "unknown"
        else:
            self._recovery += 1
            if self.state not in {"warning", "critical"} or self._recovery >= 2:
                self.state = "normal"
        return self.state

    def working_set(self, owner_physical=0):
        # These observations overlap; never add RSS and MLX or logical owner bytes.
        return max(self.last.rss_bytes or 0,
                   (self.last.mlx_active_bytes or 0) + (self.last.mlx_cached_bytes or 0),
                   owner_physical)

    def available(self, owner_physical=0, outstanding=0):
        return (None if self.ceiling is None else
                max(self.ceiling - self.working_set(owner_physical) - outstanding, 0))

    def admission(self, amount, *, owner_physical=0, outstanding=0):
        self.poll()
        if not self.config.enabled:
            return
        if self.ceiling is not None and amount > self.ceiling:
            raise CacheCapacityError("request exceeds the pressure-controlled process ceiling",
                                     retryable=False)
        available = self.available(owner_physical, outstanding)
        if self.state in {"warning", "critical"} or (available is not None and amount > available):
            self.queued_admissions += 1
            raise CacheCapacityError("admission deferred by live memory pressure", retryable=True)

    def forward(self, amount, *, owner_physical=0, outstanding=0):
        self.poll()
        if not self.config.enabled:
            return
        available = self.available(owner_physical, outstanding)
        if (available is not None and amount > available
                or self.state == "critical" and available is None):
            self.rejected_forwards += 1
            raise CacheCapacityError("forward cannot fit its transient pressure reservation",
                                     retryable=False)

    def _refresh_after_reclaim(self, snapshot):
        self._latest = (snapshot, self.clock())
        self.last_sample = -float("inf")
        self.poll()

    def shed(self, runtime):
        if self.poll() not in {"warning", "critical"}:
            return
        if self._last_shed_sample == (self.last_sample, id(runtime)):
            return
        from mlx_one.engine.device_executor import DeviceExecutor

        owner = DeviceExecutor._shared
        if owner is None or not owner.is_owner or getattr(runtime, "_transient_bytes", 0):
            return
        stats = runtime.stats()
        if not (stats.evictable_bytes or stats.retained_free_bytes
                or self.last.mlx_cached_bytes):
            return
        self._last_shed_sample = (self.last_sample, id(runtime))
        mx = sys.modules.get("mlx.core")
        synchronize = getattr(mx, "synchronize", None)
        if mx is not None and (not callable(synchronize) or
                               not callable(getattr(mx, "clear_cache", None))):
            return
        if callable(synchronize):
            synchronize()
        runtime.clear_prefixes()
        runtime.block_pool.discard_free()
        if mx is not None and callable(getattr(mx, "clear_cache", None)):
            mx.clear_cache()
        runtime._capacity_snapshot = None
        self.eviction_actions += 1
        self._refresh_after_reclaim(self.telemetry())

    def stats(self):
        return {"state": self.state, "ceiling_bytes": self.ceiling,
                "telemetry": asdict(self.last),
                "sampled_at": self.last_sample if self.last_sample >= 0 else None,
                "eviction_actions": self.eviction_actions,
                "queued_admissions": self.queued_admissions,
                "rejected_forwards": self.rejected_forwards}


def activation_forecast(config, *, batch, chunk, history, calibrated_peak=None, math_policy=None):
    """Conservative analytical increments; calibration is an explicit separate source."""
    if min(batch, chunk) < 1 or history < 0:
        raise ValueError("invalid activation forecast dimensions")
    if calibrated_peak is not None:
        if calibrated_peak < 0:
            raise ValueError("negative calibrated peak")
        return {"bytes": calibrated_peak, "source": "matching-measured-peak"}
    config = getattr(config,"text_config",config)
    hidden = int(getattr(config,"hidden_size",getattr(config,"n_embd",
                 getattr(config,"model_dim",0))))
    heads = getattr(config,"num_attention_heads",getattr(config,"n_head",
                    getattr(config,"num_query_heads",0)))
    heads = max(heads) if isinstance(heads,(tuple,list)) else int(heads)
    if min(hidden,heads) < 1:
        raise ValueError("activation forecast requires declared model dimensions")
    intermediate = int(getattr(config,"intermediate_size",getattr(config,"inner_dim",
                       max(getattr(config,"ffn_dims",(hidden*4,))))) or hidden*4)
    vocab = int(getattr(config, "vocab_size", 0))
    logits_rows = 1 if math_policy is not None and math_policy.logits == "last_only" else chunk
    amount = batch * chunk * (hidden * 12 + intermediate * 4)
    amount += batch * logits_rows * vocab * 4
    amount += batch * heads * chunk * (history + chunk) * 4
    from mlx_one.engine.optimization_policy import REFERENCE_MATH

    if math_policy is not None and math_policy != REFERENCE_MATH:
        from mlx_one.models.shared.execution import attention_scratch_bound

        dimension = int(getattr(config, "head_dim", hidden // heads))
        # The reference scores and softmax overlap are still a conservative
        # bound for an experimental fused route. Do not infer measured scratch
        # savings or use a calibration from a different math/logits policy.
        amount += attention_scratch_bound(batch=batch, heads=heads, head_dim=dimension,
                                          chunk=chunk, history=history)
    return {"bytes": amount, "source": "conservative-analytical", "measured": False}


def execution_forecast(runtime, config, *, batch, chunk, history, explicit_bytes=0,
                       history_lengths=None, widths=None, phase=None):
    """Use only a fresh calibration with identical execution identity and dimensions."""
    calibrated = None
    dimensions = {"batch": batch, "chunk": chunk, "history": history,
                  "history_lengths": list(history_lengths or [history] * batch),
                  "widths": list(widths or [chunk] * batch),
                  "phase": phase or ("decode" if chunk == 1 else "prefill")}
    registry = getattr(runtime, "qualification_registry", None)
    from mlx_one.engine.optimization_policy import REFERENCE_MATH

    policy = getattr(runtime, "math_policy", REFERENCE_MATH)
    if (registry is not None and policy == REFERENCE_MATH
            and getattr(runtime, "mode", None) != "native_optimized"):
        import json
        from dataclasses import asdict
        from pathlib import Path

        from mlx_one.engine.reporting import native_report
        from mlx_one.engine.selection import compatibility_digest, host_identity, source_identity

        scope = native_report(runtime.bundle, runtime, mode=runtime.mode,
                              hardware=host_identity(), source_revision=source_identity())["scope"]
        scope["cache_precision"] = {"f16": "float16", "bf16": "bfloat16",
                                    "q4_0": "float16,uint8", "q8_0": "float16,uint8"}.get(
                                        runtime.config.key_type)
        for record in registry.records():
            if (record.feature != "activation_peak" or asdict(record.scope) != scope
                    or registry.report(record.feature, record.scope)["status"] not in {
                        "correctness_passed", "performance_qualified"}):
                continue
            proof = json.loads(Path(record.evidence[0].uri).read_text())
            peak = proof.get("activation_peak", {})
            from mlx_one.engine.adapters import execution_contract
            from mlx_one.engine.arena import ARENA_VERSION
            from mlx_one.engine.envelope import EVIDENCE_VERSION
            from mlx_one.engine.paged import KERNEL_VERSION
            from mlx_one.peak_calibration import CALIBRATION_POLICY

            if (proof.get("schema_version") == EVIDENCE_VERSION
                    and proof.get("execution_contract") == execution_contract(
                        runtime.bundle, runtime.plan, runtime.mode)
                    and proof.get("compatibility_digest") == compatibility_digest(runtime)
                    and proof.get("kernel_version") == KERNEL_VERSION
                    and proof.get("arena_version") == ARENA_VERSION
                    and peak.get("policy") == CALIBRATION_POLICY
                    and peak.get("dimensions") == dimensions
                    and isinstance(peak.get("incremental_bytes"), int)
                    and peak["incremental_bytes"] >= 0):
                calibrated = peak["incremental_bytes"]
                break
    result = activation_forecast(config, batch=batch, chunk=chunk, history=history,
                                 calibrated_peak=calibrated,
                                 math_policy=getattr(runtime, "math_policy", None))
    result["bytes"] = (max(result["bytes"], explicit_bytes) if calibrated is not None
                       else result["bytes"] + explicit_bytes)
    if runtime is not None:
        runtime.last_execution_forecast = result
    return result
