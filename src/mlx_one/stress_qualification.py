"""Bounded resident allocation and full owner/allocator cleanup reconciliation."""
from __future__ import annotations

import os
import threading

MEMORY_METRICS = ("rss_bytes", "mlx_active_bytes", "mlx_cached_bytes", "arena_allocated_bytes",
                  "retained_free_bytes", "prefix_pinned_bytes", "active_handles",
                  "reserved_bytes", "queued_events", "active_jobs", "pending_jobs", "actor_threads")


class TouchedMemory:
    """Real resident pages, bounded independently of controller configuration."""
    def __init__(self, limit_bytes, *, allocator=bytearray, page_bytes=None):
        if limit_bytes < 1:
            raise ValueError("positive allocation cap required")
        self.limit_bytes = limit_bytes
        self.allocator = allocator
        self.page_bytes = page_bytes or os.sysconf("SC_PAGE_SIZE")
        self.buffers, self.allocated_bytes = [], 0

    def grow(self, amount):
        if amount < 0 or self.allocated_bytes + amount > self.limit_bytes:
            raise MemoryError("pressure allocation cap exceeded")
        remaining = amount
        staged = []
        try:
            while remaining:
                count = min(remaining, 16 * 1024 * 1024)
                buffer = self.allocator(count)
                for index in range(0, count, self.page_bytes):
                    buffer[index] = 1
                staged.append(buffer)
                remaining -= count
        except BaseException:
            staged.clear()
            raise
        self.buffers.extend(staged)
        self.allocated_bytes += amount

    def close(self):
        self.buffers.clear()
        self.allocated_bytes = 0


def cleanup_snapshot(engine, scheduler):
    from mlx_one.engine.pressure import sample_telemetry

    telemetry, cache = sample_telemetry(), engine.cache_stats()
    runtime = getattr(engine, "_cache_runtime", None)
    active = getattr(scheduler, "_active", None)
    if active is not None:
        active = [*active, *getattr(scheduler, "_paused", [])]
    pending = getattr(scheduler, "_pending", None)
    return {"rss_bytes": telemetry.rss_bytes,
            "mlx_active_bytes": telemetry.mlx_active_bytes,
            "mlx_cached_bytes": telemetry.mlx_cached_bytes,
            "arena_allocated_bytes": runtime.block_pool.allocated_nbytes if runtime else None,
            "retained_free_bytes": cache.get("retained_free_bytes"),
            "prefix_pinned_bytes": cache.get("pinned_bytes"),
            "active_handles": cache.get("active_handles"),
            "reserved_bytes": cache.get("reserved_bytes"),
            "queued_events": sum(j.output.qsize() for j in active) if active is not None else None,
            "active_jobs": len(active) if active is not None else None,
            "pending_jobs": len(pending) if pending is not None else None,
            "actor_threads": sum(t.name == "mlx-one-inference" and t.is_alive()
                                 for t in threading.enumerate())}


def reconcile_memory(initial, final, tolerances):
    if (not tolerances or set(tolerances) != set(MEMORY_METRICS)
            or any(not isinstance(v, int) or v < 0 for v in tolerances.values())):
        return {"status": "pending", "reason": "declare a nonnegative tolerance for every metric"}
    missing = [name for name in MEMORY_METRICS
               if initial.get(name) is None or final.get(name) is None]
    if missing:
        return {"status": "pending", "missing_telemetry": missing}
    deltas = {name: final[name] - initial[name] for name in MEMORY_METRICS}
    checks = {name: delta <= tolerances[name] for name, delta in deltas.items()}
    for name in ("active_handles", "reserved_bytes", "queued_events",
                 "active_jobs", "pending_jobs"):
        checks[name] = final[name] == 0
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "deltas": deltas, "initial": initial, "final": final, "tolerances": tolerances}


async def bounded_pressure_recovery(client, bundle, engine, scheduler, profile, output):
    """Actual resident allocation plus real admission and pre-forward failures."""
    import asyncio
    import gc
    import time
    from dataclasses import replace

    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.inference_baseline_worker import build_messages, stream_request, write_json
    from mlx_one.text.schemas import TextGenerationOptions

    controller, runtime = engine._pressure, engine._cache_runtime
    result = {"status": "pending", "pressure_scope": "bounded-worker-rss",
              "full_device_os_swap_stress": "not_tested"}
    if (controller is None or not controller.config.enabled
            or profile.pressure_allocation_bytes is None
            or profile.pressure_test_ceiling_bytes is None):
        result["reason"] = "enabled pressure control, test ceiling and allocation cap are required"
        write_json(output / "pressure-recovery.json", result)
        return result
    original = controller.config
    ceiling = profile.pressure_test_ceiling_bytes
    memory = TouchedMemory(profile.pressure_allocation_bytes)
    states, job, queued = [], None, None
    async def sample():
        await asyncio.sleep(max(original.sample_interval_s * 1.1, .01))
        controller.poll()
        states.append({**controller.stats(), "touched_bytes": memory.allocated_bytes})
        return controller.state
    async def recover():
        memory.close()
        gc.collect()
        controller.config = original
        await sample()
        await sample()
    async def reach(target):
        # Bounded steps; each is followed by a real background telemetry sample.
        for _ in range(12):
            if controller.state == target:
                return True
            if target == "warning" and controller.state == "critical":
                return False
            rss = controller.last.rss_bytes
            if rss is None:
                return False
            goal = int(ceiling * (.95 if target == "warning" else 1.02))
            amount = max(goal - rss, 4096)
            if amount > memory.limit_bytes - memory.allocated_bytes:
                return False
            memory.grow(amount)
            await sample()
        return False
    messages, _ = build_messages(bundle, min(profile.context_tokens, 2048),
                                 suffix="\nContinue writing numbered facts without stopping.")
    try:
        await sample()
        if (controller.ceiling is None or ceiling > controller.ceiling * .95
                or controller.working_set() >= ceiling * original.warning_fraction):
            result["reason"] = ("test ceiling lacks safe initial headroom "
                                "or exceeds the device limit")
            return result
        baseline = await stream_request(client, profile, messages, phase="pressure",
                                        group="before", index=0)
        controller.config = replace(original, process_ceiling_bytes=ceiling)
        await sample()
        if not await reach("warning"):
            result["reason"] = "resident allocation did not demonstrate warning within its cap"
            return result
        before_queued = controller.queued_admissions
        queued = asyncio.create_task(stream_request(client, profile, messages,
                                     phase="pressure", group="warning-queued", index=0))
        await sample()
        warning_queued = not queued.done() and controller.queued_admissions > before_queued
        await recover()
        resumed = await asyncio.wait_for(queued, timeout=profile.timeout_seconds)
        # Park a real native generation job using its existing bounded output queue.
        options = TextGenerationOptions(max_tokens=max(profile.output_tokens, 128), temperature=0)
        job = scheduler._enqueue(lambda cancel: engine.stream(model=profile.model_id,
            messages=messages, options=options, cancel=cancel, reasoning="off",
            cache_type_k=profile.cache_type, cache_type_v=profile.cache_type))
        deadline = time.monotonic() + min(profile.timeout_seconds, 60)
        while not job.output.full() and not job.terminal.done() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        if not job.output.full():
            result["reason"] = "native generation ended before a forward could be parked"
            return result
        controller.config = replace(original, process_ceiling_bytes=ceiling)
        await sample()
        if not await reach("critical"):
            result["reason"] = "resident allocation did not demonstrate critical within its cap"
            return result
        rejected_before = controller.rejected_forwards
        while not job.output.empty():
            job.output.get_nowait()
        deadline = time.monotonic() + 10
        while not job.terminal.done() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        error = job.terminal.exception() if job.terminal.done() else None
        typed = isinstance(error, CacheCapacityError) and not error.retryable
        await recover()
        after = await stream_request(client, profile, messages, phase="pressure",
                                     group="recovered-peer", index=0)
        checks = {"resident_allocation_observed": any(
                      s.get("touched_bytes", 0) > 0 for s in states),
                  "warning_admission_queued": warning_queued,
                  "critical_native_forward_rejected": typed
                    and controller.rejected_forwards > rejected_before,
                  "bounded_output_queue": job.output.maxsize == profile.output_buffer_events,
                  "peer_progress_after_recovery": after["status"] == "completed",
                  "output_parity": baseline.get("output_sha256") is not None
                    and baseline.get("output_sha256") == resumed.get("output_sha256")
                    == after.get("output_sha256"),
                  "reservations_released": runtime.stats().reserved_bytes == 0,
                  "recovered": controller.state == "normal"}
        result.update(status="passed" if all(checks.values()) else "failed", checks=checks,
                      typed_error=type(error).__name__ if error else None,
                      declared_allocation_cap=memory.limit_bytes, declared_test_ceiling=ceiling,
                      baseline=baseline, recovered=after)
        return result
    finally:
        if job is not None:
            job.cancel.set()
            while not job.output.empty():
                job.output.get_nowait()
        if queued is not None and not queued.done():
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
        await recover()
        result["states"] = states
        write_json(output / "pressure-recovery.json", result)


async def direct_stall_probe(client, bundle, engine, scheduler, profile):
    """A terminal Future must finish while the 32-event consumer stays paused."""
    import asyncio
    import time

    from mlx_one.inference_baseline_worker import build_messages, stream_request
    from mlx_one.text.schemas import TextGenerationOptions

    messages, _ = build_messages(bundle, min(profile.context_tokens, 2048),
                                 suffix="\nWrite a long numbered list of detailed facts.")
    job = scheduler._enqueue(lambda cancel: engine.stream(model=profile.model_id,
        messages=messages, options=TextGenerationOptions(max_tokens=max(profile.output_tokens, 128),
        temperature=0), cancel=cancel, reasoning="off",
        cache_type_k=profile.cache_type, cache_type_v=profile.cache_type))
    try:
        deadline = time.monotonic() + min(profile.timeout_seconds, 60)
        while not job.output.full() and not job.terminal.done() and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        if not job.output.full():
            return {"status": "pending", "reason": "generation ended before output buffer filled"}
        peer = asyncio.create_task(stream_request(client, profile, messages, phase="probe",
                                                  group="stall-peer", index=0))
        deadline = time.monotonic() + profile.output_stall_timeout_s + 2
        while not job.terminal.done() and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        error = job.terminal.exception() if job.terminal.done() else None
        peer_result = await asyncio.wait_for(peer, timeout=profile.timeout_seconds)
        checks = {"bounded_output": job.output.qsize() <= profile.output_buffer_events,
                  "separate_terminal": job.terminal.done() and not job.output.empty(),
                  "stall_terminated": isinstance(error, TimeoutError) and "stalled" in str(error),
                  "peer_progress": peer_result["status"] == "completed"}
        return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                "output_events": job.output.qsize(),
                "stall_timeout_s": profile.output_stall_timeout_s,
                "peer": peer_result}
    finally:
        job.cancel.set()
        while not job.output.empty():
            job.output.get_nowait()
