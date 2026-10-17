"""Observed headroom and one safe device-owned reclaim before capacity rejection."""
from __future__ import annotations

import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

from mlx_one.engine.cache_errors import CacheCapacityError


def _sample(runtime, *, force=False):
    from mlx_one.engine.pressure import sample_telemetry

    now = time.monotonic()
    if (force or runtime._capacity_snapshot is None
            or now - runtime._capacity_sampled_at >= 1):
        previous = runtime._capacity_snapshot
        runtime._capacity_snapshot = sample_telemetry()
        before = previous.swap_bytes if previous is not None else None
        after = runtime._capacity_snapshot.swap_bytes
        runtime._swap_delta_bytes = (after - before
                                     if before is not None and after is not None else None)
        runtime._capacity_sampled_at = time.monotonic()
    snapshot = runtime._capacity_snapshot
    # Tensor allocator changes are visible at every evaluated forward boundary.
    mx = sys.modules.get('mlx.core')
    observed = {}
    for field, method in (('mlx_active_bytes', 'get_active_memory'),
                          ('mlx_cached_bytes', 'get_cache_memory'),
                          ('mlx_peak_bytes', 'get_peak_memory')):
        function = getattr(mx, method, None)
        if callable(function):
            try:
                observed[field] = int(function())
            except (RuntimeError, ValueError):
                pass
    if observed:
        snapshot = replace(snapshot, **observed)
        runtime._capacity_snapshot = snapshot
    return snapshot


def _diagnostic(runtime, amount, transient, snapshot, attempt):
    stats = runtime.stats()
    pressure_config = getattr(runtime.pressure, 'config', None)
    controlled = pressure_config is not None and pressure_config.enabled
    fraction = min(.75, pressure_config.memory_fraction) if controlled else .75
    limits = [value for value in (
        int(snapshot.installed_bytes * fraction) if snapshot.installed_bytes else None,
        snapshot.recommended_bytes,
        pressure_config.process_ceiling_bytes if controlled else None)
        if value is not None]
    ceiling = min(limits) if limits else None
    allocator = (snapshot.mlx_active_bytes or 0) + (snapshot.mlx_cached_bytes or 0)
    owner = runtime.model_bytes + stats.physical_bytes
    working = max(snapshot.rss_bytes or 0, allocator, owner)
    requested = [dict(handle.metadata.get('admission_request', {}),
                      request_id=handle.request_id, handle_id=handle.handle_id,
                      key_type=handle.metadata.get('key_type'),
                      value_type=handle.metadata.get('value_type'),
                      allocated_bytes=handle.bundle.allocated_nbytes,
                      reserved_bytes=handle.reserved_bytes)
                 for handle in runtime._handles.values()]
    kv_dtypes = set()
    component_dtypes = set()
    for handle in runtime._handles.values():
        for spec, entry in zip(runtime.plan.state_specs, handle.bundle.entries, strict=True):
            fields = getattr(entry, '__dict__', {})
            for name in ('keys', 'values', '_keys', '_values', 'conv_state', 'recurrent_state'):
                value = fields.get(name)
                if hasattr(value, 'dtype'):
                    observed = str(value.dtype).rsplit('.', 1)[-1]
                    if spec.kind in {'full_kv', 'sliding_kv'}:
                        kv_dtypes.add(observed)
                    else:
                        component_dtypes.add((spec.kind, name, observed))
    return {
        'schema_version': 'native-admission-v1',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_id': str(runtime.bundle.model_id), 'revision': runtime.bundle.revision,
        'backend': runtime.config.backend, 'execution_mode': getattr(runtime, 'mode', None),
        'weight_precision': dict(getattr(runtime.bundle, 'quantization', None) or {}),
        'execution_dtype': getattr(runtime.bundle.model, '_mlx_one_activation_dtype', None),
        'attention_kv_precision': runtime.precision_configuration(),
        'requests': requested, 'transient': transient, 'attempt': attempt, 'stage': 'host',
        'attention_kv_tensor_dtypes': sorted(kv_dtypes),
        'component_state_dtypes': sorted(component_dtypes),
        'status': 'bounded' if ceiling is not None else 'unverified',
        'ceiling_bytes': ceiling, 'working_set_bytes': working, 'owner_bytes': owner,
        'model_bytes': runtime.model_bytes, 'cache_allocated_bytes': stats.allocated_bytes,
        'cache_physical_bytes': stats.physical_bytes, 'cache_logical_bytes': stats.logical_bytes,
        'reserved_bytes': stats.reserved_bytes, 'active_handles': stats.active_handles,
        'retained_prefix_bytes': stats.prefix_bytes,
        'reclaimable_pool_bytes': stats.retained_free_bytes,
        'evictable_bytes': stats.evictable_bytes,
        'requested_bytes': amount, 'predicted_transient_bytes': amount if transient else 0,
        'forecast': dict(getattr(runtime, 'last_execution_forecast', {})) if transient else None,
        'telemetry': asdict(snapshot),
        'swap_delta_bytes': getattr(runtime, '_swap_delta_bytes', None),
        'swap_scope': 'host-wide, not uniquely attributed to this request',
        'host_sample_age_seconds': max(time.monotonic() - runtime._capacity_sampled_at, 0),
        'formula': 'max(RSS, MLX active+cached, owner) + outstanding + requested',
        'projected_bytes': working + stats.reserved_bytes + amount,
    }


def _publish(runtime, record):
    from mlx_one.engine.observation import current_observer

    runtime.capacity_diagnostics = record
    observer = current_observer()
    if observer is not None and getattr(observer, 'capture_admission', False):
        observer({'kind': 'admission', 'diagnostics': record})


def _can_reclaim(runtime, record):
    from mlx_one.engine.device_executor import DeviceExecutor

    owner = DeviceExecutor._shared
    mx = sys.modules.get('mlx.core')
    return (owner is not None and owner.is_owner and runtime._transient_bytes == 0
            and callable(getattr(mx, 'synchronize', None))
            and callable(getattr(mx, 'clear_cache', None))
            and any(value > 0 for value in (
                record['telemetry']['mlx_cached_bytes'] or 0,
                record['reclaimable_pool_bytes'], record['evictable_bytes'])))


def check_host_capacity(runtime, amount, *, transient=False):
    """Caller holds the manager lock; active handles and reservations remain intact."""
    snapshot = _sample(runtime, force=not transient)
    for attempt in range(2):
        record = _diagnostic(runtime, amount, transient, snapshot, attempt)
        fits = record['ceiling_bytes'] is None or (
            record['projected_bytes'] <= record['ceiling_bytes'])
        critical = snapshot.os_pressure == 'critical'
        if fits and not critical:
            record.update(decision='admit', rejection_reason=None, retryable=None)
            if record['ceiling_bytes'] is None:
                record['reason'] = 'host ceiling unavailable; configured cache limit only'
                record['cache_limit_bytes'] = runtime.config.memory_budget_bytes
            _publish(runtime, record)
            return
        reason = ('request rejected by critical host memory pressure' if critical else
                  'request exceeds conservative host memory headroom')
        retryable = not transient and (record['ceiling_bytes'] is None or
                                      runtime.model_bytes + amount <= record['ceiling_bytes'])
        possible = (record['ceiling_bytes'] is None or
                    runtime.model_bytes + amount <= record['ceiling_bytes'])
        if attempt == 0 and possible and _can_reclaim(runtime, record):
            record.update(decision='reclaim', rejection_reason=reason, retryable=retryable)
            _publish(runtime, record)
            try:
                mx = sys.modules['mlx.core']
                mx.synchronize()
                # Index eviction drops only its references; active borrowers retain theirs.
                runtime.clear_prefixes()
                runtime.block_pool.discard_free()
                mx.clear_cache()
                runtime._capacity_snapshot = None
                snapshot = _sample(runtime, force=True)
                if runtime.pressure is not None:
                    runtime.pressure._refresh_after_reclaim(snapshot)
            except Exception as error:
                record = dict(record, decision="reject", stage="reclaim", retryable=False,
                              rejection_reason=str(error), error_type=type(error).__name__)
                _publish(runtime, record)
                raise
            continue
        record.update(decision='reject', rejection_reason=reason, retryable=retryable)
        _publish(runtime, record)
        raise CacheCapacityError(reason, retryable=retryable)
