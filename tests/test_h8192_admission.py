"""Deterministic admission/reclamation rules, independent of native host memory."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_cache_runtime import Model, bundle, install_fake_mlx

from mlx_one.core.cache import ConvCache, KVCache
from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.pressure import PressureTelemetry


@pytest.fixture
def admission(tmp_path, monkeypatch):
    import sys

    install_fake_mlx(monkeypatch)
    state = {'telemetry': PressureTelemetry(rss_bytes=500, installed_bytes=2000,
             mlx_active_bytes=400, mlx_cached_bytes=100, recommended_bytes=1500),
             'reads': 0, 'clears': 0, 'syncs': 0}
    def sample():
        state['reads'] += 1
        return state['telemetry']
    def clear():
        state['clears'] += 1
        current = state['telemetry']
        state['telemetry'] = replace(current, rss_bytes=current.mlx_active_bytes,
                                     mlx_cached_bytes=0)
    def sync():
        state['syncs'] += 1
    core = sys.modules['mlx.core']
    core.clear_cache, core.synchronize = clear, sync
    monkeypatch.setattr('mlx_one.engine.pressure.sample_telemetry', sample)
    monkeypatch.setattr(DeviceExecutor, '_shared', SimpleNamespace(is_owner=True))
    runtime = CacheManager(bundle(tmp_path), CacheConfig(memory_budget_bytes=10000))
    yield runtime, state, core
    runtime.close()


@pytest.mark.parametrize('amount', [0, 999, 1000])
def test_admission_below_and_exactly_at_ceiling(admission, amount):
    runtime, state, _ = admission
    with runtime.transient_reservation(amount):
        assert runtime.stats().reserved_bytes == amount
    assert runtime.stats().reserved_bytes == 0
    assert state['clears'] == 0


def test_reclaim_unused_allocator_cache_before_one_retry(admission):
    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=900,
                                 mlx_active_bytes=200, mlx_cached_bytes=700)
    with runtime.transient_reservation(700):
        assert runtime.stats().reserved_bytes == 700
    assert state['clears'] == state['syncs'] == 1
    assert state['reads'] == 2
    assert runtime.stats().reserved_bytes == 0


def test_failed_reclaim_rejects_after_one_attempt(admission, monkeypatch):
    runtime, state, core = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=1400,
                                 mlx_active_bytes=1400, mlx_cached_bytes=50)
    monkeypatch.setattr(core, 'clear_cache', lambda: state.__setitem__('clears',
                                                                    state['clears'] + 1))
    with pytest.raises(CacheCapacityError), runtime.transient_reservation(200):
        pass
    assert state['clears'] == 1 and state['reads'] == 2
    assert runtime.stats().reserved_bytes == 0


def test_no_reclaimable_buffers_reject_without_retry(admission):
    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=1400,
                                 mlx_active_bytes=1400, mlx_cached_bytes=0)
    with pytest.raises(CacheCapacityError), runtime.transient_reservation(200):
        pass
    assert state['clears'] == 0 and state['reads'] == 1


def test_reclamation_cannot_run_off_device_owner(admission, monkeypatch):
    runtime, state, _ = admission
    monkeypatch.setattr(DeviceExecutor, '_shared', SimpleNamespace(is_owner=False))
    state['telemetry'] = replace(state['telemetry'], rss_bytes=900,
                                 mlx_active_bytes=200, mlx_cached_bytes=700)
    with pytest.raises(CacheCapacityError), runtime.transient_reservation(700):
        pass
    assert state['clears'] == state['syncs'] == 0


def test_new_request_refreshes_stale_host_telemetry(admission):
    runtime, state, _ = admission
    runtime._host_capacity(100)
    state['telemetry'] = replace(state['telemetry'], rss_bytes=1400,
                                 mlx_active_bytes=1400, mlx_cached_bytes=0)
    with pytest.raises(CacheCapacityError):
        runtime._host_capacity(200)
    assert state['reads'] == 2


def test_critical_os_pressure_cannot_be_bypassed(admission):
    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], os_pressure='critical', mlx_cached_bytes=0)
    with pytest.raises(CacheCapacityError), runtime.transient_reservation(1):
        pass
    assert runtime.stats().reserved_bytes == 0


def test_outstanding_reservations_are_counted_once_and_released(admission):
    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], mlx_active_bytes=500, mlx_cached_bytes=0)
    first, second = runtime.create_handle('first'), runtime.create_handle('second')
    runtime.reserve(first, tokens=1, requested_bytes=600)
    with pytest.raises(CacheCapacityError):
        runtime.reserve(second, tokens=1, requested_bytes=500)
    assert first.reserved_bytes == 600 and second.reservation is None
    runtime.release(first)
    reservation = runtime.reserve(second, tokens=1, requested_bytes=500)
    runtime.release_reservation(reservation)
    runtime.release_reservation(reservation)
    runtime.release(second)
    assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0
    assert not runtime._reservations


def test_prior_request_cleanup_allows_later_admission(admission):
    runtime, _, _ = admission
    for number in range(3):
        handle = runtime.create_handle(str(number))
        runtime.reserve(handle, tokens=1, requested_bytes=800)
        runtime.release(handle)
        assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0


def test_dense_hybrid_carries_are_reserved_without_enabling_apc(tmp_path, monkeypatch):
    install_fake_mlx(monkeypatch)
    model = Model(caches=2)
    model.make_cache = lambda: (KVCache(), ConvCache(3))
    runtime = CacheManager(bundle(tmp_path, model=model, architecture='lfm2'))
    handle = runtime.create_handle('hybrid')
    try:
        expected = runtime.plan.bytes_per_token() * 256 + runtime.plan.carry_bytes()
        assert runtime.estimate_growth(handle, prompt_tokens=128, max_new_tokens=128) == expected
        assert runtime.plan.carry_bytes() > 0
        assert not runtime.hybrid_state  # Dense hybrid prefix restoration remains unavailable.
    finally:
        runtime.close()


def test_reclamation_preserves_live_state(admission):
    import numpy as np

    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=12000,
        installed_bytes=20000, recommended_bytes=15000, mlx_active_bytes=2000,
        mlx_cached_bytes=10000)
    runtime.config = replace(runtime.config, memory_budget_bytes=20000)
    handle = runtime.create_handle('live')
    value = np.ones((1,2,1,4), dtype=np.float16)
    handle.bundle.entries[0].update(value, value)
    before = handle.bundle.entries[0]._keys.copy()
    with runtime.transient_reservation(4000):
        np.testing.assert_array_equal(handle.bundle.entries[0]._keys, before)
        assert runtime.stats().active_handles == 1
    assert state['clears'] == 1
    runtime.release(handle)
    assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0


def test_concurrent_reservations_cannot_spend_the_same_headroom(admission):
    from concurrent.futures import ThreadPoolExecutor

    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=500,
                                 mlx_active_bytes=500, mlx_cached_bytes=0)
    handles = [runtime.create_handle(str(index)) for index in range(2)]
    def reserve(handle):
        try:
            runtime.reserve(handle, tokens=1, requested_bytes=600)
            return True
        except CacheCapacityError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, handles)) == [False, True]
    assert runtime.stats().reserved_bytes == 600
    for handle in handles:
        runtime.release(handle)
    assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0


def test_diagnostics_distinguish_physical_observations_and_owned_state(admission):
    runtime, _, _ = admission
    with runtime.transient_reservation(100):
        record = runtime.capacity_diagnostics
        assert record['projected_bytes'] == 600
        assert record['telemetry']['rss_bytes'] == 500
        assert record['telemetry']['mlx_active_bytes'] == 400
        assert record['telemetry']['mlx_cached_bytes'] == 100
        assert record['decision'] == 'admit'
        assert record['predicted_transient_bytes'] == 100
        assert record['owner_bytes'] == record['cache_physical_bytes'] == 0


def test_nested_execution_storage_cannot_be_reclaimed(admission):
    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=900,
                                 mlx_active_bytes=200, mlx_cached_bytes=700)
    with runtime.transient_reservation(500):
        with pytest.raises(CacheCapacityError), runtime.transient_reservation(200):
            pass
        assert runtime.stats().reserved_bytes == 500
    assert state['clears'] == 0 and runtime.stats().reserved_bytes == 0


def test_permanently_oversized_request_does_not_evict_for_no_benefit(admission):
    runtime, state, _ = admission
    handle = runtime.create_handle('too-large')
    with pytest.raises(CacheCapacityError) as failure:
        runtime.reserve(handle, tokens=1, requested_bytes=1600)
    assert not failure.value.retryable and state['clears'] == 0
    runtime.release(handle)


def test_configured_budget_rejection_has_final_diagnostics(admission):
    runtime, _, _ = admission
    runtime.config = replace(runtime.config, memory_budget_bytes=10)
    with pytest.raises(CacheCapacityError), runtime.transient_reservation(20):
        pass
    assert runtime.capacity_diagnostics['decision'] == 'reject'
    assert runtime.capacity_diagnostics['stage'] == 'cache-budget'
    assert not runtime.capacity_diagnostics['retryable']
    assert runtime.stats().reserved_bytes == 0


def test_explicit_fp16_conv_carry_matches_its_tensor_layout(tmp_path, monkeypatch):
    import numpy as np

    install_fake_mlx(monkeypatch)
    model = Model(caches=2)
    model._mlx_one_activation_dtype = 'float16'
    model.make_cache = lambda: (KVCache(), ConvCache(3))
    runtime = CacheManager(bundle(tmp_path, model=model, architecture='lfm2'))
    handle = runtime.create_handle('existing')
    values = np.ones((1,2,5,4), dtype=np.float16)
    handle.bundle.entries[0].update(values, values)
    handle.bundle.entries[1].update(np.ones((1,5,16), dtype=np.float16))
    handle.prefix_tokens_reused = 5
    try:
        assert runtime.plan.carry_bytes() == handle.bundle.entries[1].allocated_nbytes == 64
        assert runtime.estimate_growth(handle, prompt_tokens=5, max_new_tokens=10) == 0
    finally:
        runtime.close()


@pytest.mark.parametrize('backend', ['dense', 'block', 'paged', 'hybrid'])
def test_backend_memory_policy_keeps_the_same_ceiling(tmp_path, monkeypatch, backend):
    import sys

    install_fake_mlx(monkeypatch)
    state = {'telemetry': PressureTelemetry(rss_bytes=900, installed_bytes=2000,
             mlx_active_bytes=200, mlx_cached_bytes=700), 'clears': 0}
    def clear():
        state['clears'] += 1
        state['telemetry'] = replace(state['telemetry'], rss_bytes=200, mlx_cached_bytes=0)
    core = sys.modules['mlx.core']
    core.clear_cache, core.synchronize = clear, lambda: None
    monkeypatch.setattr('mlx_one.engine.pressure.sample_telemetry', lambda: state['telemetry'])
    monkeypatch.setattr(DeviceExecutor, '_shared', SimpleNamespace(is_owner=True))
    runtime = CacheManager(bundle(tmp_path), CacheConfig(backend=backend))
    try:
        with runtime.transient_reservation(700):
            assert runtime.capacity_diagnostics['ceiling_bytes'] == 1500
        assert state['clears'] == 1
    finally:
        runtime.close()


@pytest.mark.parametrize('on_owner', [False, True])
def test_pressure_shedding_uses_the_device_owner(admission, monkeypatch, on_owner):
    from mlx_one.engine.pressure import MemoryPressureController

    runtime, state, _ = admission
    state['telemetry'] = replace(state['telemetry'], rss_bytes=1400,
                                 mlx_active_bytes=400, mlx_cached_bytes=1000)
    monkeypatch.setattr(DeviceExecutor, '_shared', SimpleNamespace(is_owner=on_owner))
    controller = MemoryPressureController(telemetry=lambda: state['telemetry'], clock=lambda: 0)
    controller.shed(runtime)
    assert state['clears'] == state['syncs'] == int(on_owner)
    assert controller.eviction_actions == int(on_owner)


def test_controller_headroom_is_refreshed_after_reclamation(admission):
    from mlx_one.engine.pressure import MemoryPressureConfig, MemoryPressureController

    runtime, state, _ = admission
    runtime.pressure = MemoryPressureController(
        MemoryPressureConfig(process_ceiling_bytes=1000),
        telemetry=lambda: state['telemetry'], clock=lambda: 0)
    runtime.pressure.poll()
    state['telemetry'] = replace(state['telemetry'], rss_bytes=900,
                                 mlx_active_bytes=200, mlx_cached_bytes=700)
    with runtime.transient_reservation(700):
        assert runtime.capacity_diagnostics['ceiling_bytes'] == 1000
    assert state['clears'] == 1
