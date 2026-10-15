import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.model_pool import ModelPool


@pytest.fixture
def pool():
    executor=DeviceExecutor()
    value=ModelPool(executor=executor,budget_bytes=100)
    yield value
    value.close(timeout=1)
    executor.close()


def test_concurrent_acquisition_single_flight_and_draining_epoch(pool):
    entered,release=threading.Event(),threading.Event()
    loads=[]
    def load():
        loads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return object()
    with ThreadPoolExecutor(2) as callers:
        a=callers.submit(pool.acquire,'a',load,estimated_bytes=40)
        assert entered.wait(2)
        b=callers.submit(pool.acquire,'a',load,estimated_bytes=40)
        release.set()
        first,second=a.result(2),b.result(2)
    assert len(loads)==1 and first.runtime is second.runtime
    assert pool.stats()['active_leases']==2
    with pytest.raises(TimeoutError):
        pool.unload('a',timeout=0)
    first.release()
    first.release()
    second.release()
    pool.unload('a')
    newer=pool.acquire('a',lambda:object(),estimated_bytes=40)
    assert newer.runtime.epoch==2
    newer.release()
    assert pool.stats()['reserved_bytes']==0


def test_active_and_pinned_models_cannot_be_evicted(pool):
    lease=pool.acquire('a',lambda:'a',estimated_bytes=60,pinned=True)
    with pytest.raises(CacheCapacityError):
        pool.acquire('b',lambda:'b',estimated_bytes=50)
    lease.release()
    with pytest.raises(CacheCapacityError):
        pool.acquire('b',lambda:'b',estimated_bytes=50)
    pool.pin('a',False)
    with pool.acquire('b',lambda:'b',estimated_bytes=50) as runtime:
        assert runtime.value=='b'
    assert pool.stats()['resident_models']==1


def test_failed_load_and_observed_capacity_release_reservations(pool):
    def fail():raise ValueError('bad weights')
    with pytest.raises(ValueError):
        pool.acquire('a',fail,estimated_bytes=40)
    disposed=[]
    with pytest.raises(CacheCapacityError):
        pool.acquire('a',lambda:'oversized',estimated_bytes=40,measure=lambda _:101,
                     dispose=disposed.append)
    assert disposed==['oversized']
    assert pool.stats()==dict(resident_models=0,loading_models=0,resident_bytes=0,
                             reserved_bytes=0,active_leases=0)
    lease=pool.acquire('a',lambda:'valid',estimated_bytes=40)
    lease.release()
    with pytest.raises(RuntimeError):
        _ = lease.runtime


def test_ttl_and_cleanup_attempt_all_releases():
    clock=[0]
    executor=DeviceExecutor()
    pool=ModelPool(executor=executor,max_models=2,idle_ttl=5,clock=lambda:clock[0])
    disposed=[]
    def dispose(value):
        disposed.append(value)
        if value=='a':
            raise ValueError('corrupt release')
    for name in ('a','b'):
        pool.acquire(name,lambda n=name:n,dispose=dispose).release()
    clock[0]=6
    with pytest.raises(ValueError):
        pool.evict_idle()
    assert disposed==['a','b'] and pool.stats()['resident_models']==0
    pool.close()
    executor.close()


def test_reload_cannot_overtake_device_disposal(pool):
    entered,release=threading.Event(),threading.Event()
    def dispose(value):
        entered.set()
        assert release.wait(2)
    pool.acquire('a',lambda:'old',dispose=dispose).release()
    with ThreadPoolExecutor(1) as caller:
        unloading=caller.submit(pool.unload,'a')
        assert entered.wait(2)
        try:
            with pytest.raises(RuntimeError,match='draining'):
                # A new loader must never queue behind an old epoch disposal.
                pool.acquire('a',lambda:'new')
        finally:
            release.set()
        unloading.result(2)
