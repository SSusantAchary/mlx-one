"""Service leases and device ownership preserve existing text result semantics."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_dense_batching import native as native

from mlx_one.engine.model_pool import ModelPool
from mlx_one.engine.service import InferenceService


@pytest.mark.parametrize('mode',[None,'experimental_dense_batch'])
def test_service_reuses_engine_and_executes_on_device_owner(native,monkeypatch,mode):
    from mlx_one.engine.device_executor import DeviceExecutor

    observed=[]
    original=type(native.model).__call__
    def forward(self,*args,**kwargs):
        observed.append(DeviceExecutor.shared().is_owner)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(type(native.model),'__call__',forward)
    service=InferenceService()
    service.register_text('test',native,inference_mode=mode)
    first=service.generate('test','hello',max_tokens=2,temperature=0)
    second=service.generate('test','hello',max_tokens=2,temperature=0)
    assert first.text==second.text and all(observed)
    assert service.pool.stats()['resident_models']==1
    assert service.pool.stats()['active_leases']==0
    service.close()
    assert service.pool.stats()['resident_models']==0


def test_closed_stream_releases_lease_and_drain_cancels_actor(native):
    service=InferenceService()
    service.register_text('test',native,inference_mode='experimental_dense_batch')
    stream=service.stream('test','hello',max_tokens=100,temperature=0)
    next(stream)
    assert service.pool.stats()['active_leases']==1
    stream.close()
    service.unload('test',timeout=2)
    assert service.pool.stats()['active_leases']==0
    assert service.pool.stats()['resident_models']==0
    service.close()


def test_server_unload_waits_for_live_request_lease(native,monkeypatch):
    from mlx_one.server import model_manager

    manager=model_manager.ModelManager()
    monkeypatch.setattr(model_manager,'load_text_model',lambda *args,**kwargs:native)
    manager.load('test')
    lease=manager.model_lease()
    with ThreadPoolExecutor(1) as caller:
        unloading=caller.submit(manager.unload)
        assert not unloading.done()
        lease.release()
        unloading.result(2)
    with pytest.raises(RuntimeError,match='no model'):
        manager.current_model()
    manager.load('test')
    assert manager._bundle_lease.runtime.epoch==2
    manager.unload()


def test_service_capacity_failure_drains_loaded_engine(native):
    from mlx_one.engine.cache_errors import CacheCapacityError

    service=InferenceService(pool=ModelPool(budget_bytes=1))
    service.register_text('test',native,inference_mode='experimental_dense_batch')
    with pytest.raises(CacheCapacityError):
        service.generate('test','hello',max_tokens=1)
    assert service.pool.stats()['active_leases']==0
    service.close()
    assert not any(t.name=='mlx-one-inference' and t.is_alive() for t in threading.enumerate())
