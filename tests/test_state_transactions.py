from dataclasses import replace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_planner import build_cache_plan
from mlx_one.engine.cache_specs import CacheConfig, LayerStateSpec
from mlx_one.engine.components import CrossKVComponent
from mlx_one.engine.contracts import LayerExecutionSpec
from mlx_one.engine.transactions import commit_evaluated, release_all


def test_model_defined_per_layer_layout_overrides_global_inference(native):
    native.model.layer_execution_specs=lambda:(
        LayerExecutionSpec(0,'full_kv',4,2,8,6,'rope'),
        LayerExecutionSpec(1,'full_kv',6,3,4,4,'rope'))
    plan=build_cache_plan(native)
    assert [(s.num_kv_heads,s.key_head_dim,s.value_head_dim) for s in plan.layer_specs]==[
        (2,8,6),(3,4,4)]
    assert plan.bytes_per_token()==2*(2*(8+6)+3*8)
    native.model.layer_execution_specs=lambda:(LayerExecutionSpec(1,'full_kv',4,2,8,8,'rope'),)
    with pytest.raises(ValueError,match='align'):
        build_cache_plan(native)


def test_reconciliation_failure_restores_layers_and_reservations(native,monkeypatch):
    runtime=CacheManager(native,CacheConfig())
    handle=runtime.create_handle('r')
    request=ForwardRequest(native,(3,4),handle.bundle.entries,'prefill',(3,4),runtime,handle,
                           request_id='r')
    before=[dict(c.__dict__) for c in request.cache]
    def fail(_):
        raise ValueError('reconcile injection')
    monkeypatch.setattr(runtime,'reconcile',fail)
    with pytest.raises(ValueError,match='reconcile'):
        BatchedDenseRunner(native).execute((request,))
    assert handle.computed_tokens==0
    assert [c.__dict__ for c in request.cache]==before
    assert runtime.stats().reserved_bytes==0
    runtime.close()


def test_transaction_rejects_bad_final_layer_before_any_adoption(native):
    original=native.model.make_cache()
    drafts=CacheBundle(original).clone()
    value=np.ones((1,1,1,2),dtype=np.float16)
    drafts.entries[0].update(value,value)
    request=ForwardRequest(native,(3,),original,'prefill',(3,),request_id='r')
    with pytest.raises(ValueError,match='position'):
        commit_evaluated((request,),(drafts,))
    assert all(c.offset==0 for c in original)
    drafts.release()


def test_cross_kv_decoder_position_is_separate_and_snapshots_isolated(native):
    component=CrossKVComponent('encoder-assets','float16')
    keys=np.ones((1,2,7,4),dtype=np.float16)
    values=np.ones((1,2,7,6),dtype=np.float16)
    component.bind(keys,values)
    component.advance_decoder(3)
    spec=LayerStateSpec(0,'cross_kv',2,4,6,'float16')
    component.validate(spec)
    snapshot=component.clone()
    component.keys[0,0,0,0]=9
    assert snapshot.keys[0,0,0,0]==1 and snapshot.offset==3 and snapshot.encoder_tokens==7
    assert not component.capabilities().can_page
    with pytest.raises(ValueError,match='identity'):
        CrossKVComponent('')
    with pytest.raises(ValueError,match='immutable'):
        component.update(keys,values)
    with pytest.raises(ValueError,match='layout'):
        component.validate(replace(spec,value_head_dim=4))
    release_all((CacheBundle((component,)),CacheBundle((snapshot,))))


def test_dense_kv_keeps_independent_key_and_value_widths(native):
    cache=native.model.make_cache()[0]
    keys=np.ones((1,2,3,4),dtype=np.float16)
    values=np.ones((1,2,3,6),dtype=np.float16)
    a,b=cache.update(keys,values)
    assert a.shape==(1,2,3,4) and b.shape==(1,2,3,6)
    cache.clone().release()


def test_failed_clone_attempts_all_staged_cleanup():
    from mlx_one.engine.cache_errors import CacheCorruptionError

    released=[]
    class Entry:
        def __init__(self,name):
            self.name=name
        def clone(self):
            if self.name=='allocation':
                raise MemoryError('staging allocation failed')
            return self
        def reset(self):
            released.append(self.name)
            if self.name=='corrupt':
                raise CacheCorruptionError('cleanup corruption')
    with pytest.raises(CacheCorruptionError,match='cleanup corruption'):
        CacheBundle(tuple(Entry(name) for name in ('corrupt','healthy','allocation'))).clone()
    assert released==['corrupt','healthy']
