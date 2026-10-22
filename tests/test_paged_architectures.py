"""Backend-free heterogeneous-layer paging, transactional adoption and fail-closed windows."""
from dataclasses import replace

import numpy as np
import pytest
from test_dense_batching import native as native
from test_paged_hybrid import _requests
from test_paged_hybrid import advanced as advanced

from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_planner import build_cache_plan
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.contracts import LayerExecutionSpec
from mlx_one.engine.modes import ModelCapabilityUnsupported, validate_mode
from mlx_one.engine.paged import PagedRunner


def variable_bundle(advanced):
    bundle=advanced()
    layouts=(LayerExecutionSpec(0,'full_kv',4,2,4,4,'rope'),
             LayerExecutionSpec(1,'full_kv',2,1,6,6,'rope'))
    bundle.model.layer_execution_specs=lambda:layouts
    def forward(self,ids,*,cache,position_ids=None,**kwargs):
        self.calls.append(ids.shape)
        batch,width=ids.shape
        for layer,spec in zip(cache,layouts,strict=True):
            data=np.broadcast_to(ids[:,None,:,None]/32,
                (batch,spec.kv_heads,width,spec.key_dim)).astype(np.float16)
            query=np.repeat(data,spec.query_heads//spec.kv_heads,axis=1)
            if callable(getattr(layer,'attend',None)):
                attended=layer.attend(query,data,data,scale=.5,position_ids=position_ids)
            else:
                old=layer.offset
                k,v=layer.update(data,data)
                k,v=(np.repeat(x,spec.query_heads//spec.kv_heads,axis=1) for x in (k,v))
                scores=(query@k.transpose(0,1,3,2))*.5
                valid=np.arange(k.shape[2])[None,:] <= old+np.arange(width)[:,None]
                scores=np.where(valid,scores,-np.inf).astype(np.float32)
                probs=np.exp(scores-scores.max(axis=-1,keepdims=True))
                attended=(probs/probs.sum(axis=-1,keepdims=True))@v
        target=attended.astype(np.float32).sum(axis=(1,3))
        logits=-abs(np.arange(64)[None,None]-target[...,None]).astype(np.float16)
        return ModelOutput(logits,data,cache)
    type(bundle.model).__call__=forward
    return bundle


@pytest.mark.parametrize('page_width',[16,32,64,128])
def test_paged_per_layer_layouts_not_global_dimensions(advanced,page_width):
    bundle=variable_bundle(advanced)
    runtime=CacheManager(bundle,CacheConfig(backend='paged',block_size_tokens=page_width))
    requests=_requests(bundle,runtime,[tuple(range(1,page_width+2)),(2,3)])
    runner=PagedRunner()
    try:
        output=runner.execute(tuple(requests))
        for i,r in enumerate(requests):
            reference=bundle.model(np.array([r.token_ids]),cache=bundle.model.make_cache())
            np.testing.assert_allclose(output.logits[i],reference.logits[:,-1:],atol=.01,rtol=.01)
            assert [(c._blocks[0].block.layout.num_kv_heads,
                     c._blocks[0].block.layout.head_dim) for c in r.cache]==[(2,4),(1,6)]
        assert runner.last_trace['history_materializations']==0
        assert runner.last_trace['layer_layouts'][1]['key_dim']==6
    finally:
        runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0


def test_paged_reconcile_failure_rolls_back_all_rows_layers(advanced,monkeypatch):
    bundle=advanced()
    runtime=CacheManager(bundle,CacheConfig(backend='paged',prefix_cache_enabled=False))
    requests=_requests(bundle,runtime,[(1,2),(3,4,5)])
    runner=PagedRunner()
    runner.execute(tuple(requests))
    old=[[c.offset for c in r.cache] for r in requests]
    identities=[[tuple(ref.block_id for ref in c._blocks) for c in r.cache] for r in requests]
    originals=runtime.reconcile
    def failing(handle):
        originals(handle)
        if handle.request_id=='1':
            raise ValueError('second-row reconciliation failure')
    monkeypatch.setattr(runtime,'reconcile',failing)
    with pytest.raises(ValueError,match='second-row'):
        runner.execute(tuple(replace(r,phase='decode',token_ids=(9,)) for r in requests))
    assert [[c.offset for c in r.cache] for r in requests]==old
    assert [[tuple(ref.block_id for ref in c._blocks) for c in r.cache]
            for r in requests]==identities
    assert [r.handle.computed_tokens for r in requests]==[2,3]
    runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0


def test_paged_validates_every_layer_and_explicit_window_rejection(advanced):
    bundle=advanced()
    bundle.model.layer_execution_specs=lambda:(
        LayerExecutionSpec(0,'full_kv',2,1,4,4,'rope'),
        LayerExecutionSpec(1,'full_kv',2,1,258,258,'rope'))
    with pytest.raises(ModelCapabilityUnsupported,match='dimensions'):
        validate_mode(bundle,build_cache_plan(bundle),'paged')
    hybrid=advanced(True)
    hybrid.model.config.layer_types=('conv','sliding_attention')
    hybrid.model.config.sliding_window=8
    with pytest.raises(ModelCapabilityUnsupported,match='sliding'):
        validate_mode(hybrid,build_cache_plan(hybrid),'paged')


def test_compound_growth_uses_declared_state_shapes(advanced):
    from types import SimpleNamespace

    from test_compound_state import component

    from mlx_one.core.cache import KVCache
    from mlx_one.engine.components import FullKVComponent

    bundle=replace(advanced(),architecture='qwen3_5')
    bundle.model.config=SimpleNamespace(text_config=SimpleNamespace(num_attention_heads=2,
        num_key_value_heads=1,hidden_size=8,head_dim=4,layer_types=('linear_attention','full_attention')))
    bundle.model.make_cache=lambda:(component(),KVCache())
    bundle.model.make_state_components=lambda key_type,pool=None:(component(),FullKVComponent())
    runtime=CacheManager(bundle,CacheConfig(backend='hybrid'))
    handle=runtime.create_handle('compound')
    try:
        amount=runtime.estimate_growth(handle,prompt_tokens=8,max_new_tokens=4)
        assert runtime.plan.carry_bytes()==88
        # The fixture already owns its evaluated carry. Growth reserves only
        # missing storage; the live carry remains in owner accounting.
        assert handle.bundle.allocated_nbytes==88
        assert amount==256*runtime.plan.bytes_per_token()
        handle.bundle.entries[0].conv_state=None
        handle.bundle.entries[0].recurrent_state=None
        assert runtime.estimate_growth(handle,prompt_tokens=8,max_new_tokens=4)==amount+88
    finally:
        runtime.close()


@pytest.mark.parametrize('config',[{'text_config':{'hidden_size':16,'num_attention_heads':2}},
                                 {'n_embd':16,'n_head':2},
                                 {'model_dim':16,'num_query_heads':(1,2,4)}])
def test_reference_memory_forecast_normalizes_adapter_configuration(config):
    from types import SimpleNamespace

    from mlx_one.engine.pressure import activation_forecast

    if 'text_config' in config:
        config={**config,'text_config':SimpleNamespace(**config['text_config'])}
    result=activation_forecast(SimpleNamespace(**config),batch=2,chunk=4,history=16)
    assert result['bytes']>0 and result['measured'] is False


def test_superseded_adapter_contract_cannot_qualify(tmp_path):
    import json
    from dataclasses import asdict

    from native_evidence import attach_provenance

    from mlx_one.registry import (
        InferenceFeatureRegistry,
        InferenceFeatureStatus,
        InferenceScope,
        RegistryError,
    )

    scope=InferenceScope('model','a'*40,'reference_dense','topology','float16','float16',
                         '0.32.2','M4','source')
    proof=attach_provenance({'schema_version':'native-inference-qualification-v4',
        'backend':'mlx-one-native','scope':asdict(scope),'feature':'dense_execution',
        'status':'passed','correctness_passed':True},tmp_path)
    proof['execution_contract']['adapter_version']='native-adapter-v1'
    path=tmp_path/'old-contract.json'
    path.write_text(json.dumps(proof))
    with pytest.raises(RegistryError,match='execution contract'):
        InferenceFeatureRegistry().promote('dense_execution',scope,path,
                                         InferenceFeatureStatus.CORRECTNESS_PASSED)
