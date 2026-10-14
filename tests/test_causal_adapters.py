from dataclasses import replace

import pytest
from test_dense_batching import native as native

from mlx_one.engine.adapters import native_adapter
from mlx_one.engine.batching import ForwardRequest
from mlx_one.engine.contracts import (
    BatchCompatibilityKey,
    BatchPlan,
    EngineRequest,
    RequestKind,
    ScheduledSequence,
    StageKind,
    TaskKind,
    WorkItem,
)


def test_registered_adapter_uses_real_work_and_request_mapping(native):
    adapter=native_adapter(native)
    assert adapter.describe(native).architecture=='qwen2'
    state=adapter.prepare(EngineRequest(RequestKind.TEXT,native.model_id,{'prompt':'hello'}))
    assert adapter.estimate(state,adapter.plan)['fp16_kv_bytes']>0
    key=BatchCompatibilityKey(str(id(native)),0,adapter.spec.adapter_version,
        TaskKind.TEXT_GENERATION,StageKind.PREFILL,adapter.spec.quantization_digest,
        adapter.plan.topology_digest,'rope','plain-text-v1')
    request=ForwardRequest(native,(3,4),native.model.make_cache(),'prefill',(3,4),request_id='r')
    work=WorkItem('r',key,2,2,0,'tokens',payload=request)
    packed=adapter.pack((work,))
    plan=BatchPlan((ScheduledSequence('r','prefill',0,2),),2)
    result=adapter.unpack(adapter.prefill(packed,plan))
    assert result[0].request_id=='r' and result[0].committed_tokens==2
    assert all(cache.offset==2 for cache in request.cache)
    with pytest.raises(ValueError,match='compatibility'):
        adapter.pack((work,replace(work,compatibility=replace(key,model_epoch=1))))
    with pytest.raises(ValueError,match='boundary'):
        adapter.execute(packed,plan)


def test_adapter_family_binding_rejects_wrong_checkpoint(native):
    from mlx_one.engine.cache_planner import build_cache_plan
    from mlx_one.engine.native_adapters import Qwen3Adapter

    with pytest.raises(ValueError,match='supported'):
        Qwen3Adapter(native,build_cache_plan(native))


def test_odd_learned_head_dimension_is_legal_and_paging_unavailable(native):
    from mlx_one.engine.cache_planner import build_cache_plan
    from mlx_one.engine.modes import ModelCapabilityUnsupported, validate_mode

    bundle=replace(native,architecture='gpt2')
    bundle.model.config.model_type='gpt2'
    bundle.model.config.head_dim=3
    adapter=native_adapter(bundle)
    assert adapter.spec.head_dim==3 and adapter.spec.layers[0].position_policy=='learned'
    with pytest.raises(ModelCapabilityUnsupported,match='paged'):
        validate_mode(bundle,build_cache_plan(bundle),'paged')


def test_moe_forecast_counts_all_expert_outputs(native):
    bundle=replace(native,architecture='qwen2_moe')
    bundle.model.config.num_experts=8
    bundle.model.config.moe_intermediate_size=16
    bundle.model.config.num_experts_per_tok=2
    adapter=native_adapter(bundle)
    bound=adapter.additional_transient_bytes(batch=4,chunk=512)
    assert bound>4*512*2*(2*16+2)*4
    assert adapter.memory_estimate(4)['model_storage_bytes']==native.model.weights.nbytes
