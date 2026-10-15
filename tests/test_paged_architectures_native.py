"""Opt-in native per-layer page kernels; tiny models do not qualify checkpoints."""
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS') != '1':
    pytest.skip('requires preflighted Metal host',allow_module_level=True)

import mlx.core as mx
from test_runtime_causal_native import tiny_bundle
from test_runtime_compound_native import compound_bundle

from mlx_one.engine.batching import ForwardRequest
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.hybrid import HybridRunner
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.engine.paged import PagedRunner
from mlx_one.text.loading import LoadedTextModel

FAMILIES=('gpt2','llama','qwen2','qwen3','openelm','qwen2_moe','lfm2','lfm2_moe','qwen3_5')


def paged_bundle(family,tmp_path):
    if family=='qwen3_5':
        return compound_bundle(tmp_path)
    if family not in {'lfm2','lfm2_moe'}:
        return tiny_bundle(family,tmp_path)
    from mlx_one.models.language.lfm2.config import Lfm2Config
    from mlx_one.models.language.lfm2.model import Lfm2ForCausalLM
    from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
    from mlx_one.models.language.lfm2_moe.model import Lfm2MoeForCausalLM

    values=dict(model_type=family,vocab_size=64,hidden_size=16,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        conv_L_cache=3,intermediate_size=24,block_multiple_of=4,
        layer_types=['conv','full_attention'])
    if family=='lfm2_moe':
        values.update(num_dense_layers=1,moe_intermediate_size=12,num_experts=4,num_experts_per_tok=2)
        model=Lfm2MoeForCausalLM(Lfm2MoeConfig.from_dict(values))
    else:
        model=Lfm2ForCausalLM(Lfm2Config.from_dict(values))
    model.eval()
    bundle=LoadedTextModel(model,SimpleNamespace(encode=lambda t:[1,2],bos_token_id=None,
        eos_token_id=63),tmp_path,'tiny/'+family,'tiny-v1',architecture=family,context_length=128)
    configure_execution_dtype(bundle,'float16')
    return bundle


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('size',[1,2,4])
@pytest.mark.parametrize('page_width',[16,32])
def test_native_paged_family_ragged_parity(family,size,page_width,tmp_path,monkeypatch):
    mx.random.seed(51)
    bundle=paged_bundle(family,tmp_path)
    runtime=CacheManager(bundle,CacheConfig(backend='paged',block_size_tokens=page_width,
                                          prefix_cache_enabled=False))
    handles=[runtime.create_handle(str(i)) for i in range(size)]
    prompts=[tuple(1+(i+j)%50 for j in range(17 if family!='qwen3_5' else 16))
             if i==0 or family=='qwen3_5' else (i+1,i+2) for i in range(size)]
    requests=tuple(ForwardRequest(bundle,p,h.bundle.entries,'prefill',p,runtime,h,
        request_id=str(i)) for i,(p,h) in enumerate(zip(prompts,handles,strict=True)))
    oracles=[bundle.model.make_cache() for _ in requests]
    runner=HybridRunner() if runtime.hybrid_state else PagedRunner()
    calls=[]
    original=type(bundle.model).__call__
    def observed(self,ids,**kwargs):
        calls.append(ids.shape[0])
        return original(self,ids,**kwargs)
    monkeypatch.setattr(type(bundle.model),'__call__',observed)
    traces=[]
    try:
        for step in range(4):
            before=len(calls)
            output=runner.execute(requests)
            traces.append(dict(runner.last_trace))
            assert calls[before:]==[size]
            assert runner.last_trace['direct_attention_calls']>0
            assert runner.last_trace['history_materializations']==0
            for i,r in enumerate(requests):
                expected=bundle.model(mx.array([r.token_ids]),cache=oracles[i]).logits[:,-1:]
                mx.eval(expected,output.logits[i])
                assert bool(mx.allclose(output.logits[i],expected,atol=.01,rtol=.01).item())
                assert bool(mx.all(mx.argmax(output.logits[i],axis=-1)
                                   ==mx.argmax(expected,axis=-1)).item())
                for cache,reference in zip(r.cache,oracles[i],strict=True):
                    assert cache.offset==reference.offset
                    if hasattr(cache,'page_table'):
                        with pytest.raises(ValueError,match='materialized'):
                            cache.state()
                    elif cache.kind=='recurrent':
                        for name in ('conv_state','recurrent_state'):
                            assert bool(mx.allclose(getattr(cache,name),getattr(reference,name),
                                                    atol=.01,rtol=.01).item())
                    else:
                        assert bool(mx.allclose(cache.values,reference.values,
                                                atol=.01,rtol=.01).item())
            requests=tuple(replace(r,token_ids=(9+step,),phase='decode') for r in requests)
    finally:
        runtime.close()
        for row in oracles:
            for cache in row:
                cache.release()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0

    destination=os.environ.get('MLX_ONE_NATIVE_RESULTS_DIR')
    if destination:
        directory=Path(destination)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'paged-{family}-b{size}-p{page_width}.json').write_text(json.dumps({
            'scope':'tiny native correctness only','family':family,'batch':size,
            'page_width':page_width,'calls':calls,'traces':traces,
            'cleanup':vars(runtime.stats()),'qualification_promoted':False},indent=2)+'\n')


@pytest.mark.parametrize('family',FAMILIES)
def test_native_paged_prefix_cow_cancel_and_donor_disposal(family,tmp_path,monkeypatch):
    mx.random.seed(57)
    bundle=paged_bundle(family,tmp_path)
    runtime=CacheManager(bundle,CacheConfig(backend='paged',block_size_tokens=16))
    runner=HybridRunner() if runtime.hybrid_state else PagedRunner()
    prompt=tuple(range(1,17))
    fp=runtime.fingerprint(namespace='principal')
    donor=runtime.create_handle('donor')
    donor.metadata['fingerprint']=fp
    try:
        runner.execute((ForwardRequest(bundle,prompt,donor.bundle.entries,'prefill',prompt,
            runtime,donor,request_id='donor'),))
        assert runtime.publish_prefix(donor,fp,prompt)
        borrowers=[runtime.create_handle(name) for name in ('borrower','peer')]
        for borrower in borrowers:
            assert runtime.adopt_prefix(borrower,fp,prompt+(7,))==16
            for source,target in zip(donor.bundle.entries,borrower.bundle.entries,strict=True):
                if hasattr(source,'page_table'):
                    assert source._blocks[0].block_id==target._blocks[0].block_id
                elif getattr(source,'kind',None)=='recurrent':
                    assert source.conv_state is not target.conv_state
                else:
                    assert source.values is not target.values
        runtime.release(donor)
        runtime.clear_prefixes()
        requests=tuple(ForwardRequest(bundle,(7+i,),h.bundle.entries,'prefill',prompt+(7+i,),
            runtime,h,request_id=h.request_id) for i,h in enumerate(borrowers))
        output=runner.execute(requests)
        independent=[bundle.model.make_cache() for _ in requests]
        for i,cache in enumerate(independent):
            expected=bundle.model(mx.array([prompt+(7+i,)]),cache=cache).logits[:,-1:]
            mx.eval(expected,output.logits[i])
            assert bool(mx.allclose(output.logits[i],expected,atol=.01,rtol=.01).item())
            assert bool(mx.all(mx.argmax(output.logits[i],axis=-1)
                               ==mx.argmax(expected,axis=-1)).item())
        cancelled=threading.Event()
        original=type(bundle.model).__call__
        def cancel_forward(self,*args,**kwargs):
            result=original(self,*args,**kwargs)
            mx.eval(result.logits)
            cancelled.set()
            return result
        monkeypatch.setattr(type(bundle.model),'__call__',cancel_forward)
        before=borrowers[0].bundle.clone()
        output=runner.execute(tuple(replace(r,phase='decode',token_ids=(9,),
            cancel=cancelled if i==0 else None) for i,r in enumerate(requests)))
        assert borrowers[0].computed_tokens==17
        assert borrowers[1].computed_tokens==18
        for left,right in zip(borrowers[0].bundle.entries,before.entries,strict=True):
            assert left.offset==right.offset
            if hasattr(left,'page_table'):
                assert left._blocks[0].block_id==right._blocks[0].block_id
            elif left.kind=='recurrent':
                assert bool(mx.allclose(left.recurrent_state,right.recurrent_state).item())
            else:
                assert bool(mx.allclose(left.values,right.values).item())
        expected=bundle.model(mx.array([[9]]),cache=independent[1]).logits[:,-1:]
        mx.eval(expected,output.logits[1])
        assert bool(mx.allclose(output.logits[1],expected,atol=.01,rtol=.01).item())
        before.release()
        for row in independent:
            for cache in row:
                cache.release()
    finally:
        runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0


@pytest.mark.parametrize('family',['lfm2_moe','qwen3_5'])
@pytest.mark.parametrize('mode',['experimental_hybrid_batch','paged'])
def test_native_hybrid_actor_batched_service(family,mode,tmp_path,monkeypatch):
    import time

    from mlx_one import BatchingConfig, Engine
    from mlx_one.engine.actor import InferenceActor

    mx.random.seed(61)
    bundle=paged_bundle(family,tmp_path)
    bundle.tokenizer.encode=lambda text:[1+ord(c)%50 for c in text]
    bundle.tokenizer.decode=lambda ids,**kwargs:''.join(chr(65+int(t)%26) for t in ids)
    # Synchronize test arrivals so the first scheduling turn sees both requests.
    original=InferenceActor._admit
    ready=threading.Event()
    def together(self):
        if not ready.is_set():
            if self._counts['submitted']<2:
                time.sleep(.001)
                return
            ready.set()
        return original(self)
    monkeypatch.setattr(InferenceActor,'_admit',together)
    config=BatchingConfig(max_sequences=2,queue_size=4,prefill_chunk_size=2,
                          max_scheduled_tokens=16,max_prefill_tokens=8)
    with Engine(bundle,inference_mode=mode,batching_config=config) as engine:
        handles=[engine.submit(p,max_tokens=4,temperature=0) for p in ('abc','xyz')]
        results=[h.result(timeout=30) for h in handles]
        stats=engine.runtime_stats()
        traces=stats['scheduler']['forward_traces']
        assert any(t['batch_size']==2 for t in traces)
        assert stats['cache_stats']['active_handles']==stats['cache_stats']['reserved_bytes']==0
        if family=='qwen3_5':
            recurrent=[r for r in stats['layers'] if r['kind']=='recurrent']
            assert all(set(r['actual_dtypes'])=={'float16','float32'} for r in recurrent)
    with Engine(bundle) as reference:
        expected=[reference.generate(p,max_tokens=4,temperature=0) for p in ('abc','xyz')]
    assert [r.text for r in results]==[r.text for r in expected]


@pytest.mark.parametrize('family',['qwen3','openelm'])
def test_native_bfloat16_page_specialization(family,tmp_path):
    bundle=paged_bundle(family,tmp_path)
    configure_execution_dtype(bundle,'bfloat16')
    runtime=CacheManager(bundle,CacheConfig(backend='paged',key_type='bf16',value_type='bf16'))
    requests=[]
    for i,prompt in enumerate(((1,2,3),(4,))):
        h=runtime.create_handle(str(i),key_type='bf16',value_type='bf16')
        requests.append(ForwardRequest(bundle,prompt,h.bundle.entries,'prefill',prompt,
                                       runtime,h,request_id=str(i)))
    try:
        result=PagedRunner().execute(tuple(requests))
        for i,r in enumerate(requests):
            cache=bundle.model.make_cache()
            expected=bundle.model(mx.array([r.token_ids]),cache=cache).logits[:,-1:]
            mx.eval(expected,result.logits[i])
            assert bool(mx.allclose(result.logits[i],expected,atol=.05,rtol=.01).item())
            for layer in cache:
                layer.release()
    finally:
        runtime.close()
