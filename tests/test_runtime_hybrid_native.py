"""Tiny native hybrid state parity; no checkpoint or performance promotion."""
import os
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('set MLX_ONE_RUN_MLX_TESTS=1 on a preflighted Metal host',allow_module_level=True)

import mlx.core as mx

from mlx_one.engine.batching import ForwardRequest
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.hybrid import HybridRunner
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.text.loading import LoadedTextModel


def hybrid_bundle(family,tmp_path):
    from mlx_one.models.language.lfm2.config import Lfm2Config
    from mlx_one.models.language.lfm2.model import Lfm2ForCausalLM
    from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
    from mlx_one.models.language.lfm2_moe.model import Lfm2MoeForCausalLM

    values=dict(model_type=family,vocab_size=64,hidden_size=16,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
        conv_L_cache=3,intermediate_size=24,block_multiple_of=4,
        layer_types=['conv','full_attention','sliding_attention'],sliding_window=8)
    if family=='lfm2_moe':
        values.update(num_dense_layers=1,moe_intermediate_size=12,num_experts=4,num_experts_per_tok=2)
        model=Lfm2MoeForCausalLM(Lfm2MoeConfig.from_dict(values))
    else:
        model=Lfm2ForCausalLM(Lfm2Config.from_dict(values))
    model.eval()
    tokenizer=SimpleNamespace(encode=lambda text:[3+ord(c)%59 for c in text],
                              bos_token_id=None,eos_token_id=63)
    bundle=LoadedTextModel(model,tokenizer,tmp_path,'tiny/'+family,'tiny-v1',
                          architecture=family,context_length=128)
    configure_execution_dtype(bundle,'float16')
    return bundle


def compare_state(actual,reference):
    for component,expected in zip(actual,reference,strict=True):
        assert component.offset==expected.offset
        if component.kind=='conv':
            assert bool(mx.allclose(component.values,expected.values,atol=.01,rtol=.01).item())
        else:
            length=min(component.offset,getattr(component,'max_size',component.offset))
            for x,y in zip(component.state(),expected.state(),strict=True):
                assert x.dtype==mx.float16
                assert bool(mx.allclose(x[:,:,:length],y[:,:,-length:],atol=.01,rtol=.01).item())


@pytest.mark.parametrize('family',['lfm2','lfm2_moe'])
@pytest.mark.parametrize('size',[1,2,4])
def test_native_hybrid_ragged_state_and_cancel(family,size,tmp_path,monkeypatch):
    mx.random.seed(27)
    bundle=hybrid_bundle(family,tmp_path)
    runtime=CacheManager(bundle,CacheConfig(backend='hybrid',prefix_cache_enabled=False))
    handles=[runtime.create_handle(str(i)) for i in range(size)]
    prompts=[(1,2,3,4,5),(2,3),(4,),(5,6,7)][:size]
    requests=tuple(ForwardRequest(bundle,p,h.bundle.entries,'prefill',p,runtime,h,request_id=str(i))
                   for i,(p,h) in enumerate(zip(prompts,handles,strict=True)))
    independent=[bundle.model.make_cache() for _ in requests]
    runner=HybridRunner()
    try:
        for step in range(4):
            output=runner.execute(requests)
            assert runner.last_trace['batch_size']==size
            for i,request in enumerate(requests):
                expected=bundle.model(mx.array([request.token_ids]),cache=independent[i]).logits[:,-1:]
                mx.eval(expected,output.logits[i])
                assert bool(mx.allclose(output.logits[i],expected,atol=.01,rtol=.01).item())
                assert bool(mx.all(mx.argmax(output.logits[i],axis=-1)
                                   == mx.argmax(expected,axis=-1)).item())
                compare_state(request.cache,independent[i])
            from mlx_one.engine.reporting import tensor_dtypes

            assert tensor_dtypes(bundle.model.parameters()) == ("float16",)
            requests=tuple(replace(r,token_ids=(9+step,),phase='decode') for r in requests)
        cancel=threading.Event()
        before=handles[0].bundle.clone()
        original=type(bundle.model).__call__
        fired=[False]
        def cancel_forward(self,*args,**kwargs):
            result=original(self,*args,**kwargs)
            mx.eval(result.logits)
            if not fired[0]:
                cancel.set()
                fired[0]=True
            return result
        monkeypatch.setattr(type(bundle.model),'__call__',cancel_forward)
        output=runner.execute((replace(requests[0],cancel=cancel),*requests[1:]))
        compare_state(requests[0].cache,before.entries)
        for i,request in enumerate(requests[1:],start=1):
            expected=bundle.model(mx.array([request.token_ids]),cache=independent[i]).logits[:,-1:]
            mx.eval(expected,output.logits[i])
            assert bool(mx.allclose(output.logits[i],expected,atol=.01,rtol=.01).item())
        before.release()
    finally:
        runtime.close()
        for row in independent:
            for cache in row:
                cache.release()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0


@pytest.mark.parametrize('family',['lfm2','lfm2_moe'])
def test_native_complete_prefix_restore(family,tmp_path):
    mx.random.seed(31)
    bundle=hybrid_bundle(family,tmp_path)
    runtime=CacheManager(bundle,CacheConfig(backend='hybrid',block_size_tokens=16))
    fp=runtime.fingerprint(namespace='test-principal')
    donor=runtime.create_handle('donor')
    donor.metadata['fingerprint']=fp
    prompt=tuple(1+i%50 for i in range(32))
    runner=HybridRunner()
    try:
        for start in (0,16):
            runner.execute((ForwardRequest(bundle,prompt[start:start+16],donor.bundle.entries,
                'prefill',prompt,runtime,donor,request_id='donor'),))
        borrower=runtime.create_handle('borrower')
        reused=runtime.adopt_prefix(borrower,fp,prompt+(7,))
        assert reused==32
        assert borrower.bundle.entries[0].values is not donor.bundle.entries[0].values
        output=runner.execute((ForwardRequest(bundle,(7,),borrower.bundle.entries,'prefill',
            prompt+(7,),runtime,borrower,request_id='borrower'),)).logits[0]
        independent=bundle.model.make_cache()
        expected=bundle.model(mx.array([prompt+(7,)]),cache=independent).logits[:,-1:]
        mx.eval(output,expected)
        assert bool(mx.allclose(output,expected,atol=.01,rtol=.01).item())
        assert bool(mx.all(mx.argmax(output,axis=-1)==mx.argmax(expected,axis=-1)).item())
        compare_state(borrower.bundle.entries,independent)
        private=runtime.create_handle('private')
        assert runtime.adopt_prefix(private,runtime.fingerprint(namespace='other'),prompt+(7,))==0
        identical=runtime.create_handle('identical')
        assert runtime.adopt_prefix(identical,fp,prompt)==16
        for cache in independent:
            cache.release()
    finally:
        runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0
