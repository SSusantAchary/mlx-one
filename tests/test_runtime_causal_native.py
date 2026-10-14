"""Opt-in tiny native causal parity; checkpoint/performance gates remain separate."""
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('set MLX_ONE_RUN_MLX_TESTS=1 on a preflighted Metal host',allow_module_level=True)

import mlx.core as mx

from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.text.loading import LoadedTextModel


def tiny_bundle(family,tmp_path):
    if family=='gpt2':
        from mlx_one.models.language.gpt2.config import GPT2Config
        from mlx_one.models.language.gpt2.model import GPT2LMHeadModel
        model=GPT2LMHeadModel(GPT2Config(vocab_size=64,n_embd=12,n_head=4,n_layer=2,
            n_positions=32,n_ctx=32,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,
            bos_token_id=63,eos_token_id=63))
    else:
        from importlib import import_module
        names={'llama':('LlamaConfig','LlamaForCausalLM'),
               'qwen2':('Qwen2Config','Qwen2ForCausalLM'),
               'qwen3':('Qwen3Config','Qwen3ForCausalLM')}
        config_name,model_name=names[family]
        config=getattr(import_module(f'mlx_one.models.language.{family}.config'),config_name)
        model_type=getattr(import_module(f'mlx_one.models.language.{family}.model'),model_name)
        model=model_type(config.from_dict(dict(model_type=family,vocab_size=64,hidden_size=16,
            num_hidden_layers=2,intermediate_size=32,num_attention_heads=4,
            num_key_value_heads=2,head_dim=4,max_position_embeddings=256)))
    model.eval()
    tokenizer=SimpleNamespace(encode=lambda text:[1,2],bos_token_id=None,eos_token_id=63)
    bundle=LoadedTextModel(model,tokenizer,tmp_path,'tiny/'+family,'tiny-v1',
                          architecture=family,context_length=32 if family=='gpt2' else 256)
    configure_execution_dtype(bundle,'float16')
    return bundle


@pytest.mark.parametrize('family',['gpt2','llama','qwen2','qwen3'])
@pytest.mark.parametrize('size',[1,2,4])
def test_tiny_native_causal_ragged_prefill_decode(family,size,tmp_path,monkeypatch):
    mx.random.seed(17)
    bundle=tiny_bundle(family,tmp_path)
    calls=[]
    original=type(bundle.model).__call__
    def observed(self,ids,**kwargs):
        calls.append(ids.shape[0])
        return original(self,ids,**kwargs)
    monkeypatch.setattr(type(bundle.model),'__call__',observed)
    prompts=[(1,2,3,4,5),(3,4),(5,),(6,7,8)][:size]
    requests=tuple(ForwardRequest(bundle,p,bundle.model.make_cache(),'prefill',p,request_id=str(i))
                   for i,p in enumerate(prompts))
    independent=[bundle.model.make_cache() for _ in requests]
    runner=BatchedDenseRunner(bundle)
    try:
        for step in range(4):
            output=runner.execute(requests)
            for i,request in enumerate(requests):
                oracle=bundle.model(mx.array([request.token_ids]),cache=independent[i]).logits[:,-1:]
                mx.eval(oracle,output.logits[i])
                assert bool(mx.allclose(oracle,output.logits[i],atol=.01,rtol=.01).item())
                for a,b in zip(request.cache,independent[i],strict=True):
                    assert a.offset==b.offset
                    for x,y in zip(a.state(),b.state(),strict=True):
                        assert bool(mx.allclose(x,y,atol=.01,rtol=.01).item())
            requests=tuple(replace(r,token_ids=(9+step,),phase='decode') for r in requests)
        assert size in calls
        if size>1:
            assert calls.count(size)==4
    finally:
        for cache in [c for r in requests for c in r.cache]+[c for row in independent for c in row]:
            cache.release()


def test_gpt2_legal_rows_can_have_padded_extent_beyond_context(tmp_path):
    mx.random.seed(9)
    bundle=tiny_bundle('gpt2',tmp_path)
    runner=BatchedDenseRunner(bundle)
    requests=tuple(ForwardRequest(bundle,p,bundle.model.make_cache(),'prefill',p,request_id=str(i))
                   for i,p in enumerate(((1,)*31,(2,))))
    for request in requests:
        runner.execute((request,))
    requests=(replace(requests[0],token_ids=(3,)),
              replace(requests[1],token_ids=tuple(i%63 for i in range(31))))
    independent=[tuple(c.clone() for c in r.cache) for r in requests]
    try:
        output=runner.execute(requests)
        for i,request in enumerate(requests):
            oracle=bundle.model(mx.array([request.token_ids]),cache=independent[i]).logits[:,-1:]
            mx.eval(oracle,output.logits[i])
            assert bool(mx.allclose(oracle,output.logits[i],atol=.01,rtol=.01).item())
            assert all(c.offset==32 for c in request.cache)
    finally:
        for cache in [c for r in requests for c in r.cache]+[c for row in independent for c in row]:
            cache.release()
