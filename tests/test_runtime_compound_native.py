"""Opt-in tiny Qwen3.5 compound-state correctness, never checkpoint qualification."""
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS') != '1':
    pytest.skip('requires preflighted Metal host', allow_module_level=True)

import mlx.core as mx
from test_native_qwen_multimodal_mlx import qwen3_5_config

from mlx_one.engine.batching import ForwardRequest
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.components import load_snapshot, save_snapshot
from mlx_one.engine.hybrid import HybridRunner
from mlx_one.engine.lifecycle import adopt_snapshot
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.models.vision_language.qwen3_5.model import Qwen3_5ForConditionalGeneration
from mlx_one.text.loading import LoadedTextModel


def compound_bundle(tmp_path):
    model = Qwen3_5ForConditionalGeneration(qwen3_5_config())
    model.eval()
    tokenizer = SimpleNamespace(encode=lambda text: [1, 2], bos_token_id=None,eos_token_id=63)
    bundle = LoadedTextModel(model,tokenizer,tmp_path,'tiny/qwen3_5','tiny-v1',
                             architecture='qwen3_5',context_length=128)
    configure_execution_dtype(bundle,'float16')
    return bundle


def compare_compound(actual, reference):
    for left,right in zip(actual,reference,strict=True):
        assert left.offset == right.offset
        names = ('conv_state','recurrent_state') if left.kind == 'recurrent' else ('keys','values')
        for name in names:
            a,b = getattr(left,name),getattr(right,name)
            if left.kind != 'recurrent':
                a,b = a[:,:,:left.offset],b[:,:,:right.offset]
            assert a.dtype == b.dtype
            assert bool(mx.allclose(a,b,atol=.01,rtol=.01).item())
        if left.kind == 'recurrent':
            assert left.conv_state.dtype == mx.float16
            assert left.recurrent_state.dtype == mx.float32


@pytest.mark.parametrize('size',[1,2,4])
def test_native_compound_bucket_cancel(size,tmp_path,monkeypatch):
    mx.random.seed(43)
    bundle = compound_bundle(tmp_path)
    runtime = CacheManager(bundle,CacheConfig(backend='hybrid',prefix_cache_enabled=False))
    handles = [runtime.create_handle(str(i)) for i in range(size)]
    requests = tuple(ForwardRequest(bundle,(i+1,i+2,i+3),h.bundle.entries,'prefill',
        (i+1,i+2,i+3),runtime,h,request_id=str(i)) for i,h in enumerate(handles))
    oracles = [bundle.model.make_cache() for _ in handles]
    runner = HybridRunner()
    calls=[]
    original=type(bundle.model).__call__
    def observed(self,ids,*args,**kwargs):
        calls.append(ids.shape[0])
        return original(self,ids,*args,**kwargs)
    monkeypatch.setattr(type(bundle.model),'__call__',observed)
    try:
        for step in range(4):
            before=len(calls)
            output=runner.execute(requests)
            assert calls[before:] == [size]
            for i,r in enumerate(requests):
                expected=bundle.model(mx.array([r.token_ids]),cache=oracles[i]).logits[:,-1:]
                mx.eval(expected,output.logits[i])
                assert bool(mx.allclose(output.logits[i],expected,atol=.01,rtol=.01).item())
                assert bool(mx.all(mx.argmax(output.logits[i],axis=-1)
                                   ==mx.argmax(expected,axis=-1)).item())
                compare_compound(r.cache,oracles[i])
            requests=tuple(replace(r,token_ids=(9+step,),phase='decode') for r in requests)
        cancel=threading.Event()
        old=handles[0].bundle.clone()
        def cancelled(self,*args,**kwargs):
            output=original(self,*args,**kwargs)
            mx.eval(output.logits)
            cancel.set()
            return output
        monkeypatch.setattr(type(bundle.model),'__call__',cancelled)
        runner.execute((replace(requests[0],cancel=cancel),*requests[1:]))
        compare_compound(requests[0].cache,old.entries)
        for i in range(1,size):
            bundle.model(mx.array([requests[i].token_ids]),cache=oracles[i])
            compare_compound(requests[i].cache,oracles[i])
        old.release()
    finally:
        runtime.close()
        for row in oracles:
            for c in row:
                c.release()
    assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0


def test_native_compound_snapshot_prefix_fork(tmp_path):
    mx.random.seed(47)
    bundle=compound_bundle(tmp_path)
    runtime=CacheManager(bundle,CacheConfig(backend='hybrid',block_size_tokens=16))
    donor=runtime.create_handle('donor')
    fp=runtime.fingerprint(namespace='principal')
    donor.metadata['fingerprint']=fp
    prompt=tuple(1+i%50 for i in range(32))
    runner=HybridRunner()
    try:
        for start in (0,16):
            runner.execute((ForwardRequest(bundle,prompt[start:start+16],donor.bundle.entries,
                'prefill',prompt,runtime,donor,request_id='donor'),))
        save_snapshot(tmp_path/'snapshot',donor.bundle,identity={'test':'compound'})
        restored=load_snapshot(tmp_path/'snapshot',identity={'test':'compound'},
                               expected_specs=runtime.plan.state_specs)
        resumed=runtime.create_handle('resumed')
        assert adopt_snapshot(runtime,resumed,restored)
        borrower=runtime.create_handle('borrower')
        assert runtime.adopt_prefix(borrower,fp,prompt+(7,)) == 32
        for handle,token in ((resumed,8),(borrower,7)):
            result=runner.execute((ForwardRequest(bundle,(token,),handle.bundle.entries,
                'prefill',prompt+(token,),runtime,handle,request_id=handle.request_id),)).logits[0]
            oracle=bundle.model.make_cache()
            expected=bundle.model(mx.array([prompt+(token,)]),cache=oracle).logits[:,-1:]
            mx.eval(result,expected)
            assert bool(mx.allclose(result,expected,atol=.01,rtol=.01).item())
            compare_compound(handle.bundle.entries,oracle)
            for c in oracle:
                c.release()
        assert donor.computed_tokens == 32
        assert donor.bundle.entries[0].conv_state is not borrower.bundle.entries[0].conv_state
    finally:
        runtime.close()
    runtime.close()
    assert runtime.stats().active_handles == runtime.stats().reserved_bytes == 0


@pytest.mark.parametrize('mode',['experimental_hybrid_batch','paged'])
def test_native_compound_lifecycle_profile(mode,tmp_path):
    from mlx_one.engine.lifecycle import hybrid_lifecycle

    mx.random.seed(67)
    bundle=compound_bundle(tmp_path)
    bundle.tokenizer.encode=lambda text:[1+ord(c)%50 for c in text]
    def reload_bundle():
        mx.random.seed(67)
        return compound_bundle(tmp_path)
    profile=SimpleNamespace(inference_mode=mode,cache_type='f16',page_tokens=16,atol=None,rtol=None)
    result=hybrid_lifecycle(bundle,profile,tmp_path/'lifecycle',reload_bundle=reload_bundle)
    assert result['status']=='passed',result
    assert {'conv_state','recurrent_state'} <= {r.get('tensor') for r in result['comparisons']}
    assert all(c['active_handles']==c['reserved_bytes']==0 for c in result['cleanup'])

    destination=os.environ.get('MLX_ONE_NATIVE_RESULTS_DIR')
    if destination:
        directory=Path(destination)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/f'compound-lifecycle-{mode}.json').write_text(json.dumps({
            **result,'scope':'tiny native correctness only','qualification_promoted':False},
            indent=2)+'\n')
