"""Native cold restoration against fresh full-prefill, including complete carries."""
import os
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

import mlx.core as mx

from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.text.loading import LoadedTextModel


@pytest.mark.parametrize('family',['qwen2','qwen3_5','lfm2','qwen3-pinned'])
def test_ssd_restored_continuation_matches_fresh_complete_prefill(family,tmp_path):
    if family=='qwen3-pinned':
        if os.environ.get('MLX_ONE_RUN_P1_CHECKPOINTS')!='1':
            pytest.skip('pinned checkpoint qualification is opt-in')
        from mlx_one.p1_profiles import CHECKPOINTS
        from mlx_one.text.loading import load_text_model

        asset=CHECKPOINTS['qwen3-small']
        bundle=load_text_model(asset.model_id,revision=asset.revision,offline=True)
        configure_execution_dtype(bundle,'float16')
    elif family=='qwen2':
        from test_native_qwen_mlx import qwen2_config

        from mlx_one.models.language.qwen2.model import Qwen2ForCausalLM

        bundle=LoadedTextModel(Qwen2ForCausalLM(qwen2_config()),
            SimpleNamespace(encode=lambda s:[1,2],bos_token_id=None,eos_token_id=63),
            tmp_path,'tiny/qwen2','tiny-v1',architecture=family,context_length=128)
        configure_execution_dtype(bundle,'float16')
    elif family=='qwen3_5':
        from test_runtime_compound_native import compound_bundle

        bundle=compound_bundle(tmp_path)
    else:
        from test_runtime_hybrid_native import hybrid_bundle

        bundle=hybrid_bundle(family,tmp_path)
    mode='dense' if family in {'qwen2','qwen3-pinned'} else 'hybrid'
    runtime=CacheManager(bundle,CacheConfig(backend=mode,ssd_cache_directory=str(tmp_path/'ssd'),
        ssd_cache_budget_bytes=10000000,ssd_staging_budget_bytes=256*1024**2,
        block_size_tokens=16))
    donor=runtime.create_handle('donor')
    borrower=runtime.create_handle('borrower')
    fp=runtime.fingerprint(namespace='principal')
    ids=(1,2,3,4)*4
    try:
        result=bundle.model(mx.array([ids]),cache=donor.bundle.entries)
        mx.eval(result.logits)
        donor.metadata['fingerprint']=fp
        assert runtime.publish_prefix(donor,fp,ids)
        runtime.clear_prefixes()
        runtime.release(donor)
        assert runtime.adopt_prefix(borrower,fp,ids+(5,))==16
        for token in (5,6):
            if mode=='hybrid':
                from mlx_one.engine.batching import ForwardRequest
                from mlx_one.engine.hybrid import HybridRunner

                actual=HybridRunner().execute((ForwardRequest(bundle,(token,),
                    borrower.bundle.entries,'decode',ids,runtime,borrower,request_id='borrower'),
                    )).logits[0]
            else:
                actual=bundle.model(mx.array([[token]]),cache=borrower.bundle.entries).logits
            ids=ids+(token,)
            fresh_cache=tuple(bundle.model.make_cache())
            from mlx_one.core.cache import configure_kv_caches

            fresh_cache=configure_kv_caches(fresh_cache,'f16','f16')
            expected=bundle.model(mx.array([ids]),cache=fresh_cache).logits[:,-1:]
            mx.eval(actual,expected)
            assert bool(mx.allclose(actual,expected,atol=.01,rtol=.01).item())
            assert mx.array_equal(mx.argmax(actual,axis=-1),mx.argmax(expected,axis=-1)).item()
            assert {c.offset for c in borrower.bundle.entries}=={len(ids)}
            for c in fresh_cache:
                c.reset()
    finally:
        runtime.close()
    assert runtime.stats().active_handles==runtime.stats().reserved_bytes==0
