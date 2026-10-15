"""Opt-in tiny native complete-state multimodal continuation parity."""
import os
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS') != '1':
    pytest.skip('requires supervised native Metal execution',allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.engine.cache import CacheBundle
from mlx_one.vision.generation import VisionForward


@pytest.mark.parametrize('family',['qwen2_5_vl','qwen3_5','lfm2_vl'])
def test_multimodal_cached_positions_equal_fresh_prefill(family):
    from test_native_qwen_multimodal_mlx import qwen2_5_vl_config, qwen3_5_config

    if family=='lfm2_vl':
        from test_native_lfm_mlx import dense_config

        from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig
        from mlx_one.models.vision_language.lfm2_vl.model import Lfm2VLForConditionalGeneration
        from mlx_one.models.vision_language.lfm2_vl.weights import weight_contract

        config=Lfm2VLConfig.from_dict({'model_type':family,'image_token_id':63,
            'text_config':dense_config().__dict__,'vision_config':{'hidden_size':12,
            'intermediate_size':24,'num_hidden_layers':2,'num_attention_heads':3,
            'patch_size':2,'image_size':4,'downsample_factor':2,'projection_dim':16}})
        model=Lfm2VLForConditionalGeneration(config)
        assert {k:tuple(v.shape) for k,v in tree_flatten(model.parameters())}==dict(
            weight_contract(config).expected)
        ids=[1,63,2]
        kwargs={'pixel_values':mx.random.normal((1,4,12)),'spatial_shapes':((2,2),)}
    else:
        from mlx_one.core.registry import get_registration

        config=qwen2_5_vl_config() if family=='qwen2_5_vl' else qwen3_5_config()
        model=get_registration(family).model_class()(config)
        ids=[1,62,60,60,60,60,63,2]
        kwargs={'pixel_values':mx.random.normal((16,12)),'image_grid_thw':((1,4,4),),
                'mm_token_type_ids':mx.array([[0,0,1,1,1,1,0,0]])}
    bundle=SimpleNamespace(model=model)
    wrapped=VisionForward(bundle,SimpleNamespace(**kwargs),lambda:False)
    wrapped.text_bundle=bundle
    state=CacheBundle(model.make_cache())
    try:
        initial=wrapped(mx.array([ids]),cache=state.entries)
        mx.eval(initial.logits)
        if family!='lfm2_vl':
            assert wrapped.delta is not None
            assert mx.any(wrapped.delta!=0).item()
        for forced in (3,4):
            continued=wrapped(mx.array([[forced]]),cache=state.entries)
            ids.append(forced)
            fresh_kwargs=dict(kwargs)
            if 'mm_token_type_ids' in kwargs:
                fresh_kwargs['mm_token_type_ids']=mx.concatenate((
                    kwargs['mm_token_type_ids'],mx.zeros((1,len(ids)-8),mx.int32)),axis=1)
            fresh=model(mx.array([ids]),**fresh_kwargs)
            mx.eval(continued.logits,fresh.logits)
            assert mx.max(mx.abs(continued.logits[:,-1]-fresh.logits[:,-1])).item()<1e-4
            assert {entry.offset for entry in state.entries}=={len(ids)}
    finally:
        state.release()
        state.release()
