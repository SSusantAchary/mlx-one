"""Native dense B2 speculative verification with unequal histories and membership."""
import os
from dataclasses import replace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

from mlx_one.engine.batching import BatchingConfig
from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.engine.public import Engine
from mlx_one.text.generation import _stream_bundle
from mlx_one.text.schemas import TextGenerationOptions


def bundle(tmp_path):
    from test_native_qwen_mlx import qwen3_config
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from mlx_one.models.language.qwen3.model import Qwen3ForCausalLM
    from mlx_one.text.loading import LoadedTextModel
    from mlx_one.text.tokenizers import HFTokenizerAdapter

    tokenizer=Tokenizer(WordLevel({str(i):i for i in range(64)},unk_token='0'))
    tokenizer.pre_tokenizer=Whitespace()
    wrapped=HFTokenizerAdapter(tokenizer,bos_token=None,eos_token=None,bos_token_id=None,
        eos_token_id=None,pad_token_id=0,special_tokens={})
    b=LoadedTextModel(Qwen3ForCausalLM(qwen3_config()),wrapped,tmp_path,'tiny/qwen3',
                      'tiny-v1',architecture='qwen3',context_length=128)
    configure_execution_dtype(b,'float16')
    return b


def test_external_actor_b2_verification_equals_serial_and_cleans(tmp_path):
    b=bundle(tmp_path)
    options=TextGenerationOptions(max_tokens=10)
    prompts=('1 2','3 4 5 6 7')
    expected=[list(_stream_bundle(b,p,options,None)) for p in prompts]
    engine=Engine(b,inference_mode='experimental_dense_batch',draft_model=b,
        spec_type='draft-model',batching_config=BatchingConfig(max_sequences=2,queue_size=4))
    try:
        with engine._actor._condition:
            handles=[engine.submit(p,options=options) for p in prompts]
        results=[h.result(10) for h in handles]
        assert [r.text for r in results]==[''.join(c.text for c in rows) for rows in expected]
        traces=engine._actor.stats()['traces'] if 'traces' in engine._actor.stats() else list(
            engine._actor._traces)
        assert any(t['batch_size']==2 and len(t.get('staged_verification_rows',()))==2
                   for t in traces)
        assert engine._cache_manager.stats().active_handles==0
        assert engine._cache_manager.stats().reserved_bytes==0
    finally:
        engine.close()


def test_external_same_tokens_when_draft_disagrees_and_stream_cancel(tmp_path):
    import threading

    b=bundle(tmp_path)
    draft=bundle(tmp_path)
    options=TextGenerationOptions(max_tokens=6)
    reference=list(_stream_bundle(b,'1 2',options,None))
    actual=list(_stream_bundle(b,'1 2',options,None,spec_type='draft-model',_draft_bundle=draft))
    assert [c.token_id for c in actual]==[c.token_id for c in reference]
    engine=Engine(b,inference_mode='experimental_dense_batch',spec_type='draft-model',
        draft_model=draft,batching_config=BatchingConfig(max_sequences=2))
    cancel=threading.Event()
    try:
        stream=engine.stream('1 2',options=options,cancel=cancel)
        next(stream)
        cancel.set()
        list(stream)
        assert engine.generate('3 4 5',options=replace(options,max_tokens=2)).generation_tokens==2
        assert engine._cache_manager.stats().active_handles==0
        assert engine._cache_manager.stats().reserved_bytes==0
    finally:
        engine.close()
