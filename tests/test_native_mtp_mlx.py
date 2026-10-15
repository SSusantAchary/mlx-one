"""Opt-in tiny Qwen3.5 native MTP actor parity; no predictor checkpoint claim."""
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.engine.public import Engine
from mlx_one.text.generation import _stream_bundle
from mlx_one.text.schemas import TextGenerationOptions


def bundle(tmp_path):
    from test_runtime_compound_native import compound_bundle

    value=compound_bundle(tmp_path)
    value.model.enable_mtp()
    configure_execution_dtype(value,'float16')
    tokenizer=SimpleNamespace(encode=lambda text:[int(t) for t in text.split(',')],
        decode=lambda ids:','.join(str(t) for t in ids),bos_token_id=None,eos_token_id=None)
    # Generation's optional reasoning delimiters must not acquire artificial tokens.
    base=tokenizer.encode
    tokenizer.encode=lambda text:[] if text in ('<think>','</think>') else base(text)
    return replace(value,tokenizer=tokenizer,eos_token_ids=())


@pytest.mark.parametrize('limit',[1,2,5,9])
def test_native_actor_mtp_matches_independent_greedy(limit,tmp_path):
    b=bundle(tmp_path)
    options=TextGenerationOptions(max_tokens=limit)
    expected=list(_stream_bundle(b,'1,2,3',options,None))
    engine=Engine(b,spec_type='draft-mtp',inference_mode='reference_dense')
    try:
        actual=list(engine.stream('1,2,3',options=options))
        assert [c.token_id for c in actual]==[c.token_id for c in expected]
        assert actual[-1].finish_reason==expected[-1].finish_reason
        assert engine._cache_manager.stats().active_handles==0
        assert engine._cache_manager.stats().reserved_bytes==0
        assert engine.runtime_stats()['speculation']['production_qualified'] is False
    finally:
        engine.close()


def test_native_mtp_stop_rollback_and_cancel(tmp_path):
    import threading

    b=bundle(tmp_path)
    options=TextGenerationOptions(max_tokens=8)
    baseline=list(_stream_bundle(b,'1,2',options,None))
    tokens=[c.token_id for c in baseline if c.token_id is not None]
    if not tokens:
        pytest.skip('random tiny model emitted immediate EOS')
    stopped=replace(options,stop=(str(tokens[0]),))
    expected=list(_stream_bundle(b,'1,2',stopped,None))
    actual=list(_stream_bundle(b,'1,2',stopped,None,spec_type='draft-mtp'))
    assert actual==expected
    event=threading.Event()
    engine=Engine(b,spec_type='draft-mtp',inference_mode='reference_dense')
    try:
        stream=engine.stream('1,2',options=options,cancel=event)
        next(stream)
        event.set()
        list(stream)
        assert engine._cache_manager.stats().active_handles==0
    finally:
        engine.close()
