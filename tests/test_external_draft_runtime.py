"""External draft semantics and every-position verification regression fixtures."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.engine.batching import BatchedDenseRunner, ForwardRequest
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.speculation import accept_verification, tokenizer_semantics


def test_tokenizer_semantics_includes_vocab_specials_and_eos():
    tokenizer=SimpleNamespace(_tokenizer=SimpleNamespace(to_str=lambda:json.dumps(
        {'model':{'vocab':{'x':1,'y':2}}})),special_tokens={'eos':'y'},
        bos_token_id=None,eos_token_id=2,pad_token_id=0)
    first=SimpleNamespace(tokenizer=tokenizer,eos_token_ids=(2,))
    assert tokenizer_semantics(first)==tokenizer_semantics(first)
    assert tokenizer_semantics(first)!=tokenizer_semantics(SimpleNamespace(
        tokenizer=tokenizer,eos_token_ids=(1,)))
    with pytest.raises(ValueError,match='serialized'):
        tokenizer_semantics(SimpleNamespace(tokenizer=object()))


def test_rejection_at_every_proposed_position(native):
    for rejected in range(3):
        proposal=SimpleNamespace(verify_ids=(1,2,3,4),accepted=0,candidates=[])
        predictions=[2,3,4,5]
        predictions[rejected]=9
        accept_verification(proposal,np.eye(16)[np.array([predictions])])
        assert proposal.accepted==rejected
        assert proposal.candidates==[1,*[2,3,4][:rejected],9]
    with pytest.raises(ValueError,match='every'):
        accept_verification(proposal,np.zeros((1,1,16)))


def test_dense_runner_returns_all_real_logits_without_padding(native):
    runtime=CacheManager(native)
    handles=[runtime.create_handle(str(i)) for i in range(2)]
    requests=tuple(ForwardRequest(native,ids,h.bundle.entries,'prefill',ids,
        runtime,h,request_id=str(i),return_all_logits=True) for i,(ids,h) in enumerate(
            zip(((1,2,3),(4,5)),handles,strict=True)))
    runner=BatchedDenseRunner()
    try:
        output=runner.execute(requests)
        assert [x.shape[1] for x in output.logits]==[3,2]
        assert [h.computed_tokens for h in handles]==[3,2]
        assert runner.last_trace['batch_size']==2
        assert [x.shape[1] for x in runner.execute(tuple(replace(r,return_all_logits=False,
            token_ids=(6,),phase='decode') for r in requests)).logits]==[1,1]
    finally:
        runtime.close()
