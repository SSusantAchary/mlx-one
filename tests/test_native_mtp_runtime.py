"""Deterministic target/draft logits; committed cache never includes sampled bonus."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.engine.cache import CacheBundle
from mlx_one.engine.components import FullKVComponent
from mlx_one.engine.speculation import SpeculativeProposal, validate_mtp


def fixture_bundle(wrong=False,last_only=False):
    class Model:
        config=SimpleNamespace(tie_word_embeddings=False)
        language_model=SimpleNamespace(embed_tokens=lambda ids:ids)
        def make_mtp_cache(self):
            return (FullKVComponent(),)
        def mtp(self,ids,hidden,**kwargs):
            return ids+ (6 if wrong else 1)
        def lm_head(self,hidden):
            return np.eye(16,dtype=np.float32)[hidden]
        def __call__(self,ids,*,cache):
            values=np.ones((1,1,ids.shape[1],2),np.float16)
            for c in cache:
                c.update(values,values)
            logits=np.eye(16,dtype=np.float32)[(ids+1)%16]
            return SimpleNamespace(logits=logits[:,-1:] if last_only else logits)
    return SimpleNamespace(model=Model(),architecture='qwen3_5')


@pytest.mark.parametrize('wrong,consumed,expected',[(False,4,5),(False,2,3),(True,2,3)])
def test_greedy_verification_commit_and_bonus_boundary(native,wrong,consumed,expected):
    bundle=fixture_bundle(wrong)
    state=CacheBundle((FullKVComponent(),))
    state.entries[0].update(np.ones((1,1,2,2),np.float16),np.ones((1,1,2,2),np.float16))
    step=SpeculativeProposal(bundle,state.entries)
    try:
        step.propose_mtp(3,np.zeros((1,1,2)),2)
        assert state.offset==2
        assert step.candidates[:2]==[3,4]
        step.commit(consumed)
        assert state.offset==expected
    finally:
        step.close()
        step.close()
        state.release()


def test_failure_and_cancellation_release_staged_state(native):
    state=CacheBundle((FullKVComponent(),))
    cancelled=[False]
    step=SpeculativeProposal(fixture_bundle(),state.entries,cancelled=lambda:cancelled[0])
    step.propose_mtp(3,np.zeros((1,1,2)),2)
    cancelled[0]=True
    step.commit(2)
    step.close()
    assert state.offset==0
    broken=SpeculativeProposal(fixture_bundle(last_only=True),state.entries)
    with pytest.raises(ValueError,match='every'):
        broken.propose_mtp(3,np.zeros((1,1,2)),2)
    assert state.offset==0
    state.release()


def test_mtp_is_explicit_greedy_and_checkpoint_specific():
    with pytest.raises(ValueError,match='Qwen3.5'):
        validate_mtp(SimpleNamespace(architecture='qwen3',model=object()))
    with pytest.raises(ValueError,match='greedy'):
        validate_mtp(fixture_bundle(),temperature=.5)
