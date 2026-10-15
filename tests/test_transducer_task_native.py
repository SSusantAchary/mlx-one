"""Native B2 TDT joint/predictor parity, cancellation and emission timebase."""
import os

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

import mlx.core as mx

from mlx_one.audio.parakeet import greedy_decode
from mlx_one.engine.task_runners.transducer import TDTState, TransducerRunner


@pytest.mark.parametrize('token',[1,8])
def test_ready_b2_tdt_matches_independent_and_cancellation(token):
    from test_parakeet_mlx import tiny_model

    model=tiny_model()
    model.joint.head.weight=mx.zeros_like(model.joint.head.weight)
    model.joint.head.bias=mx.zeros_like(model.joint.head.bias)
    model.joint.head.bias[token]=20.
    model.joint.head.bias[10 if token==1 else 9]=20.
    encoded=mx.random.normal((2,3,4))
    states=[TDTState(n,8,12) for n in (2,3)]
    runner=TransducerRunner(model)
    while any(not s.finished for s in states):
        runner.step(encoded,states)
    assert 2 in runner.forward_batches and 1 in runner.forward_batches
    for i,state in enumerate(states):
        expected=greedy_decode(model,encoded[i:i+1],state.length,{})
        assert (tuple(state.tokens),tuple(state.durations))==expected
    cancelled=[TDTState(2,8,12),TDTState(3,8,12)]
    runner.step(encoded,cancelled,cancelled=lambda row:row==0)
    assert cancelled[0].frame==0 and cancelled[0].predictor is None
    assert cancelled[1].frame==1
    for state in (*states,*cancelled):
        state.close()


def test_native_transducer_task_batches_ragged_audio_and_preserves_timebase(tmp_path):
    from test_parakeet_mlx import tiny_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from mlx_one.engine.task_runners.transducer import ParakeetTDTAdapter
    from mlx_one.models.audio.parakeet.loading import LoadedParakeet
    from mlx_one.models.audio.parakeet.processing import ParakeetProcessor

    model=tiny_model()
    model.joint.head.weight=mx.zeros_like(model.joint.head.weight)
    model.joint.head.bias=mx.zeros_like(model.joint.head.bias)
    model.joint.head.bias[8]=20.
    model.joint.head.bias[9]=20.
    tokenizer=Tokenizer(WordLevel({str(i):i for i in range(9)},unk_token='0'))
    bundle=LoadedParakeet(model,tokenizer,ParakeetProcessor(feature_size=16),{},tmp_path,'pin','v3')
    adapter=ParakeetTDTAdapter(bundle)
    prepared=tuple(adapter.prepare('transcription',{'audio':[0.]*n,'sample_rate':16000})
                   for n in (1600,3200))
    outputs=adapter.run_batch(prepared)
    assert [x.duration for x in outputs]==[.1,.2]
    assert 2 in outputs[0].provenance['joint_forward_batches']
    assert 1 in outputs[0].provenance['joint_forward_batches']
    assert all(x.provenance['streaming'] is False for x in outputs)
