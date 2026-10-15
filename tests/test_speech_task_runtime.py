"""Backend-free task preparation, dispatch, session epochs and release failures."""
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest

from mlx_one.engine.adapter_registry import get_runtime_registration
from mlx_one.engine.cache_errors import CacheCapacityError
from mlx_one.engine.contracts import TaskKind
from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.sessions import DeviceSession
from mlx_one.engine.task_runners.speech import (
    MoonshineAdapter,
    Qwen3AlignmentAdapter,
    Qwen3ASRAdapter,
    SpeechDecoderState,
    WhisperAdapter,
)


@pytest.mark.parametrize('adapter',[WhisperAdapter,MoonshineAdapter,Qwen3ASRAdapter,
                                    Qwen3AlignmentAdapter])
def test_audio_preparation_identity_and_request_local_pcm(adapter):
    bundle=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(model_type=adapter.family)))
    runner=adapter(bundle)
    task=runner.supported_tasks[0]
    source=[.1,.2]
    payload={'audio':source,'sample_rate':16000,'text':'hello','language':'English'}
    prepared=runner.prepare(task,payload)
    source[0]=9
    assert prepared.payload['audio'][0]==pytest.approx(.1)
    assert get_runtime_registration(adapter.family).adapter() is adapter
    for changed in ({'sample_rate':True},{'audio':[float('nan')]},{'audio':[[1],[2]]}):
        with pytest.raises((ValueError,TypeError)):
            runner.prepare(task,{**payload,**changed})
    with pytest.raises(CacheCapacityError):
        runner.prepare(task,{**payload,'audio':[0.]*(180*16000+1)})


def test_alignment_does_not_dispatch_autoregressive_transcription(monkeypatch):
    import mlx_one.audio.qwen

    runner=Qwen3AlignmentAdapter(SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        model_type='qwen3_forced_aligner'))))
    p=runner.prepare(TaskKind.ALIGNMENT,{'audio':[0.],'sample_rate':16000,
                                       'text':'known text','language':'English'})
    monkeypatch.setattr(mlx_one.audio.qwen,'align_audio',lambda *a,**kw:('spans',kw['text']))
    assert runner.run_batch((p,))==(('spans','known text'),)


def test_session_order_cancel_and_disposal_failure_release_epoch():
    executor=DeviceExecutor()
    released=[]
    lease=SimpleNamespace(runtime=SimpleNamespace(key='audio',epoch=7),
                          release=lambda:released.append(True))
    class State:
        def push(self,x):
            assert executor.is_owner
            return x
        def close(self):
            assert executor.is_owner
            raise RuntimeError('dispose failure')
    session=DeviceSession(lease,executor,State,methods=('push',))
    try:
        assert session.call('push',1,event_number=0)==1
        with pytest.raises(ValueError,match='reordered'):
            session.call('push',2,event_number=0)
        assert session.call('push',3,event_number=1)==3
        with pytest.raises(RuntimeError,match='dispose'):
            session.close()
        session.close()
        assert released==[True]
        with pytest.raises(CancelledError):
            session.call('push',4)
    finally:
        executor.close()


def test_speech_state_cleanup_attempts_every_component():
    released=[]
    def broken():
        released.append('broken')
        raise RuntimeError('corruption')
    state=SpeechDecoderState(object(),(SimpleNamespace(reset=broken),
        SimpleNamespace(reset=lambda:released.append('peer'))))
    with pytest.raises(RuntimeError,match='corruption'):
        state.close()
    state.close()
    assert state.encoded is None and released==['broken','peer']
