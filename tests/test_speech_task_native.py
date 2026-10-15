"""Opt-in native ready-row speech decoder batching and cancellation isolation."""
import os

import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS')!='1':
    pytest.skip('requires supervised Metal execution',allow_module_level=True)

import mlx.core as mx

from mlx_one.engine.task_runners.speech import EncoderDecoderSpeechRunner


@pytest.mark.parametrize('family',['whisper','moonshine'])
def test_speech_b2_cross_state_and_transactional_cancellation(family):
    if family=='whisper':
        from test_native_whisper_mlx import config

        from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration

        model=WhisperForConditionalGeneration(config())
        inputs=(mx.random.normal((1,80,8)),mx.random.normal((1,80,8)))
    else:
        from test_moonshine_mlx import model as make

        model=make()
        inputs=(mx.random.normal((1,4000)),mx.random.normal((1,4000)))
    runner=EncoderDecoderSpeechRunner(model,family)
    together=runner.encode(inputs)
    separate=tuple(runner.encode((x,))[0] for x in inputs)
    try:
        for token in (2,3):
            ids=mx.array([[token],[token+1]])
            actual=runner.decode(together,ids)
            expected=mx.concatenate([runner.decode((row,),ids[i:i+1])
                for i,row in enumerate(separate)],axis=0)
            mx.eval(actual,expected)
            assert mx.max(mx.abs(actual-expected)).item()<1e-4
        previous=together[0].cache[0].offset
        runner.decode(together,mx.array([[4],[5]]),cancelled=lambda row:row==0)
        assert together[0].cache[0].offset==previous
        assert together[1].cache[0].offset==previous+1
        assert 2 in runner.forward_batches
        donor=together[1].cache[0].cross_keys
        together[0].close()
        assert together[1].cache[0].cross_keys is donor
    finally:
        for row in (*together,*separate):
            row.close()
            row.close()


def test_native_service_empty_audio_and_ordered_stream_lease():
    from test_moonshine_mlx import bundle as moonshine_bundle
    from test_qwen_audio_mlx import bundle as qwen_bundle

    from mlx_one.engine.service import InferenceService

    service=InferenceService()
    try:
        service.register_task('moonshine',moonshine_bundle())
        result=service.run_task('moonshine','transcription',{'audio':[],'sample_rate':16000})
        assert result.output.text=='' and service.pool.stats()['active_leases']==0
        service.unload('moonshine')
        service.register_task('qwen',qwen_bundle())
        with service.open_audio_session('qwen',sample_rate=16000) as session:
            assert session.call('push',[],sample_rate=16000,event_number=0)==()
            assert session.call('finish',event_number=1).text==''
            assert service.pool.stats()['active_leases']==1
        assert service.pool.stats()['active_leases']==0 and not service._sessions
    finally:
        service.close()
