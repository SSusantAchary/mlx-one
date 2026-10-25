"""TDT timebase/budgets and fail-closed audio-language task availability."""
import pytest

from mlx_one.engine.adapter_registry import get_runtime_registration
from mlx_one.engine.task_runners.transducer import ParakeetTDTAdapter, TDTState


def test_blank_and_lexical_duration_are_independent():
    state=TDTState(3,8,8)
    state.transition(8,0)
    assert state.frame==1 and state.tokens==[8,8]
    state.transition(2,0)
    assert state.frame==1 and state.tokens[-1]==2
    state.transition(3,2)
    assert state.frame==3 and state.finished
    with pytest.raises(ValueError):
        state.transition(2,1)
    state.close()
    state.close()


def test_zero_duration_loop_budget_is_explicit():
    from mlx_one.audio.moonshine import AudioResourceLimitError

    state=TDTState(3,8,1)
    state.transition(2,0)
    with pytest.raises(AudioResourceLimitError,match='truncation'):
        state.transition(2,0)
    assert state.frame==0
    assert get_runtime_registration('parakeet_tdt').adapter() is ParakeetTDTAdapter


def test_lfm_audio_requires_a_real_native_bundle_despite_task_registration():
    from mlx_one.engine.task_runners.lfm_audio import LFM2AudioAdapter

    availability=LFM2AudioAdapter.availability()
    assert availability.architecture_implemented
    assert all((availability.task_loader,availability.audio_to_text,
        availability.text_to_audio,availability.speech_to_speech,availability.codec_streaming))
    with pytest.raises(ValueError,match='checkpoint'):
        LFM2AudioAdapter(object())
    binding=get_runtime_registration('lfm2_audio')
    assert binding.integration_status=='integrated'
    assert len(binding.specifications()[0]) == 3
