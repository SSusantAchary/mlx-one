"""Fail-closed LFM2 Audio task boundary until exact codec/checkpoint contracts exist."""
from dataclasses import dataclass

from mlx_one.engine.modes import ModelCapabilityUnsupported


@dataclass(frozen=True)
class AudioLanguageAvailability:
    architecture_implemented: bool = True
    task_loader: bool = False
    audio_to_text: bool = False
    text_to_audio: bool = False
    speech_to_speech: bool = False
    codec_streaming: bool = False
    reason: str = ('No validated checkpoint, tokenizer/processor, safe tensor and codec '
                   'cadence/flush contract; waveform detokenizer alone is insufficient')


class LFM2AudioAdapter:
    """No task is invented from architecture/component availability."""
    @staticmethod
    def availability():
        return AudioLanguageAvailability()

    def __init__(self,*args,**kwargs):
        raise ModelCapabilityUnsupported(self.availability().reason)
