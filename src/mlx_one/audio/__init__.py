"""Native audio services; importing this namespace does not initialize MLX."""

from mlx_one.audio.loading import AudioModelLoadError, load_audio_model
from mlx_one.audio.qwen import align_audio
from mlx_one.audio.schemas import (
    AlignmentUnit,
    ASRDecodeOptions,
    AudioAlignment,
    AudioStreamEvent,
    QwenASROptions,
    SpeechTranscript,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    WhisperDecodeOptions,
)
from mlx_one.audio.service import transcribe
from mlx_one.audio.streaming import open_audio_stream

__all__ = [
    "ASRDecodeOptions",
    "QwenASROptions",
    "AlignmentUnit",
    "AudioAlignment",
    "AudioStreamEvent",
    "SpeechTranscript",
    "AudioModelLoadError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "WhisperDecodeOptions",
    "load_audio_model",
    "transcribe",
    "align_audio",
    "open_audio_stream",
]

from mlx_one.audio.input import AudioInput
from mlx_one.audio.public import AudioEngine
from mlx_one.audio.schemas import SpeechGenerationResult, SpeechStreamEvent


def generate_speech(*args, **kwargs):
    from mlx_one.audio.lfm import generate_speech as generate

    return generate(*args, **kwargs)


__all__ += ["AudioEngine", "AudioInput", "SpeechGenerationResult", "SpeechStreamEvent",
            "generate_speech"]
