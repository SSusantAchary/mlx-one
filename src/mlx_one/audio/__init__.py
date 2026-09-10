"""Native automatic speech-recognition services."""

from mlx_one.audio.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    WhisperDecodeOptions,
)
from mlx_one.audio.service import transcribe

__all__ = [
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "WhisperDecodeOptions",
    "transcribe",
]
