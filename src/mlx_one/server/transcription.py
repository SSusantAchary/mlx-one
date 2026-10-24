"""Private native Whisper adapter for the HTTP transcription route."""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from mlx_one.audio import WhisperDecodeOptions, transcribe
from mlx_one.models.audio.whisper.loading import LoadedWhisper
from mlx_one.models.audio.whisper.processing import AudioProcessingError, decode_audio

MAX_AUDIO_BYTES = 25 * 1024**2
MAX_AUDIO_SECONDS = 300
SUPPORTED_AUDIO_TYPES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
}
_INPUT_FORMATS = {".webm": "matroska", ".ogg": "ogg", ".m4a": "mov",
                  ".mp3": "mp3", ".wav": "wav"}


class AudioUploadError(ValueError):
    """Raised when an HTTP audio upload is unsupported or malformed."""


class NativeTranscriptionService:
    family = "whisper"

    """Convert bounded browser audio uploads with a private Whisper bundle."""

    def __init__(self, bundle: LoadedWhisper) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for browser audio transcription")
        self.bundle = bundle

    def capabilities(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "accepted_formats": sorted(SUPPORTED_AUDIO_TYPES),
            "max_bytes": MAX_AUDIO_BYTES,
            "max_duration_seconds": MAX_AUDIO_SECONDS,
            "language_detection": True,
            "model_memory_bytes": None,
        }

    def __call__(
        self,
        audio: bytes,
        *,
        media_type: str,
        language: str | None = None,
        task: str = "transcribe",
        cancel: threading.Event | None = None,
        word_timestamps: bool = False,
    ) -> Any:
        normalized = media_type.partition(";")[0].strip().lower()
        suffix = SUPPORTED_AUDIO_TYPES.get(normalized)
        if suffix is None:
            raise AudioUploadError(f"unsupported audio content type: {normalized or 'missing'}")
        if not audio:
            raise AudioUploadError("audio file is empty")
        if len(audio) > MAX_AUDIO_BYTES:
            raise AudioUploadError("audio file exceeds 25 MiB")
        if task not in {"transcribe", "translate"}:
            raise AudioUploadError("task must be 'transcribe' or 'translate'")
        handle = tempfile.NamedTemporaryFile(prefix="mlx-one-audio-", suffix=suffix, delete=False)
        path = Path(handle.name)
        try:
            with handle:
                handle.write(audio)
            try:
                waveform = decode_audio(
                    path, cancelled=cancel.is_set if cancel is not None else None,
                    max_duration_seconds=MAX_AUDIO_SECONDS, max_file_bytes=MAX_AUDIO_BYTES,
                    timeout_seconds=120,
                    input_format=_INPUT_FORMATS[suffix],
                )
            except AudioProcessingError as exc:
                raise AudioUploadError(str(exc)) from exc
            return transcribe(
                self.bundle,
                waveform,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                sample_rate=16000,
                cancelled=cancel.is_set if cancel is not None else None,
            )
        finally:
            path.unlink(missing_ok=True)

    def warmup(self) -> None:
        import mlx.core as mx

        transcribe(
            self.bundle,
            mx.zeros((1600,)),
            language="en",
            options=WhisperDecodeOptions(max_tokens=1, beam_size=1, best_of=1),
        )
