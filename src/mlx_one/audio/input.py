"""Bounded host audio decoding, channel mixing and band-limited resampling."""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_BYTES = 25 * 1024**2
MAX_SECONDS = 600
_DECODE_SLOTS = threading.BoundedSemaphore(2)
FORMATS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mp4": "mov",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/webm": "matroska",
}


@dataclass(frozen=True)
class AudioInput:
    source: Any
    sample_rate: int | None = None
    media_type: str | None = None
    offset_seconds: float = 0.0


def prepare_audio(
    source, *, sample_rate=None, media_type=None, max_seconds=MAX_SECONDS, cancelled=None
):
    """Return finite mono float32 host PCM at 16 kHz, without peak normalization."""
    import numpy as np

    offset = 0.0
    if isinstance(source, AudioInput):
        sample_rate, media_type = source.sample_rate, source.media_type
        offset, source = source.offset_seconds, source.source
    if not np.isfinite(offset) or offset < 0:
        raise ValueError("audio offset must be finite and non-negative")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("audio duration bound must be finite and positive")
    if cancelled and cancelled():
        raise RuntimeError("audio preparation cancelled")
    if isinstance(source, (bytes, bytearray, memoryview)):
        if not source or len(source) > MAX_BYTES:
            raise ValueError("encoded audio must contain 1 to 25 MiB of data")
        fmt = FORMATS.get((media_type or "").partition(";")[0].strip().lower())
        if fmt is None:
            raise ValueError("encoded audio requires a supported audio MIME type")
        with tempfile.TemporaryDirectory(prefix="mlx-one-audio-") as folder:
            path = Path(folder) / ("upload." + fmt)
            path.write_bytes(source)
            waveform = _decode(path, max_seconds, cancelled, fmt)
    elif isinstance(source, (str, Path)):
        waveform = _decode(Path(source).expanduser(), max_seconds, cancelled)
    else:
        if type(sample_rate) is not int or not 8000 <= sample_rate <= 192000:
            raise ValueError("waveform input requires sample_rate between 8000 and 192000")
        # Bound before copying untrusted user arrays.
        if len(source) > max_seconds * sample_rate:
            raise ValueError("audio exceeds its duration limit")
        waveform = np.asarray(source)
        if waveform.ndim not in {1, 2} or waveform.ndim == 2 and waveform.shape[1] not in {1, 2}:
            raise ValueError("waveform must be [samples] or [samples, 1 or 2 channels]")
        if waveform.dtype == np.int16:
            waveform = waveform.astype(np.float32) / 32768
        elif waveform.dtype.kind not in "f":
            raise ValueError("waveform must be floating point or explicitly int16 PCM")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform samples must be finite")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = resample(waveform, sample_rate, 16000, cancelled=cancelled)
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all() or waveform.size > max_seconds * 16000:
        raise ValueError("decoded audio exceeds duration bound or contains non-finite samples")
    return waveform


def _decode(path, seconds, cancelled, fmt=None):
    from mlx_one.models.audio.whisper.processing import AudioProcessingError, decode_audio

    # Cancellation remains responsive while waiting for a bounded decoder slot.
    while not _DECODE_SLOTS.acquire(timeout=0.05):
        if cancelled and cancelled():
            raise RuntimeError("audio preparation cancelled")
    try:
        return decode_audio(
            path,
            max_duration_seconds=seconds,
            max_file_bytes=MAX_BYTES,
            cancelled=cancelled,
            input_format=fmt,
        )
    except AudioProcessingError as exc:
        raise ValueError(str(exc)) from exc
    finally:
        _DECODE_SLOTS.release()


def resample(waveform, source_rate, target_rate, *, cancelled=None):
    """Windowed sinc interpolation, low-pass cutoff adjusted for downsampling."""
    import numpy as np

    waveform = np.asarray(waveform, dtype=np.float32)
    if source_rate == target_rate or not waveform.size:
        return waveform.copy()
    count = round(waveform.size * target_rate / source_rate)
    output = np.empty(count, dtype=np.float32)
    cutoff = min(1.0, target_rate / source_rate)
    radius = int(32 / cutoff)
    taps = np.arange(-radius + 1, radius + 1)
    for start in range(0, count, 4096):
        if cancelled and cancelled():
            raise RuntimeError("audio resampling cancelled")
        positions = np.arange(start, min(start + 4096, count)) * source_rate / target_rate
        indices = np.floor(positions).astype(np.int64)[:, None] + taps
        delta = positions[:, None] - indices
        kernel = (
            cutoff
            * np.sinc(cutoff * delta)
            * np.where(np.abs(delta) < radius, 0.5 + 0.5 * np.cos(np.pi * delta / radius), 0)
        )
        kernel /= kernel.sum(axis=1, keepdims=True)
        output[start : start + len(positions)] = (
            waveform[np.clip(indices, 0, len(waveform) - 1)] * kernel
        ).sum(axis=1)
    return output
