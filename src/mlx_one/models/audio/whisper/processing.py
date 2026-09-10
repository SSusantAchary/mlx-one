"""Whisper-compatible FFmpeg decoding and log-Mel preprocessing."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_SECONDS = 30
N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
N_FRAMES = N_SAMPLES // HOP_LENGTH


class AudioProcessingError(RuntimeError):
    """Raised when audio cannot be decoded or converted to Whisper features."""


@dataclass(frozen=True)
class WhisperProcessorConfig:
    feature_size: int = 80
    sampling_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    chunk_length: int = CHUNK_SECONDS
    mel_filters: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def from_directory(cls, directory: str | Path) -> WhisperProcessorConfig:
        path = Path(directory) / "preprocessor_config.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AudioProcessingError(f"cannot read preprocessor_config.json: {exc}") from exc
        filters = data.get("mel_filters")
        return cls(
            feature_size=int(data.get("feature_size", 80)),
            sampling_rate=int(data.get("sampling_rate", SAMPLE_RATE)),
            n_fft=int(data.get("n_fft", N_FFT)),
            hop_length=int(data.get("hop_length", HOP_LENGTH)),
            chunk_length=int(data.get("chunk_length", CHUNK_SECONDS)),
            mel_filters=(
                tuple(tuple(float(value) for value in row) for row in filters)
                if isinstance(filters, list)
                else None
            ),
        )


def decode_audio(path: str | Path, *, ffmpeg: str = "ffmpeg") -> Any:
    """Decode any FFmpeg-supported file to mono 16-kHz float32 samples."""

    source = Path(path)
    if not source.is_file():
        raise AudioProcessingError(f"audio file does not exist: {source}")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise AudioProcessingError("ffmpeg is required to decode audio file inputs")
    command = [
        executable,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(source),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except OSError as exc:
        raise AudioProcessingError(f"cannot execute ffmpeg: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise AudioProcessingError(
            "ffmpeg could not decode audio" + (f": {detail[-1]}" if detail else "")
        )
    if len(result.stdout) % 4:
        raise AudioProcessingError("ffmpeg returned malformed float32 audio")
    import mlx.core as mx

    return mx.array(memoryview(result.stdout).cast("f"))


def pad_or_trim(waveform: Any, length: int = N_SAMPLES) -> Any:
    import mlx.core as mx

    if waveform.ndim != 1:
        raise AudioProcessingError("waveform must be mono with shape [samples]")
    if waveform.shape[0] > length:
        return waveform[:length]
    if waveform.shape[0] < length:
        return mx.pad(waveform, ((0, length - waveform.shape[0]),))
    return waveform


def log_mel_spectrogram(
    waveform: Any,
    *,
    num_mel_bins: int,
    mel_filters: Any | None = None,
    padding: int = 0,
) -> Any:
    """Return normalized Whisper log-Mel features as [mel, frames]."""

    import mlx.core as mx

    if waveform.ndim != 1:
        raise AudioProcessingError("waveform must be mono with shape [samples]")
    if padding:
        waveform = mx.pad(waveform, ((0, padding),))
    waveform = mx.pad(waveform, ((N_FFT // 2, N_FFT // 2),), mode="reflect")
    frame_count = 1 + (waveform.shape[0] - N_FFT) // HOP_LENGTH
    frames = mx.stack(
        [waveform[index * HOP_LENGTH : index * HOP_LENGTH + N_FFT] for index in range(frame_count)]
    )
    window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(N_FFT) / N_FFT)
    spectrum = mx.fft.rfft(frames * window, axis=-1)
    magnitudes = mx.abs(spectrum) ** 2
    filters = mel_filterbank(num_mel_bins) if mel_filters is None else mx.array(mel_filters)
    mel = magnitudes[:-1] @ filters.T
    log_spec = mx.log10(mx.maximum(mel, 1e-10))
    log_spec = mx.maximum(log_spec, mx.max(log_spec) - 8.0)
    return ((log_spec + 4.0) / 4.0).T


def mel_filterbank(num_mel_bins: int) -> Any:
    """Generate librosa-compatible Slaney-normalized filters."""

    import mlx.core as mx

    if num_mel_bins not in {80, 128}:
        raise AudioProcessingError("Whisper supports 80 or 128 mel bins")

    def hz_to_mel(frequency: float) -> float:
        f_sp = 200.0 / 3
        if frequency < 1000:
            return frequency / f_sp
        return 15 + math.log(frequency / 1000) / (math.log(6.4) / 27)

    def mel_to_hz(mel: float) -> float:
        f_sp = 200.0 / 3
        if mel < 15:
            return f_sp * mel
        return 1000 * math.exp((math.log(6.4) / 27) * (mel - 15))

    mel_min, mel_max = hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2)
    mel_points = [
        mel_min + (mel_max - mel_min) * index / (num_mel_bins + 1)
        for index in range(num_mel_bins + 2)
    ]
    frequencies = [mel_to_hz(value) for value in mel_points]
    fft_frequencies = [index * SAMPLE_RATE / N_FFT for index in range(N_FFT // 2 + 1)]
    filters = [[0.0] * len(fft_frequencies) for _ in range(num_mel_bins)]
    for index in range(num_mel_bins):
        lower, center, upper = frequencies[index : index + 3]
        scale = 2.0 / (upper - lower)
        for frequency_index, frequency in enumerate(fft_frequencies):
            lower_slope = (frequency - lower) / (center - lower)
            upper_slope = (upper - frequency) / (upper - center)
            filters[index][frequency_index] = max(0.0, min(lower_slope, upper_slope)) * scale
    return mx.array(filters)


def whisper_features(waveform: Any, config: WhisperProcessorConfig) -> Any:
    waveform = pad_or_trim(waveform, config.sampling_rate * config.chunk_length)
    features = log_mel_spectrogram(
        waveform,
        num_mel_bins=config.feature_size,
        mel_filters=config.mel_filters,
    )
    return features[None, ...]
