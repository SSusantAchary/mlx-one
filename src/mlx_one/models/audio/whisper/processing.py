"""Whisper-compatible FFmpeg decoding and log-Mel preprocessing."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from collections.abc import Callable
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


def decode_audio(
    path: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    cancelled: Callable[[], bool] | None = None,
    max_duration_seconds: float | None = None,
    max_file_bytes: int | None = None,
    timeout_seconds: float = 120,
    input_format: str | None = None,
) -> Any:
    """Decode local audio with a timeout and optional byte/duration limits.

    HTTP callers must supply both limits. Python's existing long-form file
    workflow remains available; FFmpeg cannot fetch remote playlist resources.
    """

    for name, value in (("max_duration_seconds", max_duration_seconds),
                        ("timeout_seconds", timeout_seconds)):
        if (value is None and name == "max_duration_seconds"):
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise AudioProcessingError(f"{name} must be finite and positive")
    if max_file_bytes is not None and (type(max_file_bytes) is not int or max_file_bytes < 1):
        raise AudioProcessingError("max_file_bytes must be a positive integer")
    if input_format is not None and input_format not in {"wav", "mp3", "ogg", "mov", "matroska"}:
        raise AudioProcessingError("unsupported bounded audio input format")

    source = Path(path)
    if not source.is_file():
        raise AudioProcessingError(f"audio file does not exist: {source}")
    if max_file_bytes is not None and source.stat().st_size > max_file_bytes:
        raise AudioProcessingError("audio file exceeds the byte limit")
    executable = shutil.which(ffmpeg)
    if executable is None:
        raise AudioProcessingError("ffmpeg is required to decode audio file inputs")
    command = [
        executable,
        "-nostdin",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file,pipe",
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
    max_samples = None
    if input_format is not None:
        position = command.index("-i")
        command[position:position] = ["-f", input_format]
    if max_duration_seconds is not None:
        max_samples = int(max_duration_seconds * SAMPLE_RATE)
        # One extra sample distinguishes oversized input from a valid boundary.
        command[-1:-1] = ["-t", str((max_samples + 1) / SAMPLE_RATE)]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancelled is not None and cancelled():
                _stop_decoder(process)
                raise AudioProcessingError("audio decoding was cancelled")
            if time.monotonic() >= deadline:
                _stop_decoder(process)
                raise AudioProcessingError("audio decoding timed out")
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    except OSError as exc:
        raise AudioProcessingError(f"cannot execute ffmpeg: {exc}") from exc
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise AudioProcessingError(
            "ffmpeg could not decode audio" + (f": {detail[-1]}" if detail else "")
        )
    if len(stdout) % 4:
        raise AudioProcessingError("ffmpeg returned malformed float32 audio")
    if max_samples is not None and len(stdout) // 4 > max_samples:
        raise AudioProcessingError("audio exceeds the duration limit")
    import numpy as np

    if not stdout or not np.isfinite(np.frombuffer(stdout, dtype="<f4")).all():
        raise AudioProcessingError("decoded audio is empty or non-finite")
    import mlx.core as mx

    return mx.array(memoryview(stdout).cast("f"))


def _stop_decoder(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=1)


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
