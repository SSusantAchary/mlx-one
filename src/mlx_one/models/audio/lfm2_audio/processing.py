"""Waveform resampling and log-Mel preprocessing for LFM2.5-Audio."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from mlx_one.models.audio.lfm2_audio.config import AudioPreprocessorConfig


def resample_waveform(waveform: Any, source_rate: int, target_rate: int) -> Any:
    if source_rate == target_rate:
        return waveform
    if source_rate < 1 or target_rate < 1:
        raise ValueError("sample rates must be positive")
    target_length = max(1, round(waveform.shape[-1] * target_rate / source_rate))
    locations = mx.arange(target_length) * (waveform.shape[-1] - 1) / max(target_length - 1, 1)
    left = mx.floor(locations).astype(mx.int32)
    right = mx.minimum(left + 1, waveform.shape[-1] - 1)
    fraction = locations - left
    return waveform[..., left] * (1 - fraction) + waveform[..., right] * fraction


def waveform_to_log_mel(waveform: Any, config: AudioPreprocessorConfig) -> Any:
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [samples] or [batch, samples]")
    if waveform.shape[1] < config.window_length:
        waveform = mx.pad(waveform, ((0, 0), (0, config.window_length - waveform.shape[1])))
    frame_count = 1 + (waveform.shape[1] - config.window_length) // config.hop_length
    frames = mx.stack(
        [
            waveform[
                :,
                index * config.hop_length : index * config.hop_length + config.window_length,
            ]
            for index in range(frame_count)
        ],
        axis=1,
    )
    window = 0.5 - 0.5 * mx.cos(
        2 * math.pi * mx.arange(config.window_length) / config.window_length
    )
    padded = mx.pad(frames * window, ((0, 0), (0, 0), (0, config.n_fft - config.window_length)))
    spectrum = mx.fft.rfft(padded, axis=-1)
    power = mx.abs(spectrum) ** 2
    filters = mel_filterbank(config)
    features = mx.log(mx.maximum(power @ filters.T, 1e-10))
    if config.normalization == "per_feature":
        mean = mx.mean(features, axis=1, keepdims=True)
        variance = mx.mean((features - mean) ** 2, axis=1, keepdims=True)
        features = (features - mean) / mx.sqrt(variance + 1e-5)
    return features


def mel_filterbank(config: AudioPreprocessorConfig) -> Any:
    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)

    def mel_to_hz(value: float) -> float:
        return 700.0 * (10 ** (value / 2595.0) - 1.0)

    low = hz_to_mel(0.0)
    high = hz_to_mel(config.sampling_rate / 2)
    mel_points = [
        low + (high - low) * index / (config.num_mel_bins + 1)
        for index in range(config.num_mel_bins + 2)
    ]
    bins = [
        min(config.n_fft // 2, int((config.n_fft + 1) * mel_to_hz(point) / config.sampling_rate))
        for point in mel_points
    ]
    filters = [[0.0] * (config.n_fft // 2 + 1) for _ in range(config.num_mel_bins)]
    for mel_index in range(config.num_mel_bins):
        left, center, right = bins[mel_index : mel_index + 3]
        for frequency in range(left, center):
            filters[mel_index][frequency] = (frequency - left) / max(center - left, 1)
        for frequency in range(center, right):
            filters[mel_index][frequency] = (right - frequency) / max(right - center, 1)
    return mx.array(filters)
