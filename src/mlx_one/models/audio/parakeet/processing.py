"""Parakeet-specific centered STFT, preemphasis and per-feature normalization."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParakeetProcessor:
    feature_size: int = 128
    sampling_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    preemphasis: float = 0.97

    @classmethod
    def from_dict(cls, data: dict) -> ParakeetProcessor:
        feature = data.get("feature_extractor")
        if not isinstance(feature, dict) or data.get("processor_class") != "ParakeetProcessor":
            raise ValueError("Parakeet requires processor_config.json with a feature extractor")
        expected = {
            "feature_extractor_type": "ParakeetFeatureExtractor",
            "sampling_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "preemphasis": 0.97,
            "padding_value": 0.0,
            "padding_side": "right",
            "return_attention_mask": True,
        }
        if any(feature.get(key, value) != value for key, value in expected.items()):
            raise ValueError("unsupported Parakeet preprocessing settings")
        return cls(feature_size=feature["feature_size"])


def mel_filters(config: ParakeetProcessor) -> Any:
    """Librosa/Slaney filter construction, including its float32 triangle rounding."""
    import mlx.core as mx

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    def hz_to_mel(f):
        return f / (200 / 3) if f < 1000 else 15 + math.log(f / 1000) / (math.log(6.4) / 27)

    def mel_to_hz(m):
        return m * (200 / 3) if m < 15 else 1000 * math.exp((math.log(6.4) / 27) * (m - 15))

    maximum = hz_to_mel(config.sampling_rate / 2)
    points = [
        mel_to_hz(maximum * i / (config.feature_size + 1)) for i in range(config.feature_size + 2)
    ]
    filters = []
    for i in range(config.feature_size):
        low, center, high = points[i : i + 3]
        scale = 2 / (high - low)
        row = []
        for j in range(config.n_fft // 2 + 1):
            hz = j * config.sampling_rate / config.n_fft
            triangle = max(0.0, min((hz - low) / (center - low), (high - hz) / (high - center)))
            row.append(f32(f32(triangle) * scale))
        filters.append(row)
    return mx.array(filters, dtype=mx.float32)


def preprocess(waveform: Any, config: ParakeetProcessor) -> tuple[Any, int]:
    import mlx.core as mx

    if waveform.ndim != 1 or not mx.all(mx.isfinite(waveform)).item():
        raise ValueError("Parakeet waveform must contain finite mono samples")
    valid_frames = waveform.shape[0] // config.hop_length
    if valid_frames < 2:
        raise ValueError("nonempty Parakeet audio requires at least 320 samples")
    x = waveform.astype(mx.float32)
    x = mx.concatenate((x[:1], x[1:] - config.preemphasis * x[:-1]))
    x = mx.pad(x, ((config.n_fft // 2, config.n_fft // 2),))
    count = 1 + (x.shape[0] - config.n_fft) // config.hop_length
    frames = x[mx.arange(count)[:, None] * config.hop_length + mx.arange(config.n_fft)[None]]
    window = 0.5 - 0.5 * mx.cos(
        2 * math.pi * mx.arange(config.win_length) / (config.win_length - 1)
    )
    window = mx.pad(
        window,
        (((config.n_fft - config.win_length) // 2, (config.n_fft - config.win_length) // 2),),
    )
    spectrum = mx.fft.rfft(frames * window, axis=-1)
    # Preserve the reference sqrt-then-square path rather than changing its rounding.
    magnitude = mx.sqrt(spectrum.real**2 + spectrum.imag**2) ** 2
    features = mx.log(magnitude @ mel_filters(config).T + 2**-24)
    valid = features[:valid_frames]
    # Center the reduction before summing. This is the same per-feature mean,
    # but avoids an FP32 summation residual becoming a large normalized value
    # when the reference variance is zero (for example, digital silence).
    origin = valid[0]
    mean = origin + mx.mean(valid - origin, axis=0)
    variance = mx.sum((valid - mean) ** 2, axis=0) / (valid_frames - 1)
    normalized = (features - mean) / (mx.sqrt(variance) + 1e-5)
    normalized = mx.where(mx.arange(count)[:, None] < valid_frames, normalized, 0.0)
    return normalized[None], valid_frames
