"""Native bounded STFT/ISTFT for the publisher's audio frontend and codec."""

from __future__ import annotations

import math


def istft(spectrum, *, hop_length, win_length, window, center=False, normalized=True):
    import mlx.core as mx

    frames = mx.fft.irfft(spectrum.T, n=win_length, axis=-1) * window
    count = frames.shape[0]
    length = (count - 1) * hop_length + win_length
    output = mx.zeros((length,), dtype=mx.float32)
    envelope = mx.zeros((length,), dtype=mx.float32)
    # Scatter overlap-add avoids constructing a quadratic stack of padded frames.
    indices = mx.arange(count)[:, None] * hop_length + mx.arange(win_length)[None, :]
    output = output.at[indices.reshape(-1)].add(frames.reshape(-1))
    envelope = envelope.at[indices.reshape(-1)].add(
        mx.broadcast_to(window**2, frames.shape).reshape(-1)
    )
    if normalized:
        output = output / mx.maximum(envelope, 1e-8)
    return output[win_length // 2 : -win_length // 2] if center else output


def preprocess(waveform, config):
    """NeMo-style per-feature mel normalization, inference mode (no dither)."""
    import mlx.core as mx

    from mlx_one.models.audio.parakeet.processing import ParakeetProcessor, mel_filters

    if waveform.ndim != 1 or waveform.size < config.n_fft:
        raise ValueError("LFM speech input requires at least one FFT window of mono PCM")
    if (
        config.sample_rate != 16000
        or config.normalize != "per_feature"
        or config.window != "hann"
        or config.frame_splicing != 1
        or not config.log
    ):
        raise ValueError("unsupported LFM audio frontend configuration")
    x = waveform.astype(mx.float32)
    x = mx.concatenate((x[:1], x[1:] - config.preemph * x[:-1]))
    n, hop, win = config.n_fft, config.hop_length, config.win_length
    x = mx.pad(x, ((n // 2, n // 2),))
    count = 1 + (x.size - n) // hop
    indices = mx.arange(count)[:, None] * hop + mx.arange(n)[None, :]
    window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(win) / (win - 1))
    window = mx.pad(window, ((int((n - win) // 2), int((n - win + 1) // 2)),))
    spectrum = mx.fft.rfft(x[indices] * window, axis=-1)
    power = mx.sqrt(spectrum.real**2 + spectrum.imag**2) ** 2
    filters = mel_filters(ParakeetProcessor(feature_size=config.features))
    features = mx.log(power @ filters.T + 2**-24)
    valid = waveform.size // hop
    if valid < 2:
        raise ValueError("LFM normalization requires at least two frames")
    origin = features[0]
    mean = origin + mx.mean(features[:valid] - origin, axis=0)
    variance = mx.sum((features[:valid] - mean) ** 2, axis=0) / (valid - 1)
    features = (features - mean) / (mx.sqrt(variance) + 1e-5)
    return mx.where(mx.arange(count)[:, None] < valid, features, 0)[None]
