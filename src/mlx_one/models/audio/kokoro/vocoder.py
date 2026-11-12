"""Native Kokoro harmonic excitation, adaptive residual vocoder and ISTFT.

Channels-last convolutions and MLX FFTs perform all neural/spectral execution.
The existing native overlap-add ISTFT is reused with Kokoro's periodic Hann
window, n_fft=20, hop=5 and centered output.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.kokoro.blocks import AdaIN, AdaResBlock, Conv
from mlx_one.models.audio.lfm2_audio.dsp import istft


def linear_upsample(value, scale):
    positions = (mx.arange(value.shape[1] * scale, dtype=mx.float32) + 0.5) / scale - 0.5
    positions = mx.clip(positions, 0, value.shape[1] - 1)
    low = mx.floor(positions).astype(mx.int32)
    high = mx.minimum(low + 1, value.shape[1] - 1)
    weight = positions - low
    return (
        value[:, low, :] * (1 - weight[None, :, None]) + value[:, high, :] * weight[None, :, None]
    )


def source_stft(waveform):
    # Torch's centered STFT uses reflection padding and a periodic Hann window.
    value = mx.concatenate(
        (waveform[:, 1:11][:, ::-1], waveform, waveform[:, -11:-1][:, ::-1]), axis=1
    )
    indices = mx.arange(1 + (value.shape[1] - 20) // 5)[:, None] * 5 + mx.arange(20)[None, :]
    window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(20) / 20)
    spectrum = mx.fft.rfft(value[:, indices] * window, axis=-1)
    # A real even-length FFT has exactly real DC/Nyquist bins. Roundoff in a
    # generic FFT must not flip atan2 between -pi and +pi at those endpoints:
    # phase is itself an input feature to this particular vocoder.
    imaginary = mx.concatenate(
        (
            mx.zeros_like(spectrum.imag[..., :1]),
            spectrum.imag[..., 1:-1],
            mx.zeros_like(spectrum.imag[..., -1:]),
        ),
        axis=-1,
    )
    return mx.concatenate((mx.abs(spectrum), mx.arctan2(imaginary, spectrum.real)), axis=-1)


class HarmonicSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_linear = nn.Linear(9, 1)

    def __call__(self, f0, forcing=None):
        if forcing is None:
            frequency = f0[..., None] * mx.arange(1, 10)[None, None, :]
            phase = mx.cumsum((frequency / 24000) % 1, axis=1) * (2 * math.pi * 300)
            phase = linear_upsample(phase, 300)
            voiced = mx.repeat((f0 > 10)[..., None], 300, axis=1).astype(mx.float32)
            noise_scale = voiced * 0.003 + (1 - voiced) * (0.1 / 3)
            forcing = mx.sin(phase) * 0.1 * voiced + mx.random.normal(phase.shape) * noise_scale
        return mx.tanh(self.l_linear(forcing))[..., 0]


class SnakeResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = [
            Conv(channels, channels, kernel, padding=d * (kernel - 1) // 2, dilation=d)
            for d in dilations
        ]
        self.convs2 = [
            Conv(channels, channels, kernel, padding=(kernel - 1) // 2) for _ in dilations
        ]
        self.adain1 = [AdaIN(channels) for _ in dilations]
        self.adain2 = [AdaIN(channels) for _ in dilations]
        self.alpha1 = [mx.ones((1, channels, 1)) for _ in dilations]
        self.alpha2 = [mx.ones((1, channels, 1)) for _ in dilations]

    def __call__(self, x, style):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
            strict=True,
        ):
            a1, a2 = a1.transpose(0, 2, 1), a2.transpose(0, 2, 1)
            value = n1(x, style)
            value = c1(value + mx.sin(a1 * value) ** 2 / a1)
            value = n2(value, style)
            x = x + c2(value + mx.sin(a2 * value) ** 2 / a2)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_source = HarmonicSource()
        self.ups = [
            nn.ConvTranspose1d(512, 256, 20, stride=10, padding=5),
            nn.ConvTranspose1d(256, 128, 12, stride=6, padding=3),
        ]
        self.noise_convs = [Conv(22, 256, 12, stride=6, padding=3), Conv(22, 128, 1)]
        self.noise_res = [SnakeResBlock(256, 7), SnakeResBlock(128, 11)]
        self.resblocks = [SnakeResBlock(ch, k) for ch in (256, 128) for k in (3, 7, 11)]
        self.conv_post = Conv(128, 22, 7, padding=3)

    def __call__(self, x, style, f0, forcing=None):
        harmonic = source_stft(self.m_source(f0, forcing))
        for i in range(2):
            source = self.noise_res[i](self.noise_convs[i](harmonic), style)
            x = self.ups[i](nn.leaky_relu(x, negative_slope=0.1))
            if i == 1:
                x = mx.concatenate((x[:, 1:2, :], x), axis=1)
            if x.shape != source.shape:
                raise ValueError("Kokoro harmonic and upsampling frame counts differ")
            x = x + source
            x = sum(self.resblocks[i * 3 + j](x, style) for j in range(3)) / 3
        value = self.conv_post(nn.leaky_relu(x, negative_slope=0.01))
        magnitude = mx.exp(value[0, :, :11])
        phase = mx.sin(value[0, :, 11:])
        spectrum = magnitude * (mx.cos(phase) + 1j * mx.sin(phase))
        window = 0.5 - 0.5 * mx.cos(2 * math.pi * mx.arange(20) / 20)
        return istft(
            spectrum.T, hop_length=5, win_length=20, window=window, center=True, normalized=True
        )


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = AdaResBlock(514, 1024)
        self.decode = [
            AdaResBlock(1090, 1024),
            AdaResBlock(1090, 1024),
            AdaResBlock(1090, 1024),
            AdaResBlock(1090, 512, True),
        ]
        self.F0_conv, self.N_conv = (
            Conv(1, 1, 3, stride=2, padding=1),
            Conv(1, 1, 3, stride=2, padding=1),
        )
        self.asr_res = [Conv(512, 64, 1)]
        self.generator = Generator()

    def __call__(self, asr, f0, noise, style, forcing=None):
        frequency = self.F0_conv(f0[..., None])
        amplitude = self.N_conv(noise[..., None])
        x = self.encode(mx.concatenate((asr, frequency, amplitude), axis=-1), style)
        residual = self.asr_res[0](asr)
        for block in self.decode:
            x = block(mx.concatenate((x, residual, frequency, amplitude), axis=-1), style)
        return self.generator(x, style, f0, forcing)
