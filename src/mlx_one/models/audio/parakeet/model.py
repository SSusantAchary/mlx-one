"""Native MLX FastConformer and recurrent TDT predictor/joint network.

Adapted from the pinned Apache-2.0 Transformers Parakeet source; see NOTICE.
No model-global decoder state, reference fallback, or implicit precision change.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.parakeet.config import ParakeetConfig


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, x):
        return self.linear2(nn.silu(self.linear1(x)))


class InferenceBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.running_mean = mx.zeros((channels,))
        self.running_var = mx.ones((channels,))
        self.num_batches_tracked = mx.array(0, dtype=mx.int64)

    def __call__(self, x):
        return (
            (x - self.running_mean) * mx.rsqrt(self.running_var + 1e-5)
        ) * self.weight + self.bias


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.pointwise_conv1 = nn.Conv1d(d, 2 * d, 1, bias=config.convolution_bias)
        self.depthwise_conv = nn.Conv1d(
            d,
            d,
            config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=d,
            bias=config.convolution_bias,
        )
        self.norm = InferenceBatchNorm(d)
        self.pointwise_conv2 = nn.Conv1d(d, d, 1, bias=config.convolution_bias)

    def __call__(self, x, valid):
        first, gate = mx.split(self.pointwise_conv1(x), 2, axis=-1)
        x = first * mx.sigmoid(gate)
        if valid is not None:
            x = mx.where(valid[..., None], x, 0)
        return self.pointwise_conv2(nn.silu(self.norm(self.depthwise_conv(x))))


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.heads = config.num_attention_heads
        self.head_dim = d // self.heads
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(d, d, bias=config.attention_bias))
        self.relative_k_proj = nn.Linear(d, d, bias=False)
        self.bias_u = mx.zeros((self.heads, self.head_dim))
        self.bias_v = mx.zeros((self.heads, self.head_dim))

    @staticmethod
    def relative_shift(scores):
        b, h, t, p = scores.shape
        padded = mx.pad(scores, ((0, 0), (0, 0), (0, 0), (1, 0)))
        return padded.reshape(b, h, -1, t)[:, :, 1:].reshape(b, h, t, p)

    def __call__(self, x, positions, valid):
        b, t, _ = x.shape

        def heads(value):
            return value.reshape(b, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        position_keys = heads(self.relative_k_proj(positions))
        positional = (
            self.relative_shift((q + self.bias_v[None, :, None]) @ position_keys.swapaxes(-1, -2))[
                ..., :t
            ]
            * self.head_dim**-0.5
        )
        if valid is not None:
            mask = valid[:, None, :, None] & valid[:, None, None, :]
            positional = mx.where(mask, positional, -3.4028234663852886e38)
        output = mx.fast.scaled_dot_product_attention(
            q + self.bias_u[None, :, None],
            k,
            v,
            scale=self.head_dim**-0.5,
            mask=positional,
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(b, t, -1))


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feed_forward1 = FeedForward(config)
        self.feed_forward2 = FeedForward(config)
        self.self_attn = RelativeAttention(config)
        self.conv = Convolution(config)
        for name in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ):
            setattr(self, name, nn.LayerNorm(config.hidden_size, eps=1e-5))

    def __call__(self, x, positions, valid):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), positions, valid)
        x = x + self.conv(self.norm_conv(x), valid)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Subsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.subsampling_conv_channels
        self.layers = [nn.Conv2d(1, c, 3, stride=2, padding=1), nn.ReLU()]
        for _ in range(2):
            self.layers.extend(
                [
                    nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c),
                    nn.Conv2d(c, c, 1),
                    nn.ReLU(),
                ]
            )
        self.linear = nn.Linear(c * (config.num_mel_bins // 8), config.hidden_size)

    def __call__(self, features, lengths):
        x = features[..., None]
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                if layer.stride != (1, 1):
                    lengths = (lengths + 1) // 2
                valid = mx.arange(x.shape[1])[None, :] < lengths[:, None]
                x = mx.where(valid[:, :, None, None], x, 0)
        # Torch flattens [channels, frequency], not [frequency, channels].
        x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.linear(x), lengths


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.subsampling = Subsampling(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, features, lengths, *, check=None):
        x, lengths = self.subsampling(features, lengths)
        if self.config.scale_input:
            x = x * math.sqrt(self.config.hidden_size)
        t, d = x.shape[1], self.config.hidden_size
        inv = 1.0 / (10000.0 ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
        freqs = mx.arange(t - 1, -t, -1, dtype=mx.float32)[:, None] * inv[None]
        positions = mx.stack((mx.sin(freqs), mx.cos(freqs)), axis=-1).reshape(1, 2 * t - 1, d)
        positions = mx.broadcast_to(positions.astype(x.dtype), (x.shape[0], 2 * t - 1, d))
        valid = mx.arange(t)[None, :] < lengths[:, None]
        for layer in self.layers:
            if check is not None:
                check()
            x = layer(x, positions, valid)
            mx.eval(x)
        return x, lengths


class PredictorLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.decoder_hidden_size
        self.num_layers = config.num_decoder_layers
        for index in range(self.num_layers):
            for side in ("ih", "hh"):
                setattr(
                    self,
                    f"weight_{side}_l{index}",
                    mx.zeros((4 * self.hidden_size, self.hidden_size)),
                )
                setattr(self, f"bias_{side}_l{index}", mx.zeros((4 * self.hidden_size,)))

    def __call__(self, x, state=None):
        if state is None:
            h = c = mx.zeros((self.num_layers, x.shape[0], self.hidden_size), dtype=x.dtype)
        else:
            h, c = state
        hidden, cells = [], []
        for index in range(self.num_layers):
            gates = (
                x @ getattr(self, f"weight_ih_l{index}").T
                + getattr(self, f"bias_ih_l{index}")
                + h[index] @ getattr(self, f"weight_hh_l{index}").T
                + getattr(self, f"bias_hh_l{index}")
            )
            inp, forget, candidate, out = mx.split(gates, 4, axis=-1)
            cell = mx.sigmoid(forget) * c[index] + mx.sigmoid(inp) * mx.tanh(candidate)
            x = mx.sigmoid(out) * mx.tanh(cell)
            hidden.append(x)
            cells.append(cell)
        return x, (mx.stack(hidden), mx.stack(cells))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.decoder_hidden_size)
        self.lstm = PredictorLSTM(config)
        self.decoder_projector = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)

    def __call__(self, token, state=None):
        tokens=mx.array([token]) if isinstance(token,int) else token
        if tokens.ndim!=1:
            raise ValueError("TDT predictor tokens require one ID per row")
        x, state = self.lstm(self.embedding(tokens), state)
        return self.decoder_projector(x), state


class Joint(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config.decoder_hidden_size, config.vocab_size + len(config.durations))

    def __call__(self, encoder, decoder):
        return self.head(nn.relu(encoder + decoder))


class ParakeetTDT(nn.Module):
    def __init__(self, config: ParakeetConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.encoder_projector = nn.Linear(config.hidden_size, config.decoder_hidden_size)
        self.decoder = Decoder(config)
        self.joint = Joint(config)

    def encode(self, features, lengths, *, check=None):
        x, lengths = self.encoder(features, lengths, check=check)
        return self.encoder_projector(x), lengths
