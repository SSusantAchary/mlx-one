"""Native channels-last Kokoro acoustic blocks and shared ALBERT encoder.

Publisher weight normalization is folded offline. Recurrent gates retain
PyTorch's i/f/g/o ordering; both directions execute MLX's native LSTM primitive.
"""

import math
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.embeddings.bert.model import BertSelfAttention, _activation
from mlx_one.models.shared.encoder import bidirectional_attention_mask


class Conv(nn.Module):
    def __init__(self, inp, out, kernel, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.weight = mx.zeros((out, kernel, inp))
        self.bias = mx.zeros((out,)) if bias else None
        self.stride, self.padding, self.dilation = stride, padding, dilation

    def __call__(self, x):
        value = mx.conv1d(x, self.weight, self.stride, self.padding, self.dilation)
        return value if self.bias is None else value + self.bias


class DepthwiseUpsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = mx.zeros((channels, 3, 1))
        self.bias = mx.zeros((channels,))
        self.channels = channels

    def __call__(self, x):
        zeros = mx.zeros_like(x)
        interleaved = mx.stack((x, zeros), axis=2).reshape(x.shape[0], x.shape[1] * 2, x.shape[2])
        return (
            mx.conv1d(interleaved, self.weight[:, ::-1, :], padding=1, groups=self.channels)
            + self.bias
        )


class BiLSTM(nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.forward_lstm = nn.LSTM(inp, hidden)
        self.backward_lstm = nn.LSTM(inp, hidden)

    def __call__(self, x):
        forward, _ = self.forward_lstm(x)
        backward, _ = self.backward_lstm(x[:, ::-1, :])
        return mx.concatenate((forward, backward[:, ::-1, :]), axis=-1)


class InstanceNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, x):
        mean = mx.mean(x, axis=1, keepdims=True)
        variance = mx.mean((x - mean) ** 2, axis=1, keepdims=True)
        return (x - mean) * mx.rsqrt(variance + 1e-5) * self.weight + self.bias


class AdaIN(nn.Module):
    def __init__(self, channels, style=128):
        super().__init__()
        self.norm = InstanceNorm(channels)
        self.fc = nn.Linear(style, channels * 2)

    def __call__(self, x, style):
        gamma, beta = mx.split(self.fc(style), 2, axis=-1)
        return (1 + gamma[:, None, :]) * self.norm(x) + beta[:, None, :]


class AdaLayerNorm(nn.Module):
    def __init__(self, channels=512, style=128):
        super().__init__()
        self.fc = nn.Linear(style, channels * 2)

    def __call__(self, x, style):
        gamma, beta = mx.split(self.fc(style), 2, axis=-1)
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (1 + gamma[:, None, :]) * (x - mean) * mx.rsqrt(variance + 1e-5) + beta[:, None, :]


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = mx.ones((width,))
        self.beta = mx.zeros((width,))

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + 1e-5) * self.gamma + self.beta


class AdaResBlock(nn.Module):
    def __init__(self, inp, out, upsample=False):
        super().__init__()
        self.conv1, self.conv2 = Conv(inp, out, 3, padding=1), Conv(out, out, 3, padding=1)
        self.norm1, self.norm2 = AdaIN(inp), AdaIN(out)
        self.conv1x1 = Conv(inp, out, 1, bias=False) if inp != out else None
        self.pool = DepthwiseUpsample(inp) if upsample else None
        self.upsample = upsample

    def __call__(self, x, style):
        shortcut = mx.repeat(x, 2, axis=1) if self.upsample else x
        if self.conv1x1 is not None:
            shortcut = self.conv1x1(shortcut)
        value = nn.leaky_relu(self.norm1(x, style), negative_slope=0.2)
        if self.pool is not None:
            value = self.pool(value)
        value = self.conv1(value)
        value = self.conv2(nn.leaky_relu(self.norm2(value, style), negative_slope=0.2))
        return (value + shortcut) / math.sqrt(2)


class AlbertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(178, 128)
        self.position_embeddings = nn.Embedding(512, 128)
        self.token_type_embeddings = nn.Embedding(2, 128)
        self.LayerNorm = nn.LayerNorm(128, eps=1e-12)

    def __call__(self, ids):
        return self.LayerNorm(
            self.word_embeddings(ids)
            + self.position_embeddings(mx.arange(ids.shape[1]))
            + self.token_type_embeddings(mx.zeros_like(ids))
        )


class AlbertAttention(BertSelfAttention):
    def __init__(self):
        cfg = SimpleNamespace(
            num_attention_heads=12, hidden_size=768, attention_probs_dropout_prob=0
        )
        super().__init__(cfg)
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = nn.LayerNorm(768, eps=1e-12)

    def __call__(self, x, mask):
        return self.LayerNorm(x + self.dense(super().__call__(x, mask)))


class AlbertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AlbertAttention()
        self.ffn = nn.Linear(768, 2048)
        self.ffn_output = nn.Linear(2048, 768)
        self.full_layer_layer_norm = nn.LayerNorm(768, eps=1e-12)

    def __call__(self, x, mask):
        x = self.attention(x, mask)
        return self.full_layer_layer_norm(x + self.ffn_output(_activation(self.ffn(x), "gelu_new")))


class AlbertGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.albert_layers = [AlbertLayer()]


class AlbertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(128, 768)
        self.albert_layer_groups = [AlbertGroup()]

    def __call__(self, x):
        x = self.embedding_hidden_mapping_in(x)
        mask = bidirectional_attention_mask(mx.ones(x.shape[:2], mx.bool_), x.dtype)
        for _ in range(12):
            x = self.albert_layer_groups[0].albert_layers[0](x, mask)
        return x


class KokoroAlbert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = AlbertEmbeddings()
        self.encoder = AlbertEncoder()
        self.pooler = nn.Linear(768, 768)

    def __call__(self, ids):
        return self.encoder(self.embeddings(ids))


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(178, 512)
        self.cnn = [[Conv(512, 512, 5, padding=2), ChannelNorm(512)] for _ in range(3)]
        self.lstm = BiLSTM(512, 256)

    def __call__(self, ids):
        x = self.embedding(ids)
        for conv, norm in self.cnn:
            x = nn.leaky_relu(norm(conv(x)), negative_slope=0.2)
        return self.lstm(x)


class DurationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstms = [layer for _ in range(3) for layer in (BiLSTM(640, 256), AdaLayerNorm())]

    def __call__(self, x, style):
        expanded = mx.broadcast_to(style[:, None, :], (*x.shape[:2], 128))
        x = mx.concatenate((x, expanded), axis=-1)
        for i in range(0, 6, 2):
            x = self.lstms[i](x)
            x = self.lstms[i + 1](x, style)
            x = mx.concatenate((x, expanded), axis=-1)
        return x


class LinearNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(512, 50)

    def __call__(self, x):
        return self.linear_layer(x)


class ProsodyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = DurationEncoder()
        self.lstm = BiLSTM(640, 256)
        self.duration_proj = LinearNorm()
        self.shared = BiLSTM(640, 256)
        self.F0 = [AdaResBlock(512, 512), AdaResBlock(512, 256, True), AdaResBlock(256, 256)]
        self.N = [AdaResBlock(512, 512), AdaResBlock(512, 256, True), AdaResBlock(256, 256)]
        self.F0_proj, self.N_proj = Conv(256, 1, 1), Conv(256, 1, 1)

    def f0_noise(self, encoding, style):
        x = self.shared(encoding)
        f0, noise = x, x
        for block in self.F0:
            f0 = block(f0, style)
        for block in self.N:
            noise = block(noise, style)
        return self.F0_proj(f0)[..., 0], self.N_proj(noise)[..., 0]
