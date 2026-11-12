"""Native pre-norm ViT patch encoder and supervised image classifier."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.embeddings.bert.model import BertIntermediate, BertSelfAttention


class ViTLayerNorm(nn.LayerNorm):
    """Centered two-pass FP32 statistics for the checkpoint's very small epsilon."""

    def __call__(self, hidden):
        value = hidden.astype(mx.float32)
        centered = value - mx.mean(value, axis=-1, keepdims=True)
        normalized = centered * mx.rsqrt(
            mx.mean(centered * centered, axis=-1, keepdims=True) + self.eps
        )
        return (normalized * self.weight + self.bias).astype(hidden.dtype)


class PatchProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, pixels):
        patches = self.projection(pixels.transpose(0, 2, 3, 1))
        return patches.reshape(patches.shape[0], -1, patches.shape[-1])


class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = PatchProjection(config)
        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.position_embeddings = mx.zeros(
            (1, (config.image_size // config.patch_size) ** 2 + 1, config.hidden_size)
        )

    def __call__(self, pixels):
        patches = self.patch_embeddings(pixels)
        cls = mx.broadcast_to(self.cls_token, (patches.shape[0], 1, patches.shape[-1]))
        return mx.concatenate((cls, patches), axis=1) + self.position_embeddings


class DenseOutput(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)

    def __call__(self, hidden):
        return self.dense(hidden)


class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.output = DenseOutput(config.hidden_size, config.hidden_size)

    def __call__(self, hidden):
        return self.output(self.attention(hidden, mx.array(0, dtype=hidden.dtype)))


class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_before = ViTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = ViTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = ViTAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = DenseOutput(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden):
        hidden = hidden + self.attention(self.layernorm_before(hidden))
        return hidden + self.output(self.intermediate(self.layernorm_after(hidden)))


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = [ViTLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden):
        for layer in self.layer:
            hidden = layer(hidden)
        return hidden


class ViTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = ViTLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, pixels):
        return self.layernorm(self.encoder(self.embeddings(pixels)))


class ViTForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vit = ViTBackbone(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, pixel_values):
        expected = (self.config.num_channels, self.config.image_size, self.config.image_size)
        if (
            pixel_values.ndim != 4
            or pixel_values.shape[1:] != expected
            or not pixel_values.shape[0]
        ):
            raise ValueError(
                "ViT pixels require a nonempty NCHW batch at the configured RGB resolution"
            )
        hidden = self.vit(pixel_values)
        return ModelOutput(self.classifier(hidden[:, 0]), hidden)
