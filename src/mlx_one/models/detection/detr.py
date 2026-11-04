"""DETR ResNet50 native inference candidate, pinned Transformers 4.57.1.

Apache-2.0; encoder/decoder/position math follows modeling_detr.py at
8cb5963cc22174954e7dca2c0a3320b7dc2f4edc. The checkpoint uses the timm
ResNet50 bottleneck/stride layout. No SAM3 or shared OCR math is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_one.models.classification.clip import CLIPAttention

REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
HASHES = {
    "model.safetensors": "830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81",
    "config.json": "e7bcf3992363f27717a863f14b193140ad2e41d4338ee012730e58a92cae17e6",
    "preprocessor_config.json": "0673fea2a6d3cf92cdbab3c7426c0ecdf8a4729a2a4d5199033dcd66a2b8759b",
}


class FrozenBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight, self.bias = mx.ones(channels), mx.zeros(channels)
        self.running_mean, self.running_var = mx.zeros(channels), mx.ones(channels)

    def __call__(self, x):
        scale = self.weight * mx.rsqrt(self.running_var + 1e-5)
        return x * scale + (self.bias - self.running_mean * scale)


class Bottleneck(nn.Module):
    def __init__(self, input_channels, channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channels, 1, bias=False)
        self.bn1 = FrozenBatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = FrozenBatchNorm(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, 1, bias=False)
        self.bn3 = FrozenBatchNorm(channels * 4)
        self.downsample = []
        if stride != 1 or input_channels != channels * 4:
            self.downsample = [
                nn.Conv2d(input_channels, channels * 4, 1, stride=stride, bias=False),
                FrozenBatchNorm(channels * 4),
            ]

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        for module in self.downsample:
            residual = module(residual)
        return nn.relu(x + residual)


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm(64)
        input_channels = 64
        for index, (channels, blocks) in enumerate(
            zip((64, 128, 256, 512), (3, 4, 6, 3), strict=True), 1
        ):
            layers = []
            for j in range(blocks):
                layers.append(
                    Bottleneck(input_channels, channels, 2 if index > 1 and j == 0 else 1)
                )
                input_channels = channels * 4
            setattr(self, f"layer{index}", layers)

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        # Explicit -inf matches torch MaxPool2d padding, including odd dimensions.
        x = mx.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), constant_values=-float("inf"))
        x = nn.MaxPool2d(3, stride=2)(x)
        features = []
        for i in range(1, 5):
            for layer in getattr(self, f"layer{i}"):
                x = layer(x)
            features.append(x)
        return features


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet50()


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_encoder = ConvEncoder()


def position_embedding(mask):
    y, x = mx.cumsum(mask, axis=1).astype(mx.float32), mx.cumsum(mask, axis=2).astype(mx.float32)
    y = y / (y[:, -1:, :] + 1e-6) * (2 * math.pi)
    x = x / (x[:, :, -1:] + 1e-6) * (2 * math.pi)
    dims = mx.power(10000, 2 * mx.floor(mx.arange(128).astype(mx.float32) / 2) / 128)

    def encode(z):
        z = z[..., None] / dims
        return mx.stack((mx.sin(z[..., 0::2]), mx.cos(z[..., 1::2])), axis=-1).reshape(
            *z.shape[:-1], 128
        )

    return mx.concatenate((encode(y), encode(x)), axis=-1)


class DetrAttention(CLIPAttention):
    def __init__(self):
        super().__init__(256, 8)

    def __call__(self, x, query_position, *, keys=None, key_position=None, mask=None):
        original = x
        x = x + query_position
        cross = keys is not None
        values = keys if cross else original
        keys = keys + key_position if cross else x
        batch, n, dim = x.shape
        q = (self.q_proj(x) * self.scale).reshape(batch, n, 8, 32).transpose(0, 2, 1, 3)
        k = self.k_proj(keys).reshape(batch, -1, 8, 32).transpose(0, 2, 1, 3)
        v = self.v_proj(values).reshape(batch, -1, 8, 32).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mx.where(
                mask[:, None, None, :],
                mx.array(0, x.dtype),
                mx.array(mx.finfo(x.dtype).min, x.dtype),
            )
        output = mx.softmax(scores, axis=-1, precise=True) @ v
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, n, dim))


class EncoderLayer(nn.Module):
    def __init__(self, decoder=False):
        super().__init__()
        self.self_attn = DetrAttention()
        self.self_attn_layer_norm = nn.LayerNorm(256, eps=1e-5)
        self.fc1, self.fc2 = nn.Linear(256, 2048), nn.Linear(2048, 256)
        self.final_layer_norm = nn.LayerNorm(256, eps=1e-5)
        if decoder:
            self.encoder_attn = DetrAttention()
            self.encoder_attn_layer_norm = nn.LayerNorm(256, eps=1e-5)

    def __call__(self, x, position, mask, memory=None, memory_position=None):
        x = self.self_attn_layer_norm(
            x + self.self_attn(x, position, mask=mask if memory is None else None)
        )
        if memory is not None:
            x = self.encoder_attn_layer_norm(
                x
                + self.encoder_attn(
                    x, position, keys=memory, key_position=memory_position, mask=mask
                )
            )
        return self.final_layer_norm(x + self.fc2(nn.relu(self.fc1(x))))


class Encoder(nn.Module):
    def __init__(self, decoder=False):
        super().__init__()
        self.layers = [EncoderLayer(decoder) for _ in range(6)]
        if decoder:
            self.layernorm = nn.LayerNorm(256, eps=1e-5)


class DetrCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.input_projection = nn.Conv2d(2048, 256, 1)
        self.query_position_embeddings = nn.Embedding(100, 256)
        self.encoder, self.decoder = Encoder(), Encoder(True)

    def __call__(self, pixels, pixel_mask):
        features = self.backbone.conv_encoder.model(pixels)
        feature = features[-1]
        batch, h, w, _ = feature.shape
        # Source nearest interpolation samples floor(output_index * input/output).
        iy = mx.floor(mx.arange(h) * (pixel_mask.shape[1] / h)).astype(mx.int32)
        ix = mx.floor(mx.arange(w) * (pixel_mask.shape[2] / w)).astype(mx.int32)
        mask = pixel_mask[:, iy[:, None], ix[None, :]].astype(mx.bool_)
        position = position_embedding(mask).reshape(batch, h * w, 256)
        x = self.input_projection(feature).reshape(batch, h * w, 256)
        stages = {f"backbone_{i + 1}": f.transpose(0, 3, 1, 2) for i, f in enumerate(features)}
        stages.update(projected=x, positions=position, feature_mask=mask)
        mask = mask.reshape(batch, h * w)
        for i, layer in enumerate(self.encoder.layers, 1):
            x = layer(x, position, mask)
            stages[f"encoder_{i:02}"] = x
        queries = mx.broadcast_to(self.query_position_embeddings.weight[None], (batch, 100, 256))
        decoded = mx.zeros_like(queries)
        for i, layer in enumerate(self.decoder.layers, 1):
            decoded = layer(decoded, queries, mask, x, position)
            stages[f"decoder_{i:02}"] = decoded
        decoded = self.decoder.layernorm(decoded)
        stages["decoder_norm"] = decoded
        return decoded, stages


class BoxHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 4)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


class DetrForObjectDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DetrCore()
        self.class_labels_classifier = nn.Linear(256, 92)
        self.bbox_predictor = BoxHead()

    def __call__(self, pixels, pixel_mask):
        decoded, stages = self.model(pixels, pixel_mask)
        return {
            **stages,
            "logits": self.class_labels_classifier(decoded),
            "pred_boxes": mx.sigmoid(self.bbox_predictor(decoded)),
        }


def load_detr(path):
    root = Path(path)
    for name, digest in HASHES.items():
        if hashlib.file_digest((root / name).open("rb"), "sha256").hexdigest() != digest:
            raise ValueError(f"DETR asset checksum mismatch: {name}")
    cfg = json.loads((root / "config.json").read_text())
    if cfg["backbone"] != "resnet50" or cfg["d_model"] != 256:
        raise ValueError("unsupported DETR checkpoint architecture")
    tensors = mx.load(str(root / "model.safetensors"))
    mapped = {}
    for name, value in tensors.items():
        if name.endswith("num_batches_tracked"):
            if value.ndim != 0 or name not in {
                f"model.backbone.conv_encoder.model.layer{i}.0.downsample.1.num_batches_tracked"
                for i in range(1, 5)
            }:
                raise ValueError("unexpected source-only DETR buffer")
            continue
        mapped[name] = value.transpose(0, 2, 3, 1) if value.ndim == 4 else value
    model = DetrForObjectDetection()
    expected = dict(tree_flatten(model.parameters()))
    if set(mapped) != set(expected):
        raise ValueError(f"DETR inventory mismatch: {set(mapped) ^ set(expected)}")
    for name, value in mapped.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"DETR shape/dtype mismatch: {name}")
    model.load_weights(list(mapped.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
