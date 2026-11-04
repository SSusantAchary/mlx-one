"""Native Depth Anything V2 Small / DINOv2 checkpoint candidate.

Source Transformers v4.57.1 8cb5963cc22174954e7dca2c0a3320b7dc2f4edc,
Apache-2.0. Reuses existing native vision attention and GELU MLP primitives.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_one.models.depth.interpolation import resize
from mlx_one.models.shared.vision import VisionMlp
from mlx_one.models.vision_language.lfm2_vl.model import VisionAttention

REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 384, 14, stride=14)


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = mx.zeros((1, 1, 384))
        self.mask_token = mx.zeros((1, 384))
        self.position_embeddings = mx.zeros((1, 1370, 384))
        self.patch_embeddings = PatchEmbeddings()

    def __call__(self, pixels):
        patches = self.patch_embeddings.projection(pixels)
        batch, h, w, dim = patches.shape
        x = mx.concatenate(
            (mx.broadcast_to(self.cls_token, (batch, 1, dim)), patches.reshape(batch, h * w, dim)),
            axis=1,
        )
        position = self.position_embeddings
        if h * w != 1369 or pixels.shape[1] != pixels.shape[2]:
            patch_pos = resize(position[:, 1:].reshape(1, 37, 37, dim), (h, w), mode="bicubic")
            position = mx.concatenate((position[:, :1], patch_pos.reshape(1, h * w, dim)), axis=1)
        return x + position


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = mx.ones(384)

    def __call__(self, x):
        return x * self.lambda1


class DinoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attention = VisionAttention(384, 6)
        self.layer_scale1 = LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = VisionMlp(384, 1536, "gelu")
        self.layer_scale2 = LayerScale()

    def __call__(self, x):
        x = self.layer_scale1(self.attention(self.norm1(x))) + x
        return self.layer_scale2(self.mlp(self.norm2(x))) + x


class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = [DinoLayer() for _ in range(12)]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder = DinoEncoder()
        self.layernorm = nn.LayerNorm(384, eps=1e-6)

    def __call__(self, pixels):
        x = self.embeddings(pixels)
        trace = {"embeddings": x}
        features = []
        for i, layer in enumerate(self.encoder.layer, 1):
            x = layer(x)
            trace[f"block_{i:02}"] = x
            if i in (3, 6, 9, 12):
                f = self.layernorm(x)
                features.append(f)
                trace[f"tap_{i:02}"] = f
        return features, trace


class ReassembleLayer(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.projection = nn.Conv2d(384, channels, 1)
        self.factor = factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, int(factor), stride=int(factor))
        elif factor < 1:
            self.resize = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x):
        x = self.projection(x)
        return x if self.factor == 1 else self.resize(x)


class Reassemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            ReassembleLayer(c, f) for c, f in zip((48, 96, 192, 384), (4, 2, 1, 0.5), strict=True)
        ]


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution1 = nn.Conv2d(64, 64, 3, padding=1)
        self.convolution2 = nn.Conv2d(64, 64, 3, padding=1)

    def __call__(self, x):
        return self.convolution2(nn.relu(self.convolution1(nn.relu(x)))) + x


class FusionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(64, 64, 1)
        self.residual_layer1 = Residual()
        self.residual_layer2 = Residual()

    def __call__(self, x, residual=None, size=None):
        if residual is not None:
            if residual.shape != x.shape:
                residual = resize(residual, x.shape[1:3], align_corners=False)
            x = x + self.residual_layer1(residual)
        x = self.residual_layer2(x)
        size = (x.shape[1] * 2, x.shape[2] * 2) if size is None else size
        return self.projection(resize(x, size, align_corners=True))


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [FusionLayer() for _ in range(4)]


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reassemble_stage = Reassemble()
        self.convs = [nn.Conv2d(c, 64, 3, padding=1, bias=False) for c in (48, 96, 192, 384)]
        self.fusion_stage = Fusion()

    def __call__(self, features, h, w):
        trace = {}
        reassembled = []
        for i, (feature, layer, conv) in enumerate(
            zip(features, self.reassemble_stage.layers, self.convs, strict=True)
        ):
            x = layer(feature[:, 1:].reshape(feature.shape[0], h, w, 384))
            trace[f"reassemble_{i}"] = x.transpose(0, 3, 1, 2)
            x = conv(x)
            reassembled.append(x)
            trace[f"neck_conv_{i}"] = x.transpose(0, 3, 1, 2)
        reassembled = reassembled[::-1]
        x = None
        for i, (feature, layer) in enumerate(
            zip(reassembled, self.fusion_stage.layers, strict=True)
        ):
            target = reassembled[i + 1].shape[1:3] if i < 3 else None
            x = layer(feature, size=target) if x is None else layer(x, feature, target)
            trace[f"fusion_{i}"] = x.transpose(0, 3, 1, 2)
        return x, trace


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)

    def __call__(self, x, size):
        x = self.conv1(x)
        x = resize(x, size, align_corners=True)
        return nn.relu(self.conv3(nn.relu(self.conv2(x))))[:, :, :, 0]


class DepthAnything(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()

    def __call__(self, pixels):
        if (
            pixels.ndim != 4
            or pixels.shape[0] != 1
            or pixels.shape[-1] != 3
            or pixels.dtype != mx.float32
            or pixels.shape[1] % 14
            or pixels.shape[2] % 14
            or pixels.shape[1] * pixels.shape[2] > 1_000_000
        ):
            raise ValueError("depth candidate requires bounded B1 FP32 NHWC patch-aligned pixels")
        h, w = pixels.shape[1] // 14, pixels.shape[2] // 14
        features, trace = self.backbone(pixels)
        x, neck = self.neck(features, h, w)
        return {**trace, **neck, "predicted_depth": self.head(x, (h * 14, w * 14))}


def load_depth(path):
    root = Path(path)
    # Locks are generated before validation; constants below are the only accepted assets.
    for name, expected in HASHES.items():
        if hashlib.file_digest((root / name).open("rb"), "sha256").hexdigest() != expected:
            raise ValueError(f"Depth Anything asset mismatch: {name}")
    cfg = json.loads((root / "config.json").read_text())
    if cfg["model_type"] != "depth_anything" or cfg["backbone_config"]["hidden_size"] != 384:
        raise ValueError("unsupported depth architecture")
    source = mx.load(str(root / "model.safetensors"))
    weights = {}
    for name, value in source.items():
        for a, b in [
            (".attention.attention.query.", ".attention.q_proj."),
            (".attention.attention.key.", ".attention.k_proj."),
            (".attention.attention.value.", ".attention.v_proj."),
            (".attention.output.dense.", ".attention.out_proj."),
        ]:
            name = name.replace(a, b)
        if value.ndim == 4:
            value = (
                value.transpose(1, 2, 3, 0)
                if name
                in {
                    "neck.reassemble_stage.layers.0.resize.weight",
                    "neck.reassemble_stage.layers.1.resize.weight",
                }
                else value.transpose(0, 2, 3, 1)
            )
        weights[name] = value
    model = DepthAnything()
    expected = dict(tree_flatten(model.parameters()))
    if set(expected) != set(weights):
        raise ValueError(f"Depth tensor inventory mismatch: {set(expected) ^ set(weights)}")
    for name, value in weights.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"Depth shape/dtype mismatch: {name}")
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


HASHES = {
    "model.safetensors": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
    "config.json": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
    "preprocessor_config.json": "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
}
