"""Fixed-resolution SigLIP architecture used by the pinned SigLIP2 checkpoint.

The checkpoint resolves to SiglipModel, not the NaFlex Siglip2Model. Encoder
blocks and attention are reused from the existing native LFM2 vision tower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.classification.clip import CLIPTextEmbeddings
from mlx_one.models.vision_language.lfm2_vl.model import VisionBlock


@dataclass(frozen=True)
class FixedSiglipConfig:
    model_type: str = "siglip"
    vocab_size: int = 256000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    image_size: int = 224
    patch_size: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, payload):
        if payload.get("model_type") != "siglip":
            raise ValueError("fixed checkpoint must use SiglipModel")
        if payload.get("text_config", {}).get("vocab_size") != 256000:
            raise ValueError("unsupported fixed SigLIP vocabulary")
        return cls()


def gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class SiglipEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = [VisionBlock(cfg) for _ in range(cfg.num_hidden_layers)]

    def __call__(self, x, trace=False):
        states = [x] if trace else None
        for layer in self.layers:
            x = layer(x)
            if states is not None:
                states.append(x)
        return x, tuple(states) if states is not None else None


class SiglipMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x):
        return self.fc2(gelu_tanh(self.fc1(x)))


class PoolAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg.hidden_size
        self.in_proj_weight = mx.zeros((3 * dim, dim))
        self.in_proj_bias = mx.zeros((3 * dim,))
        self.out_proj = nn.Linear(dim, dim)
        self.heads = cfg.num_attention_heads

    def __call__(self, query, values):
        dim = query.shape[-1]
        heads = self.heads

        def project(x, index):
            p = x @ self.in_proj_weight[index * dim : (index + 1) * dim].T
            p = p + self.in_proj_bias[index * dim : (index + 1) * dim]
            return p.reshape(p.shape[0], p.shape[1], heads, dim // heads).transpose(0, 2, 1, 3)

        q = project(query, 0)
        k = project(values, 1)
        v = project(values, 2)
        probabilities = mx.softmax(
            (q @ k.transpose(0, 1, 3, 2)) * ((dim // heads) ** -0.5), axis=-1, precise=True
        )
        x = (probabilities @ v).transpose(0, 2, 1, 3).reshape(query.shape)
        return self.out_proj(x)


class PoolHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.probe = mx.zeros((1, 1, cfg.hidden_size))
        self.attention = PoolAttention(cfg)
        self.layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = SiglipMLP(cfg)

    def __call__(self, x):
        probe = mx.broadcast_to(self.probe, (x.shape[0], 1, x.shape[-1]))
        x = self.attention(probe, x)
        return (x + self.mlp(self.layernorm(x)))[:, 0]


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, cfg.hidden_size, cfg.patch_size, stride=cfg.patch_size)
        self.position_embedding = nn.Embedding(
            (cfg.image_size // cfg.patch_size) ** 2, cfg.hidden_size
        )

    def __call__(self, pixels):
        x = self.patch_embedding(pixels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x + self.position_embedding(mx.arange(x.shape[1])[None, :])


class SiglipVisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(cfg)
        self.encoder = SiglipEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.head = PoolHead(cfg)

    def __call__(self, pixels, trace=False):
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (224, 224, 3):
            raise ValueError("fixed SigLIP expects NHWC [batch,224,224,3]")
        x, states = self.encoder(self.embeddings(pixels), trace)
        x = self.post_layernorm(x)
        return x, self.head(x), states


class SiglipTextTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(cfg)
        self.encoder = SiglipEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.head = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, ids, trace=False):
        if ids.ndim != 2 or ids.shape[1] != 64:
            raise ValueError("fixed SigLIP text must be padded/truncated to 64 tokens")
        # This checkpoint's tokenizer model_input_names is input_ids only; its
        # padding participates in bidirectional attention and final-position pooling.
        x, states = self.encoder(self.embeddings(ids), trace)
        x = self.final_layer_norm(x)
        return x, self.head(x[:, -1]), states


class FixedSiglipModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.vision_model = SiglipVisionTower(cfg)
        self.text_model = SiglipTextTower(cfg)
        self.logit_scale = mx.zeros((1,))
        self.logit_bias = mx.zeros((1,))

    def __call__(self, pixels, ids, *, trace=False):
        vision, image_features, vision_states = self.vision_model(pixels, trace)
        text, text_features, text_states = self.text_model(ids, trace)
        images = image_features / mx.sqrt(mx.sum(image_features**2, axis=-1, keepdims=True))
        texts = text_features / mx.sqrt(mx.sum(text_features**2, axis=-1, keepdims=True))
        logits = ((texts @ images.T) * mx.exp(self.logit_scale) + self.logit_bias).T
        result = {
            "logits_per_image": logits,
            "image_embeds": images,
            "text_embeds": texts,
            "image_features": image_features,
            "text_features": text_features,
        }
        if trace:
            result.update(vision_last_hidden=vision, text_last_hidden=text)
            result.update({f"vision_layer_{i:02}": x for i, x in enumerate(vision_states)})
            result.update({f"text_layer_{i:02}": x for i, x in enumerate(text_states)})
        return result


MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
HASHES = {
    "config.json": "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
    "preprocessor_config.json": "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    "tokenizer_config.json": "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
    "model.safetensors": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
}


@dataclass(frozen=True)
class LoadedFixedSiglip:
    model: FixedSiglipModel
    tokenizer: object
    model_path: object
    model_id: str = MODEL_ID
    revision: str = REVISION
    architecture: str = "siglip"


def load_fixed_siglip(path, *, revision=REVISION):
    import hashlib
    import json
    import re
    from pathlib import Path

    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer

    if revision != REVISION:
        raise ValueError("unsupported fixed SigLIP checkpoint revision")
    root = Path(path)
    for name, expected in HASHES.items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"fixed SigLIP asset checksum mismatch: {name}")
    cfg = FixedSiglipConfig.from_dict(json.loads((root / "config.json").read_text()))
    model = FixedSiglipModel(cfg)
    tensors = mx.load(str(root / "model.safetensors"))
    # Existing native VisionBlock keeps FFN projections directly on the block.
    tensors = {re.sub(r"(\.encoder\.layers\.\d+)\.mlp\.", r"\1.", k): v for k, v in tensors.items()}
    conv = "vision_model.embeddings.patch_embedding.weight"
    tensors[conv] = tensors[conv].transpose(0, 2, 3, 1)
    expected = dict(tree_flatten(model.parameters()))
    if set(tensors) != set(expected):
        raise ValueError(
            f"fixed SigLIP tensor inventory mismatch: missing={set(expected) - set(tensors)}, "
            f"unexpected={set(tensors) - set(expected)}"
        )
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"fixed SigLIP tensor shape/dtype mismatch: {name}")
    model.load_weights(list(tensors.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_padding(length=64, pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=64)
    return LoadedFixedSiglip(model, tokenizer, root.resolve())
