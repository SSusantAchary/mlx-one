"""Native CLIP using MLX-One vision attention and shared QuickGELU primitives.

Architecture: Transformers 4.57.1, Apache-2.0. The OpenAI repository is MIT;
the pinned HF weight card has no license field. Only the audited exact
safetensors conversion is accepted; see NOTICE.md for provenance limits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from mlx_one.models.classification.clip_config import CLIPConfig
from mlx_one.models.shared.transformer import quick_gelu
from mlx_one.models.vision_language.lfm2_vl.model import VisionAttention

MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
HASHES = {
    "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    "model.safetensors": "de1e7e3f5fa864a03f816d496defd31b96ffe818c890fe71ce9893b6e25430a9",
}


class CLIPAttention(VisionAttention):
    def __call__(self, x, mask=None):
        if mask is None:
            return super().__call__(x)
        batch, length, dim = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q = self.q_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(shape).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        result = mx.softmax(scores, axis=-1, precise=True) @ v
        return self.out_proj(result.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class CLIPMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x):
        return self.fc2(quick_gelu(self.fc1(x)))


class CLIPLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.hidden_act != "quick_gelu":
            raise ValueError("unsupported CLIP activation")
        self.self_attn = CLIPAttention(cfg.hidden_size, cfg.num_attention_heads)
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = CLIPMLP(cfg)

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = [CLIPLayer(cfg) for _ in range(cfg.num_hidden_layers)]

    def __call__(self, x, mask=None, trace=False):
        states = [x] if trace else None
        for layer in self.layers:
            x = layer(x, mask)
            if states is not None:
                states.append(x)
        return x, tuple(states) if states is not None else None


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.class_embedding = mx.zeros((cfg.hidden_size,))
        self.patch_embedding = nn.Conv2d(
            3, cfg.hidden_size, cfg.patch_size, stride=cfg.patch_size, bias=False
        )
        self.position_embedding = nn.Embedding(
            (cfg.image_size // cfg.patch_size) ** 2 + 1, cfg.hidden_size
        )

    def __call__(self, pixels):
        x = self.patch_embedding(pixels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        cls = mx.broadcast_to(self.class_embedding, (x.shape[0], 1, x.shape[-1]))
        x = mx.concatenate([cls, x], axis=1)
        return x + self.position_embedding(mx.arange(x.shape[1])[None, :])


class CLIPVisionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(cfg)
        self.pre_layrnorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.encoder = CLIPEncoder(cfg)
        self.post_layernorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, pixels, trace=False):
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (224, 224, 3):
            raise ValueError("CLIP expects NHWC [batch,224,224,3] pixels")
        embedding = self.embeddings(pixels)
        x, states = self.encoder(self.pre_layrnorm(embedding), trace=trace)
        return x, self.post_layernorm(x[:, 0]), states, embedding


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)

    def __call__(self, ids):
        return self.token_embedding(ids) + self.position_embedding(mx.arange(ids.shape[1])[None, :])


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.eos_token_id = cfg.eos_token_id

    def __call__(self, ids, attention_mask, trace=False):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= 77 or ids.shape != attention_mask.shape:
            raise ValueError("CLIP text IDs and mask must have [batch,sequence<=77] shape")
        embedding = self.embeddings(ids)
        n = ids.shape[1]
        minimum = mx.array(mx.finfo(embedding.dtype).min, embedding.dtype)
        allowed = mx.arange(n)[None, :] <= mx.arange(n)[:, None]
        causal = mx.where(allowed[None, None, :, :], mx.array(0, embedding.dtype), minimum)
        padding = mx.where(
            attention_mask[:, None, None, :].astype(mx.bool_), mx.array(0, embedding.dtype), minimum
        )
        x, states = self.encoder(embedding, causal + padding, trace)
        x = self.final_layer_norm(x)
        index = mx.argmax(
            ids if self.eos_token_id == 2 else (ids == self.eos_token_id).astype(mx.int32), axis=-1
        )
        return x, x[mx.arange(ids.shape[0]), index], states, embedding


class CLIPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.vision_model = CLIPVisionTransformer(cfg.vision_config)
        self.text_model = CLIPTextTransformer(cfg.text_config)
        self.visual_projection = nn.Linear(
            cfg.vision_config.hidden_size, cfg.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            cfg.text_config.hidden_size, cfg.projection_dim, bias=False
        )
        self.logit_scale = mx.array(cfg.logit_scale_init_value)

    def __call__(self, pixels, input_ids, attention_mask, *, trace=False):
        vision, pooled_image, vision_states, vision_embedding = self.vision_model(pixels, trace)
        text, pooled_text, text_states, text_embedding = self.text_model(
            input_ids, attention_mask, trace
        )
        image_features = self.visual_projection(pooled_image)
        text_features = self.text_projection(pooled_text)
        images = image_features / mx.sqrt(mx.sum(image_features**2, axis=-1, keepdims=True))
        texts = text_features / mx.sqrt(mx.sum(text_features**2, axis=-1, keepdims=True))
        logits = ((texts @ images.T) * mx.exp(self.logit_scale)).T
        result = {
            "logits_per_image": logits,
            "image_embeds": images,
            "text_embeds": texts,
            "image_features": image_features,
            "text_features": text_features,
        }
        if trace:
            result.update(
                vision_last_hidden=vision,
                text_last_hidden=text,
                vision_embedding=vision_embedding,
                text_embedding=text_embedding,
            )
            result.update({f"vision_layer_{i:02}": v for i, v in enumerate(vision_states)})
            result.update({f"text_layer_{i:02}": v for i, v in enumerate(text_states)})
        return result


@dataclass(frozen=True)
class LoadedCLIP:
    model: CLIPModel
    tokenizer: object
    model_path: Path
    model_id: str = MODEL_ID
    revision: str = REVISION
    architecture: str = "clip"
    context_length: int = 77

    @property
    def quantization(self):
        return {}


def load_clip(path, *, revision=REVISION):
    if revision != REVISION:
        raise ValueError("unsupported CLIP checkpoint revision")
    root = Path(path)
    for name, expected in HASHES.items():
        digest = hashlib.sha256()
        with (root / name).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"CLIP asset checksum mismatch: {name}")
    payload = json.loads((root / "config.json").read_text())
    cfg = CLIPConfig.from_dict(payload)
    model = CLIPModel(cfg)
    tensors = mx.load(str(root / "model.safetensors"))
    for name, length in [
        ("text_model.embeddings.position_ids", 77),
        ("vision_model.embeddings.position_ids", 50),
    ]:
        if (
            name not in tensors
            or not mx.array_equal(tensors[name], mx.arange(length)[None, :]).item()
        ):
            raise ValueError(f"invalid source-only CLIP position buffer: {name}")
        tensors.pop(name)
    key = "vision_model.embeddings.patch_embedding.weight"
    tensors[key] = tensors[key].transpose(0, 2, 3, 1)
    expected = dict(tree_flatten(model.parameters()))
    if set(expected) != set(tensors):
        raise ValueError("CLIP tensor inventory mismatch")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != mx.float32:
            raise ValueError(f"CLIP tensor shape/dtype mismatch: {name}")
    model.load_weights(list(tensors.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    return LoadedCLIP(model, tokenizer, root.resolve())
