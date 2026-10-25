"""Small native adapters for ported OCR layers and MLX-One's existing state types.

No scheduler, sampler, tokenizer wrapper, or third-party inference backend lives here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import EncoderDecoderKVCache
from mlx_one.core.cache import KVCache as NativeKVCache
from mlx_one.models.shared.reference_math import attention_mask


class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


@dataclass
class LanguageModelOutput:
    logits: object
    encoder_outputs: object = None
    cross_attention_states: object = None


@dataclass
class InputEmbeddingsFeatures:
    inputs_embeds: object = None
    attention_mask: object = None
    decoder_inputs_embeds: object = None


class KVCache(NativeKVCache):
    def update_and_fetch(self, keys, values):
        return self.update(keys, values)

    def clone(self):
        result = KVCache()
        result.keys, result.values, result._offset = self.keys, self.values, self.offset
        return result


SimpleKVCache = KVCache


class _CrossView:
    def __init__(self, parent):
        self.parent = parent

    @property
    def keys(self):
        return self.parent.cross_keys

    @property
    def values(self):
        return self.parent.cross_values

    def update_and_fetch(self, keys, values):
        return self.parent.update_cross(keys, values)


class CacheList(EncoderDecoderKVCache):
    """BART's two named attention states backed by the shared encoder/decoder cache."""

    def __init__(self, *_):
        super().__init__()
        self.self_attention = KVCache()

    def __getitem__(self, index):
        return self.self_attention if index == 0 else _CrossView(self)

    def clone(self):
        result = CacheList()
        result.self_attention = self.self_attention.clone()
        result.cross_keys, result.cross_values = self.cross_keys, self.cross_values
        return result


def cache_position_offset(cache):
    return cache.offset if cache is not None else 0


def kv_sequence_length(keys):
    return keys.shape[-2]


def create_attention_mask(hidden, cache=None, **_):
    if isinstance(cache, (list, tuple)):
        cache = cache[0] if cache else None
    return attention_mask(hidden.shape[1], cache_position_offset(cache), None)


def scaled_dot_product_attention(q, k, v, cache=None, *, scale, mask=None):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def check_array_shape(value):
    return value.ndim == 4 and value.shape[1] == value.shape[2]


def rotate_half(value):
    left, right = mx.split(value, 2, axis=-1)
    return mx.concatenate([-right, left], axis=-1)


class MRoPERotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=500000, **_):
        super().__init__()
        self.dim, self.base = dim, base

    def __call__(self, hidden, position_ids):
        inv = self.base ** (-mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        freqs = position_ids[..., None].astype(mx.float32) * inv
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(hidden.dtype), mx.sin(emb).astype(hidden.dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, *, mrope_section, **_):
    if sum(mrope_section) * 2 != q.shape[-1]:
        raise ValueError("MRoPE sections must cover the exact head dimension")
    sections = list(mrope_section) * 2
    splits = []
    offset = 0
    for width in sections[:-1]:
        offset += width
        splits.append(offset)
    c = mx.concatenate([part[i % 3] for i, part in enumerate(mx.split(cos, splits, -1))], -1)
    s = mx.concatenate([part[i % 3] for i, part in enumerate(mx.split(sin, splits, -1))], -1)
    return q * c[:, None] + rotate_half(q) * s[:, None], (
        k * c[:, None] + rotate_half(k) * s[:, None]
    )


def bilinear_interpolate(image, new_height, new_width, align_corners=False):
    """Interpolate H,W,... using half-pixel coordinates and edge replication."""

    def axis(length, target):
        positions = (
            mx.linspace(0, length - 1, target)
            if align_corners and target > 1
            else (mx.arange(target) + 0.5) * length / target - 0.5
        )
        floor = mx.floor(positions)
        return (
            mx.clip(floor, 0, length - 1).astype(mx.int32),
            (mx.clip(floor + 1, 0, length - 1).astype(mx.int32)),
            positions - floor,
        )

    y0, y1, wy = axis(image.shape[0], new_height)
    x0, x1, wx = axis(image.shape[1], new_width)
    wy = wy.reshape((new_height, 1) + (1,) * (image.ndim - 2))
    wx = wx.reshape((1, new_width) + (1,) * (image.ndim - 2))
    return (image[y0[:, None], x0[None]] * (1 - wx) + image[y0[:, None], x1[None]] * wx) * (
        1 - wy
    ) + (image[y1[:, None], x0[None]] * (1 - wx) + image[y1[:, None], x1[None]] * wx) * wy


def bicubic_interpolate(image, size, antialias=False):
    """NCHW bicubic positions; AA uses the normalized, stretched a=-0.5 filter.

    Follows ATen's half-pixel sampling and per-output border normalization.
    Reference: mlx-vlm interpolate.py at the pinned OCR provenance revision.
    No weight/image values leave MLX; only interpolation indices are constructed.
    """
    import math

    def weights(source, target):
        scale = source / target
        support = 2 * max(scale, 1)
        taps = math.ceil(support) * 2 + 1
        center = scale * (mx.arange(target, dtype=mx.float32) + 0.5)
        start = mx.maximum(mx.floor(center - support + 0.5), 0)
        end = mx.minimum(mx.floor(center + support + 0.5), source)
        positions = start[:, None] + mx.arange(taps)[None]
        distance = mx.abs((positions - center[:, None] + 0.5) / max(scale, 1))
        a = -0.5
        first = ((a + 2) * distance - (a + 3)) * distance * distance + 1
        second = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        value = mx.where(distance <= 1, first, mx.where(distance < 2, second, 0))
        value = mx.where(positions < end[:, None], value, 0)
        value = value / mx.maximum(value.sum(-1, keepdims=True), 1e-30)
        return mx.minimum(positions, source - 1).astype(mx.int32), value

    height, width = image.shape[-2:]
    if antialias:
        iy, wy = weights(height, size[0])
        ix, wx = weights(width, size[1])
        value = mx.take(image, iy, axis=2)
        value = (value * wy[None, None, :, :, None]).sum(3)
        value = mx.take(value, ix, axis=3)
        return (value * wx[None, None, None, :, :]).sum(4)
    value = image.transpose(0, 2, 3, 1)
    value = nn.Upsample(
        scale_factor=(size[0] / height, size[1] / width), mode="cubic", align_corners=False
    )(value)
    return value.transpose(0, 3, 1, 2)


def interpolate(image, size, **_):
    return bicubic_interpolate(image, size, antialias=True)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMLP(SwiGLUMLP):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__(
            hidden_size or config.hidden_size, intermediate_size or config.intermediate_size
        )


class Qwen2LanguageModel(nn.Module):
    """GOT's exact Qwen2 decoder, sharing MLX-One's existing implementation."""

    def __init__(self, config):
        super().__init__()
        from mlx_one.models.language.qwen2.config import Qwen2Config
        from mlx_one.models.language.qwen2.model import Qwen2Model

        self.config = Qwen2Config.from_dict({**vars(config), "model_type": "qwen2"})
        self.model = Qwen2Model(self.config)
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids=None, inputs_embeds=None, cache=None, **_):
        hidden, _, _ = self.model(
            input_ids if inputs_embeds is None else None,
            input_embeddings=inputs_embeds,
            cache=cache,
        )
        project = (
            self.model.embed_tokens.as_linear if self.config.tie_word_embeddings else self.lm_head
        )
        return LanguageModelOutput(project(hidden))
