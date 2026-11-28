"""Common diffusion attention, positional embeddings and feed-forward math."""

import math

import mlx.core as mx
import mlx.nn as nn


def timestep_embedding(t: mx.array, width: int) -> mx.array:
    half = width // 2
    phase = t.astype(mx.float32)[:, None] * mx.exp(
        (-math.log(10000.0) / half) * mx.arange(half, dtype=mx.float32)
    )
    return mx.concatenate((mx.cos(phase), mx.sin(phase)), axis=-1)


def rotary_angles(ids: mx.array, axes: tuple[int, ...], theta: float):
    phases = [
        ids[:, i, None] * theta ** (-mx.arange(0, width, 2, dtype=mx.float32) / width)
        for i, width in enumerate(axes)
    ]
    phase = mx.concatenate(phases, axis=-1)
    return mx.cos(phase), mx.sin(phase)


def rotate(x: mx.array, angles) -> mx.array:
    cos, sin = angles
    pair = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    a, b = pair[..., 0], pair[..., 1]
    return (
        mx.stack((a * cos - b * sin, a * sin + b * cos), axis=-1).reshape(x.shape).astype(x.dtype)
    )


def split_heads(x: mx.array, heads: int) -> mx.array:
    return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(0, 2, 1, 3)


def attention(q: mx.array, k: mx.array, v: mx.array, angles=None) -> mx.array:
    if angles is not None:
        q, k = rotate(q, angles), rotate(k, angles)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    return out.transpose(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)


def swiglu(x: mx.array) -> mx.array:
    gate, value = mx.split(x, 2, axis=-1)
    return nn.silu(gate) * value
