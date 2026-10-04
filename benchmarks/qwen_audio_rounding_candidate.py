"""Experimental Qwen rounding candidate; never a production loader/default.

Real-checkpoint gates failed. This module is opt-in validation tooling, preserving
the established runtime until the complete unchanged acceptance matrix passes.
Attention rounding follows the frozen PyTorch 2.7.1 CPU reference; all operations
execute in MLX and stored checkpoint parameter dtypes are preserved.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.qwen3_asr.model import AudioAttention


def candidate_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any = None,
    cpu_flash: bool = False,
) -> Any:
    # The frozen PyTorch 2.7.1 CPU BF16 flash kernel rounds both GEMM results
    # and unnormalized exponentials, before float32 online normalization.
    # A float32 SDPA followed by one cast is NOT its numerical contract.
    # Reference: aten/src/ATen/native/cpu/FlashAttentionKernel.cpp (v2.7.1).
    dtype = q.dtype
    # Only the audio MHA uses this kernel in our pinned reference. The text
    # decoder's mask-free GQA dispatch uses the high-precision math kernel.
    if dtype == mx.bfloat16 and cpu_flash:
        groups = q.shape[1] // k.shape[1]
        if groups != 1:
            k, v = [mx.repeat(value, groups, axis=1) for value in (k, v)]
        q32 = q.astype(mx.float32)
        maximum = mx.full((*q.shape[:-1], 1), -float("inf"))
        denominator = mx.zeros_like(maximum)
        numerator = mx.zeros(q.shape, dtype=mx.float32)
        # CPU reference processes at most 512 keys per online-softmax block.
        # Preserve this boundary instead of retaining a full quadratic matrix.
        for start in range(0, k.shape[2], 512):
            keys = k[:, :, start : start + 512].astype(mx.float32)
            values = v[:, :, start : start + 512].astype(mx.float32)
            scores = (q32 @ keys.swapaxes(-1, -2)).astype(dtype).astype(mx.float32) * scale
            if mask is not None:
                scores = scores + mask[..., start : start + 512].astype(mx.float32)
            current = mx.maximum(maximum, mx.max(scores, axis=-1, keepdims=True))
            correction = mx.exp(maximum - current)
            exponentials = mx.exp(scores - current)
            contribution = (
                (exponentials.astype(dtype).astype(mx.float32) @ values)
                .astype(dtype)
                .astype(mx.float32)
            )
            numerator = numerator * correction + contribution
            denominator = denominator * correction + mx.sum(exponentials, axis=-1, keepdims=True)
            maximum = current
        return (numerator / denominator).astype(dtype)
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        k.astype(mx.float32),
        v.astype(mx.float32),
        scale=scale,
        mask=None if mask is None else mask.astype(mx.float32),
    ).astype(dtype)


class AudioLayerNorm(nn.LayerNorm):
    """Reference LayerNorm rounds after float32 normalization and affine gain.

    Passing BF16 affine parameters directly to the fused MLX operation changes
    its rounding. Cast temporary operands, not stored checkpoint parameters.
    """

    def __call__(self, x: Any) -> Any:
        return mx.fast.layer_norm(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.bias.astype(mx.float32),
            self.eps,
        ).astype(x.dtype)


class AudioConv2d(nn.Conv2d):
    """Accumulate convolution and bias before the single reference dtype cast."""

    def __call__(self, x: Any) -> Any:
        value = mx.conv2d(
            x.astype(mx.float32),
            self.weight.astype(mx.float32),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if "bias" in self:
            value = value + self.bias.astype(mx.float32)
        return value.astype(x.dtype)


class CandidateAudioAttention(AudioAttention):
    def __call__(self, x: Any) -> Any:
        b, length, _ = x.shape
        q, k, v = [
            projection(x).reshape(b, length, self._heads, self._head_dim).transpose(0, 2, 1, 3)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        out = candidate_attention(q, k, v, scale=self._head_dim**-0.5, cpu_flash=True)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


def install_candidate(model: Any) -> None:
    """Change only this validation process's model; never shared global classes."""
    audio = model.audio_tower
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        getattr(audio, name).__class__ = AudioConv2d
    audio.ln_post.__class__ = AudioLayerNorm
    for layer in audio.layers:
        layer.self_attn_layer_norm.__class__ = AudioLayerNorm
        layer.final_layer_norm.__class__ = AudioLayerNorm
        layer.self_attn.__class__ = CandidateAudioAttention
