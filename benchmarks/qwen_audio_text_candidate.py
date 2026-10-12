"""Opt-in text-SDPA math-order diagnostic; not a production default."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.audio.qwen3_asr.model import TextAttention, rotary
from mlx_one.models.shared.transformer import causal_mask


def math_attention(q, k, v, *, scale, mask=None):
    """Mirror the pinned CPU GQA math kernel's split scaling in float32."""
    groups = q.shape[1] // k.shape[1]
    q32 = q.astype(mx.float32) * scale**0.5
    k32 = mx.repeat(k, groups, axis=1).astype(mx.float32) * scale**0.5
    v32 = mx.repeat(v, groups, axis=1).astype(mx.float32)
    scores = q32 @ k32.swapaxes(-1, -2)
    if mask is not None:
        scores = scores + mask.astype(mx.float32)
    probabilities = mx.softmax(scores, axis=-1)
    probabilities = mx.where(mx.isfinite(probabilities), probabilities, 0.0)
    return (probabilities @ v32).astype(q.dtype)


class CandidateTextAttention(TextAttention):
    def __call__(self, hidden, cache):
        c = self._config
        b, length, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).reshape(b, length, c.num_attention_heads, c.head_dim)
        ).transpose(0, 2, 1, 3)
        k = self.k_norm(
            self.k_proj(hidden).reshape(b, length, c.num_key_value_heads, c.head_dim)
        ).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden).reshape(b, length, c.num_key_value_heads, c.head_dim)
        v = v.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        q, k = rotary(q, offset, c.rope_theta), rotary(k, offset, c.rope_theta)
        if cache is not None:
            k, v = cache.update(k, v)
        out = math_attention(q, k, v, scale=c.head_dim**-0.5, mask=causal_mask(length, offset))
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


def install_candidate(model):
    for layer in model.model.layers:
        layer.self_attn.__class__ = CandidateTextAttention


class CompensatedLinear(nn.Linear):
    """Diagnostic BF16 GEMM with compensated float32 tile accumulation."""

    def __call__(self, x):
        if x.dtype != mx.bfloat16:
            return super().__call__(x)
        total = mx.zeros((*x.shape[:-1], self.weight.shape[0]), dtype=mx.float32)
        error = mx.zeros_like(total)
        for start in range(0, x.shape[-1], 256):
            value = x[..., start : start + 256].astype(mx.float32)
            weight = self.weight[:, start : start + 256].astype(mx.float32)
            partial = value @ weight.T
            corrected = partial - error
            updated = total + corrected
            error = (updated - total) - corrected
            total = updated
        if "bias" in self:
            total = total + self.bias.astype(mx.float32)
        return total.astype(x.dtype)


def install_compensated(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.__class__ = CompensatedLinear


def install_layernorm(model):
    from qwen_audio_rounding_candidate import AudioLayerNorm

    model.audio_tower.ln_post.__class__ = AudioLayerNorm
    for layer in model.audio_tower.layers:
        layer.self_attn_layer_norm.__class__ = AudioLayerNorm
        layer.final_layer_norm.__class__ = AudioLayerNorm
