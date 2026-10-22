# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
"""LFM2 blocks adapted from local mlx_lm/models/lfm2.py.
Source SHA256: 558dcbec61442b1f7216c0122d08d44083ea71d1510be23f4f934e62b0f71376.
The causal convolution keeps MLX-One's flat weight names and private cache owner.
See models/UPSTREAM_LICENSE.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.reference_math import rotary, scaled_dot_product_attention, swiglu


class ShortConv(nn.Module):
    def __init__(self, args, layer_idx):
        super().__init__()
        self.args = args
        self.L_cache = args.conv_L_cache
        conv = nn.Conv1d(
            args.hidden_size,
            args.hidden_size,
            self.L_cache,
            groups=args.hidden_size,
            bias=args.conv_bias,
        )
        self.conv = conv.weight[..., 0]
        self.conv_bias = conv.bias if args.conv_bias else None
        self.in_proj = nn.Linear(args.hidden_size, 3 * args.hidden_size, bias=args.conv_bias)
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size, bias=args.conv_bias)

    def __call__(self, x, mask=None, cache=None):
        BCx = self.in_proj(x)
        B, C, x = mx.split(BCx, 3, axis=-1)
        Bx = B * x
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)
        if cache is not None:
            Bx = cache.update(Bx)
        else:
            Bx = mx.pad(Bx, [(0, 0), (self.L_cache - 1, 0), (0, 0)])
        conv_out = mx.conv1d(Bx, self.conv[..., None], groups=self.args.hidden_size)
        if self.conv_bias is not None:
            conv_out = conv_out + self.conv_bias
        return self.out_proj(C * conv_out)


class Attention(nn.Module):
    def __init__(self, args: Any):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.hidden_size // n_heads

        self.scale = head_dim**-0.5

        self.q_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)
        self.k_layernorm = nn.RMSNorm(head_dim, eps=args.norm_eps)

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.args = args
        self.rope = nn.RoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids=None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_layernorm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_layernorm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = rotary(queries, self.rope, self.args, cache.offset, position_ids)
            keys = rotary(keys, self.rope, self.args, cache.offset, position_ids)
            if callable(getattr(cache, "attend", None)):
                output = cache.attend(
                    queries, keys, values, scale=self.scale, position_ids=position_ids
                )
                return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))
            keys, values = cache.update(keys, values)
        else:
            queries = rotary(queries, self.rope, self.args, 0, position_ids)
            keys = rotary(keys, self.rope, self.args, 0, position_ids)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, mask=mask, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_dim: int,
        multiple_of: int,
        auto_adjust_ff_dim: bool,
        ffn_dim_multiplier: float | None,
    ):
        super().__init__()
        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class Lfm2DecoderLayer(nn.Module):
    def __init__(self, args: Any, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_attention_layer = layer_idx in args.full_attn_idxs

        if self.is_attention_layer:
            self.self_attn = Attention(args)
        else:
            self.conv = ShortConv(args, layer_idx)
        self.feed_forward = MLP(
            dim=args.block_dim,
            ff_dim=args.block_ff_dim,
            multiple_of=args.block_multiple_of,
            auto_adjust_ff_dim=args.block_auto_adjust_ff_dim,
            ffn_dim_multiplier=args.block_ffn_dim_multiplier,
        )

        self.operator_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids=None,
    ) -> mx.array:

        if self.is_attention_layer:
            r = self.self_attn(
                self.operator_norm(x), mask=mask, cache=cache, position_ids=position_ids
            )
        else:
            r = self.conv(
                self.operator_norm(x),
                mask=mask,
                cache=cache,
            )
        h = x + r
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
