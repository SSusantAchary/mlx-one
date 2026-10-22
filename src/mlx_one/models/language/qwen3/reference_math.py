# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: MIT
"""qwen3 blocks ported from local mlx_lm/models/qwen3.py.

Source SHA256: 7e29b7055d8e526a0ede6e2740a3038ad24d0cef6a1abb446c411d3fbd0efa2d.
Only cache.update, explicit positions and MLX-One's existing paged-cache dispatch
adapt the source blocks. No reference library is imported at runtime.
See models/UPSTREAM_LICENSE and reference-source-provenance.md.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.reference_math import (
    initialize_rope,
    rotary,
    scaled_dot_product_attention,
    swiglu,
)


class Attention(nn.Module):
    def __init__(self, args: Any):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.args = args
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
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

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = rotary(queries, self.rope, self.args, cache.offset, position_ids)
            keys = rotary(keys, self.rope, self.args, cache.offset, position_ids)
            if callable(getattr(cache, "attend", None)):
                output = cache.attend(
                    queries, keys, values, scale=self.scale, position_ids=position_ids
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)
            keys, values = cache.update(keys, values)
        else:
            queries = rotary(queries, self.rope, self.args, 0, position_ids)
            keys = rotary(keys, self.rope, self.args, 0, position_ids)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        position_ids=None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
