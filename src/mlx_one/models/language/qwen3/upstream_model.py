# Copyright © 2023-2024 Apple Inc.
# SPDX-License-Identifier: MIT
"""Native Qwen3 candidate adapted from Apple's mlx-lm v0.31.3 Qwen3.

Pinned source: ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/qwen3.py.
See UPSTREAM_LICENSE. MLX-One additions adapt outputs, configuration and cache
ownership; attention, projections, normalization, RoPE and MLP use upstream math.
The canonical implementation in model.py is deliberately preserved unchanged.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_kv_caches
from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.modes import ModelCapabilityUnsupported
from mlx_one.models.language.qwen3.upstream_adapter import cache_views
from mlx_one.models.language.qwen3.upstream_math import (
    create_attention_mask,
    initialize_rope,
    scaled_dot_product_attention,
    swiglu,
)
from mlx_one.ops.logits import project_logits


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim, base=args.rope_theta, traditional=False, scaling_config=args.rope_scaling
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(batch, length, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(batch, length, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None, *, output_hidden_states=False):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        caches = [None] * len(self.layers) if cache is None else cache
        mask = create_attention_mask(h, caches[0])
        states = [h] if output_hidden_states else None
        for layer, entry in zip(self.layers, caches, strict=True):
            h = layer(h, mask, entry)
            if states is not None:
                states.append(h)
        h = self.norm(h)
        if states is not None:
            states[-1] = h
        return h, tuple(states) if states is not None else None


class Qwen3UpstreamForCausalLM(nn.Module):
    """MLX-One output adapter; candidate-only, dense unpadded B1 qualification scope."""

    _mlx_one_qwen3_candidate = True

    def __init__(self, config):
        super().__init__()
        if config.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        self.config = self.args = config
        self.model_type = config.model_type
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return make_kv_caches(self.config.num_hidden_layers)

    def __call__(
        self,
        input_ids=None,
        *,
        cache=None,
        input_embeddings=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
        logits_mode=None,
    ):
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of input_ids or input_embeddings")
        value = input_ids if input_embeddings is None else input_embeddings
        if value.ndim != (2 if input_embeddings is None else 3) or value.shape[1] < 1:
            raise ValueError("candidate input must have a nonempty sequence dimension")
        if value.shape[0] != 1:
            raise ModelCapabilityUnsupported("Qwen3 candidate supports reference B1 only")
        views = cache_views(cache, len(self.layers))
        offset = 0 if views is None else views[0].offset
        if attention_mask is not None:
            if attention_mask.shape != (1, offset + value.shape[1]):
                raise ValueError("attention_mask must cover the complete cached sequence")
            if not bool(mx.all(attention_mask != 0).item()):
                raise ModelCapabilityUnsupported("Qwen3 candidate padding is not qualified")
        if position_ids is not None:
            expected = mx.arange(offset, offset + value.shape[1])[None, :]
            if position_ids.shape != expected.shape or not bool(
                mx.all(position_ids == expected).item()
            ):
                raise ModelCapabilityUnsupported("Qwen3 candidate requires contiguous positions")
        hidden, states = self.model(
            input_ids, views, input_embeddings, output_hidden_states=output_hidden_states
        )
        projection = (
            self.model.embed_tokens.as_linear if self.config.tie_word_embeddings else (self.lm_head)
        )
        logits = project_logits(hidden, projection, logits_mode, attention_mask=attention_mask)
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)
