# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
"""Native GPT-OSS, adapted from mlx-lm gpt_oss.py. See models/UPSTREAM_LICENSE.

YaRN also honors the checkpoint's truncate flag (Transformers' reference),
which older MLX-LM rope_utils snapshots did not propagate.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import KVCache
from mlx_one.core.outputs import ModelOutput
from mlx_one.core.sliding_cache import SlidingKVCache
from mlx_one.models.shared.decoder import attention_mask_for
from mlx_one.models.shared.moe_decoder import validate_decoder_state
from mlx_one.models.shared.sparse import SelectedExperts
from mlx_one.ops.logits import project_logits


class YarnRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dims = config.head_dim
        scaling = config.rope_scaling or {}
        factor = scaling.get("factor", 1.0)
        original = scaling.get("original_max_position_embeddings", config.initial_context_length)

        def correction(rotations):
            return (
                self.dims
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(config.rope_theta))
            )

        low, high = (
            correction(scaling.get("beta_fast", 32)),
            correction(scaling.get("beta_slow", 1)),
        )
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, self.dims - 1)
        if low == high:
            high += 0.001
        ramp = mx.clip((mx.arange(self.dims // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
        base_freqs = config.rope_theta ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims)
        self._freqs = base_freqs / ((1 - ramp) + ramp / factor)
        # This non-parameter constant must not retain a lazy initialization
        # stream when a loaded bundle moves to the Engine's device owner.
        mx.eval(self._freqs)

        def mscale(value):
            return 1.0 if factor <= 1 else 0.1 * value * math.log(factor) + 1

        self.mscale = scaling.get(
            "attention_factor",
            mscale(scaling.get("mscale", 1)) / mscale(scaling.get("mscale_all_dim", 0)),
        )

    def __call__(self, x, offset=0, position_ids=None):
        x = x * self.mscale
        if position_ids is None:
            return mx.fast.rope(
                x,
                self.dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=self._freqs,
            )
        angles = position_ids[:, None, :, None].astype(mx.float32) / self._freqs
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        left, right = mx.split(x, 2, axis=-1)
        return mx.concatenate((left * cos - right * sin, right * cos + left * sin), axis=-1)


class Attention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.config = config
        self.window = (
            config.sliding_window if config.layer_types[index] == "sliding_attention" else None
        )
        d, h, kv, head = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj = nn.Linear(d, h * head, bias=True)
        self.k_proj = nn.Linear(d, kv * head, bias=True)
        self.v_proj = nn.Linear(d, kv * head, bias=True)
        self.o_proj = nn.Linear(h * head, d, bias=True)
        self.sinks = mx.zeros((h,))
        self.rope = YarnRoPE(config)

    def __call__(self, x, *, cache=None, attention_mask=None, position_ids=None):
        b, n, _ = x.shape
        c = self.config
        q = self.q_proj(x).reshape(b, n, c.num_attention_heads, c.head_dim).swapaxes(1, 2)
        k = self.k_proj(x).reshape(b, n, c.num_key_value_heads, c.head_dim).swapaxes(1, 2)
        v = self.v_proj(x).reshape(b, n, c.num_key_value_heads, c.head_dim).swapaxes(1, 2)
        offset = 0 if cache is None else cache.offset
        retained = min(offset, getattr(cache, "max_size", offset)) if cache is not None else 0
        q, k = (self.rope(t, offset=offset, position_ids=position_ids) for t in (q, k))
        if cache is not None:
            k, v = cache.update(k, v)
            k, v = k.astype(q.dtype), v.astype(q.dtype)
        mask = attention_mask_for(
            n, offset, retained, window=self.window, padding=attention_mask, dtype=q.dtype
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=c.head_dim**-0.5, mask=mask, sinks=self.sinks.astype(q.dtype)
        )
        return self.o_proj(out.swapaxes(1, 2).reshape(b, n, -1))


class SparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=True)
        self.experts = SelectedExperts(
            config.hidden_size,
            config.intermediate_size,
            config.num_local_experts,
            bias=True,
            swiglu_limit=config.swiglu_limit,
        )
        self._router_logits = None

    def __call__(self, x):
        logits = self.router(x)
        self._router_logits = logits
        indices = mx.stop_gradient(
            mx.argpartition(logits, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        )
        scores = mx.softmax(mx.take_along_axis(logits, indices, axis=-1), axis=-1, precise=True)
        return (self.experts(x, indices) * scores[..., None]).sum(axis=-2)


class DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = Attention(config, index)
        self.mlp = SparseMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, **settings):
        x = x + self.self_attn(self.input_layernorm(x), **settings)
        return x + self.mlp(self.post_attention_layernorm(x))


class GptOssModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        tokens,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        _, position_ids = validate_decoder_state(self, tokens, cache, attention_mask, position_ids)
        x = self.embed_tokens(tokens)
        states = [x] if output_hidden_states else None
        for layer, state in zip(self.layers, cache or (None,) * len(self.layers), strict=True):
            x = layer(x, cache=state, attention_mask=attention_mask, position_ids=position_ids)
            if states is not None:
                states.append(x)
        x = self.norm(x)
        if states is not None:
            states[-1] = x
        return x, tuple(states) if states is not None else None


class GptOssForCausalLM(nn.Module):
    _mlx_one_reference_first_math = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self):
        caches = tuple(
            SlidingKVCache(self.config.sliding_window) if kind == "sliding_attention" else KVCache()
            for kind in self.config.layer_types
        )
        if getattr(self, "_mlx_one_tq3", False):
            from mlx_one.core.cache import configure_kv_caches

            return configure_kv_caches(caches, "f16", "f16")
        return caches

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
        logits_mode=None,
    ):
        hidden, states = self.model(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = project_logits(hidden, self.lm_head, logits_mode, attention_mask=attention_mask)
        routers = tuple(layer.mlp._router_logits for layer in self.model.layers)
        return ModelOutput(
            logits,
            hidden,
            cache=cache,
            hidden_states=states,
            router_logits=routers if self.config.output_router_logits else None,
        )
