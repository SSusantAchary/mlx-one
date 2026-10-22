"""Native Gemma 3 text: four offset norms, scaled embedding and local/global KV."""

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import KVCache
from mlx_one.core.sliding_cache import SlidingKVCache
from mlx_one.models.shared.decoder import CausalLM, attention_mask_for, prepare_decoder
from mlx_one.models.shared.transformer import apply_rope


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, values):
        return mx.fast.rms_norm(values, 1 + self.weight, self.eps)


def add_residual(left, right):
    if left.dtype != mx.float16:
        return left + right
    bound = mx.finfo(mx.float16).max
    return mx.clip(left.astype(mx.float32) + right.astype(mx.float32), -bound, bound).astype(
        mx.float16
    )


class Gemma3Attention(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.config = config
        self.is_sliding = config.layer_types[index] == "sliding_attention"
        d, h, kv, head = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj, self.k_proj = (
            nn.Linear(d, h * head, bias=False),
            nn.Linear(d, kv * head, bias=False),
        )
        self.v_proj, self.o_proj = (
            nn.Linear(d, kv * head, bias=False),
            nn.Linear(h * head, d, bias=False),
        )
        self.q_norm = Gemma3RMSNorm(head, config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(head, config.rms_norm_eps)

    def __call__(self, hidden, *, cache=None, attention_mask=None, position_ids=None):
        c = self.config
        b, n, _ = hidden.shape
        q = (
            self.q_proj(hidden)
            .reshape(b, n, c.num_attention_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(hidden)
            .reshape(b, n, c.num_key_value_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden)
            .reshape(b, n, c.num_key_value_heads, c.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        offset = 0 if cache is None else cache.offset
        retained = min(offset, getattr(cache, "max_size", offset)) if cache is not None else 0
        scaling = {} if self.is_sliding else c.rope_scaling or {}
        theta = c.rope_local_base_freq if self.is_sliding else c.rope_theta
        settings = {
            "offset": offset,
            "theta": theta,
            "position_ids": position_ids,
            "scaling_factor": scaling.get("factor", 1.0),
        }
        q, k = apply_rope(q, **settings), apply_rope(k, **settings)
        if cache is not None:
            k, v = cache.update(k, v)
            k, v = k.astype(q.dtype), v.astype(q.dtype)
        mask = attention_mask_for(
            n,
            offset,
            retained,
            padding=attention_mask,
            dtype=q.dtype,
            window=c.sliding_window if self.is_sliding else None,
        )
        scale = c.query_pre_attn_scalar**-0.5
        if c.attn_logit_softcapping is None:
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        else:
            k, v = (
                mx.repeat(x, c.num_attention_heads // c.num_key_value_heads, axis=1) for x in (k, v)
            )
            cap = c.attn_logit_softcapping
            scores = cap * mx.tanh((q @ k.transpose(0, 1, 3, 2)) * scale / cap)
            output = mx.softmax(scores + mask, axis=-1, precise=True) @ v
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(b, n, -1))


class Gemma3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden):
        return self.down_proj(nn.gelu_approx(self.gate_proj(hidden)) * self.up_proj(hidden))


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.self_attn = Gemma3Attention(config, index)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, hidden, **settings):
        mixed = self.self_attn(self.input_layernorm(hidden), **settings)
        hidden = add_residual(hidden, self.post_attention_layernorm(mixed))
        mixed = self.mlp(self.pre_feedforward_layernorm(hidden))
        return add_residual(hidden, self.post_feedforward_layernorm(mixed))


class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        _, position_ids = prepare_decoder(self, input_ids, cache, attention_mask, position_ids)
        hidden = self.embed_tokens(input_ids)
        # Match the reference's checkpoint-precision embedding multiplier.
        hidden = hidden * mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(hidden.dtype)
        states = [hidden] if output_hidden_states else None
        for layer, state in zip(self.layers, cache or (None,) * len(self.layers), strict=True):
            hidden = layer(
                hidden, cache=state, attention_mask=attention_mask, position_ids=position_ids
            )
            if states is not None:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, tuple(states) if states is not None else None


class Gemma3ForCausalLM(CausalLM):
    _mlx_one_reference_first_math = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self):
        return tuple(
            SlidingKVCache(self.config.sliding_window) if kind == "sliding_attention" else KVCache()
            for kind in self.config.layer_types
        )
