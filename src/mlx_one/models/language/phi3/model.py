"""Native Phi-3: packed projections, partial rotary embeddings and dense KV."""

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_kv_caches
from mlx_one.core.sliding_cache import SlidingKVCache
from mlx_one.models.shared.decoder import CausalLM, attention_mask_for, prepare_decoder
from mlx_one.models.shared.reference_math import swiglu
from mlx_one.models.shared.transformer import apply_rope


class Phi3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        q = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, q + 2 * kv, bias=False)
        self.o_proj = nn.Linear(q, config.hidden_size, bias=False)

    def rotary(self, values, offset, position_ids):
        config = self.config
        size = config.rotary_dim
        scaling = config.rope_scaling or {}
        kind = scaling.get("rope_type", scaling.get("type", "default"))
        rotated = values[..., :size]
        if kind in {"longrope", "su"}:
            # Phi3 reference selects factors by configured extension, not decode chunk length.
            extended = config.max_position_embeddings > config.original_max_position_embeddings
            factors = scaling["long_factor" if extended else "short_factor"]
            frequencies = config.rope_theta ** (mx.arange(0, size, 2) / size) * mx.array(factors)
            ratio = config.max_position_embeddings / config.original_max_position_embeddings
            gain = (
                math.sqrt(1 + math.log(ratio) / math.log(config.original_max_position_embeddings))
                if ratio > 1
                else 1.0
            )
            if position_ids is None:
                rotated = mx.fast.rope(
                    rotated * gain,
                    size,
                    traditional=False,
                    base=None,
                    scale=1.0,
                    freqs=frequencies,
                    offset=offset,
                )
            else:
                angles = position_ids[:, None, :, None].astype(mx.float32) / frequencies
                first, second = mx.split(rotated.astype(mx.float32) * gain, 2, axis=-1)
                cosine, sine = mx.cos(angles), mx.sin(angles)
                rotated = mx.concatenate(
                    (first * cosine - second * sine, first * sine + second * cosine), axis=-1
                ).astype(values.dtype)
        else:
            rotated = apply_rope(
                rotated,
                offset=offset,
                theta=config.rope_theta,
                position_ids=position_ids,
                scaling_factor=scaling.get("factor", 1.0) if kind == "linear" else 1.0,
            )
        return mx.concatenate((rotated, values[..., size:]), axis=-1)

    def __call__(self, hidden, *, mask, cache=None, position_ids=None):
        c = self.config
        b, n, _ = hidden.shape
        q_width = c.num_attention_heads * c.head_dim
        kv_width = c.num_key_value_heads * c.head_dim
        q, k, v = mx.split(self.qkv_proj(hidden), [q_width, q_width + kv_width], axis=-1)
        q = q.reshape(b, n, c.num_attention_heads, c.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, n, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, n, c.num_key_value_heads, c.head_dim).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        q, k = self.rotary(q, offset, position_ids), self.rotary(k, offset, position_ids)
        if cache is not None:
            k, v = cache.update(k, v)
            k, v = k.astype(q.dtype), v.astype(q.dtype)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=c.head_dim**-0.5, mask=mask)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(b, n, -1))


class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, hidden):
        gate, up = mx.split(self.gate_up_proj(hidden), 2, axis=-1)
        return self.down_proj(swiglu(gate, up))


class Phi3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Phi3Attention(config)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, hidden, **settings):
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), **settings)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Phi3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        offset, position_ids = prepare_decoder(self, input_ids, cache, attention_mask, position_ids)
        hidden = self.embed_tokens(input_ids)
        retained = min(offset, getattr(cache[0], "max_size", offset)) if cache else 0
        mask = attention_mask_for(
            input_ids.shape[1],
            offset,
            retained,
            padding=attention_mask,
            dtype=hidden.dtype,
            window=self.config.sliding_window,
        )
        states = [hidden] if output_hidden_states else None
        for layer, state in zip(self.layers, cache or (None,) * len(self.layers), strict=True):
            hidden = layer(hidden, mask=mask, cache=state, position_ids=position_ids)
            if states is not None:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, tuple(states) if states is not None else None


class Phi3ForCausalLM(CausalLM):
    _mlx_one_reference_first_math = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self):
        if self.config.sliding_window is not None:
            return tuple(
                SlidingKVCache(self.config.sliding_window)
                for _ in range(self.config.num_hidden_layers)
            )
        return make_kv_caches(self.config.num_hidden_layers)
