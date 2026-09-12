"""Native MLX implementation of the GPT-2 causal language model."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_kv_caches
from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.gpt2.config import GPT2Config
from mlx_one.models.shared.transformer import causal_mask


def gelu_new(value: Any) -> Any:
    return 0.5 * value * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)))


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        output_attentions: bool,
    ) -> tuple[Any, Any | None]:
        batch, length, hidden_size = hidden.shape
        query, key, value = mx.split(self.c_attn(hidden), 3, axis=-1)
        shape = (batch, length, self.n_head, self.head_dim)
        query = query.reshape(shape).transpose(0, 2, 1, 3)
        key = key.reshape(shape).transpose(0, 2, 1, 3)
        value = value.reshape(shape).transpose(0, 2, 1, 3)
        if cache is not None:
            key, value = cache.update(key, value)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        probabilities = self.attn_dropout(mx.softmax(scores, axis=-1, precise=True))
        attended = probabilities @ value
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, hidden_size)
        output = self.resid_dropout(self.c_proj(attended))
        return output, probabilities if output_attentions else None


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.inner_dim, bias=True)
        self.c_proj = nn.Linear(config.inner_dim, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def __call__(self, hidden: Any) -> Any:
        return self.dropout(self.c_proj(gelu_new(self.c_fc(hidden))))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        output_attentions: bool,
    ) -> tuple[Any, Any | None]:
        attended, weights = self.attn(
            self.ln_1(hidden),
            mask=mask,
            cache=cache,
            output_attentions=output_attentions,
        )
        hidden = hidden + attended
        return hidden + self.mlp(self.ln_2(hidden)), weights


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        cache: tuple[Any, ...] | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> tuple[Any, tuple[Any, ...] | None, tuple[Any, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if cache is not None and len(cache) != len(self.h):
            raise ValueError("cache count must match GPT-2 layer count")
        offset = 0 if cache is None else cache[0].offset
        length = input_ids.shape[1]
        if offset + length > self.config.n_positions:
            raise ValueError("GPT-2 sequence exceeds n_positions")
        if position_ids is None:
            position_ids = mx.arange(offset, offset + length)[None, :]
        if position_ids.shape not in {(1, length), input_ids.shape}:
            raise ValueError("position_ids must have shape [1, sequence] or match input_ids")
        hidden = self.drop(self.wte(input_ids) + self.wpe(position_ids))
        mask = causal_mask(length, offset)
        if attention_mask is not None:
            expected_mask_shape = (input_ids.shape[0], offset + length)
            if attention_mask.ndim != 2 or attention_mask.shape != expected_mask_shape:
                raise ValueError("attention_mask must cover the complete cached sequence")
            padding = mx.where(attention_mask[:, None, None, :] != 0, 0.0, -1e9)
            mask = padding if mask is None else mask + padding
        states = [hidden] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for index, block in enumerate(self.h):
            hidden, weights = block(
                hidden,
                mask=mask,
                cache=None if cache is None else cache[index],
                output_attentions=output_attentions,
            )
            if states is not None:
                states.append(hidden)
            if attentions is not None:
                attentions.append(weights)
        hidden = self.ln_f(hidden)
        if states is not None:
            states[-1] = hidden
        return (
            hidden,
            tuple(states) if states is not None else None,
            (tuple(attentions) if attentions is not None else None),
        )


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)

    def make_cache(self) -> tuple[Any, ...]:
        return make_kv_caches(self.config.n_layer)

    def __call__(self, input_ids: Any, **kwargs: Any) -> ModelOutput:
        cache = kwargs.get("cache")
        hidden, states, attentions = self.transformer(input_ids, **kwargs)
        return ModelOutput(
            self.transformer.wte.as_linear(hidden),
            hidden,
            cache=cache,
            hidden_states=states,
            attentions=attentions,
        )
