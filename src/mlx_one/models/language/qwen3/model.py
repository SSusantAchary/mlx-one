"""Native MLX implementation of dense Qwen3."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.shared.transformer import DecoderLayer, causal_mask


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, qkv_bias=False, qk_norm=True)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Any | None,
        *,
        cache: tuple[Any, ...] | None = None,
        input_embeddings: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Any, tuple[Any, ...] | None, tuple[Any, ...] | None]:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("provide exactly one of input_ids or input_embeddings")
        hidden = self.embed_tokens(input_ids) if input_embeddings is None else input_embeddings
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache count must match decoder layer count")
        offset = 0 if cache is None else cache[0].offset
        mask = causal_mask(hidden.shape[1], offset)
        if attention_mask is not None:
            expected = (hidden.shape[0], offset + hidden.shape[1])
            if attention_mask.shape != expected:
                raise ValueError("attention_mask must cover the complete cached sequence")
            if position_ids is None:
                positions = mx.maximum(mx.cumsum(attention_mask, axis=-1) - 1, 0)
                position_ids = positions[:, -hidden.shape[1] :]
            padding = mx.where(attention_mask[:, None, None, :] != 0, 0.0, -1e9)
            mask = padding if mask is None else mask + padding
        states = [hidden] if output_hidden_states else None
        for layer, layer_cache in zip(
            self.layers, cache or (None,) * len(self.layers), strict=True
        ):
            hidden = layer(
                hidden,
                mask=mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
            if states is not None:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, cache, tuple(states) if states is not None else None


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: Any, **kwargs: Any) -> ModelOutput:
        hidden, cache, states = self.model(input_ids, **kwargs)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)
