"""Native MLX implementation of dense Qwen2/Qwen2.5."""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.shared.transformer import DecoderLayer, causal_mask


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, qkv_bias=True) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Any | None,
        *,
        cache: tuple[Any, ...] | None = None,
        input_embeddings: Any | None = None,
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
        states = [hidden] if output_hidden_states else None
        layer_caches = cache or (None,) * len(self.layers)
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
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


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: Any,
        *,
        cache: tuple[Any, ...] | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden, cache, states = self.model(
            input_ids,
            cache=cache,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)
