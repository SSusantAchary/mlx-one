"""Native MLX implementation of Apple OpenELM."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.openelm.config import OpenELMConfig
from mlx_one.models.shared.transformer import apply_rope, causal_mask


class OpenELMAttention(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_query_heads = config.num_query_heads[layer_index]
        self.num_kv_heads = config.num_kv_heads[layer_index]
        self.scale = config.head_dim**-0.5
        self.rope_theta = config.rope_freq_constant
        projection_heads = self.num_query_heads + 2 * self.num_kv_heads
        self.qkv_proj = nn.Linear(
            config.model_dim, projection_heads * config.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.num_query_heads * config.head_dim, config.model_dim, bias=False
        )
        if config.normalize_qk_projections:
            self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        position_ids: Any | None,
    ) -> Any:
        batch, length, _ = hidden.shape
        projected = self.qkv_proj(hidden).reshape(
            batch,
            length,
            self.num_query_heads + 2 * self.num_kv_heads,
            self.head_dim,
        )
        queries, keys, values = mx.split(
            projected,
            (self.num_query_heads, self.num_query_heads + self.num_kv_heads),
            axis=2,
        )
        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        queries = apply_rope(
            queries,
            offset=offset,
            theta=self.rope_theta,
            position_ids=position_ids,
        )
        keys = apply_rope(
            keys,
            offset=offset,
            theta=self.rope_theta,
            position_ids=position_ids,
        )
        if cache is not None:
            keys, values = cache.update(keys, values)
        repeats = self.num_query_heads // self.num_kv_heads
        if repeats != 1:
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        attended = probabilities @ values
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.out_proj(attended)


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int) -> None:
        super().__init__()
        self.intermediate_dim = config.ffn_dims[layer_index]
        self.ffn_with_glu = config.ffn_with_glu
        first_output = self.intermediate_dim * (2 if self.ffn_with_glu else 1)
        self.proj_1 = nn.Linear(config.model_dim, first_output, bias=False)
        self.proj_2 = nn.Linear(self.intermediate_dim, config.model_dim, bias=False)

    def __call__(self, value: Any) -> Any:
        projected = self.proj_1(value)
        if self.ffn_with_glu:
            gate, projected = mx.split(projected, 2, axis=-1)
            projected = (gate * mx.sigmoid(gate)) * projected
        else:
            projected = projected * mx.sigmoid(projected)
        return self.proj_2(projected)


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: OpenELMConfig, layer_index: int) -> None:
        super().__init__()
        self.attn = OpenELMAttention(config, layer_index)
        self.ffn = OpenELMFeedForwardNetwork(config, layer_index)
        self.attn_norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        position_ids: Any | None,
    ) -> Any:
        hidden = hidden + self.attn(
            self.attn_norm(hidden),
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


class OpenELMModel(nn.Module):
    def __init__(self, config: OpenELMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = [
            OpenELMDecoderLayer(config, index)
            for index in range(config.num_transformer_layers)
        ]
        self.norm = nn.RMSNorm(config.model_dim, eps=config.rms_norm_eps)

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
        hidden = (
            self.token_embeddings(input_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache count must match decoder layer count")
        offset = 0 if cache is None else cache[0].offset
        mask = causal_mask(hidden.shape[1], offset)
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


class OpenELMForCausalLM(nn.Module):
    def __init__(self, config: OpenELMConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = OpenELMModel(config)
        if not config.share_input_output_layers:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def __call__(self, input_ids: Any, **kwargs: Any) -> ModelOutput:
        hidden, cache, states = self.transformer(input_ids, **kwargs)
        logits = (
            self.transformer.token_embeddings.as_linear(hidden)
            if self.config.share_input_output_layers
            else self.lm_head(hidden)
        )
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)
