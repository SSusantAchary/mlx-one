"""Native MLX MPNet encoder with Sentence Transformers pooling."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput
from mlx_one.models.embeddings.mpnet.config import MPNetEmbeddingConfig
from mlx_one.models.shared.encoder import (
    bidirectional_attention_mask,
    cosine_similarity,
    l2_normalize,
    masked_mean_pool,
)


def _activation(value: Any, name: str) -> Any:
    if name == "gelu":
        return nn.gelu(value)
    return 0.5 * value * (
        1 + mx.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3))
    )


def create_position_ids_from_input_ids(input_ids: Any, padding_idx: int) -> Any:
    mask = input_ids != padding_idx
    incremental = mx.cumsum(mask.astype(mx.int32), axis=1) * mask
    return incremental.astype(mx.int32) + padding_idx


def relative_position_bucket(
    relative_position: Any, *, num_buckets: int = 32, max_distance: int = 128
) -> Any:
    """Exact bidirectional MPNet/T5 logarithmic relative-position buckets."""

    half = num_buckets // 2
    distance = -relative_position
    result = (distance < 0).astype(mx.int32) * half
    distance = mx.abs(distance)
    max_exact = half // 2
    large = max_exact + (
        mx.log(mx.maximum(distance, max_exact).astype(mx.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (half - max_exact)
    ).astype(mx.int32)
    large = mx.minimum(large, half - 1)
    return result + mx.where(distance < max_exact, distance, large)


class MPNetEmbeddings(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, input_ids: Any, position_ids: Any | None) -> Any:
        if position_ids is None:
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
        hidden = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.dropout(self.LayerNorm(hidden))


class MPNetSelfAttention(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def __call__(self, hidden: Any, mask: Any, position_bias: Any) -> Any:
        batch, length, dim = hidden.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.q(hidden).reshape(shape).transpose(0, 2, 1, 3)
        key = self.k(hidden).reshape(shape).transpose(0, 2, 1, 3)
        value = self.v(hidden).reshape(shape).transpose(0, 2, 1, 3)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + position_bias + mask
        attended = self.dropout(mx.softmax(scores, axis=-1, precise=True)) @ value
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return self.o(attended)


class MPNetAttention(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.attn = MPNetSelfAttention(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden: Any, mask: Any, position_bias: Any) -> Any:
        return self.LayerNorm(
            self.dropout(self.attn(hidden, mask, position_bias)) + hidden
        )


class MPNetIntermediate(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.activation = config.hidden_act

    def __call__(self, hidden: Any) -> Any:
        return _activation(self.dense(hidden), self.activation)


class MPNetOutput(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden: Any, residual: Any) -> Any:
        return self.LayerNorm(self.dropout(self.dense(hidden)) + residual)


class MPNetLayer(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.attention = MPNetAttention(config)
        self.intermediate = MPNetIntermediate(config)
        self.output = MPNetOutput(config)

    def __call__(self, hidden: Any, mask: Any, position_bias: Any) -> Any:
        attended = self.attention(hidden, mask, position_bias)
        return self.output(self.intermediate(attended), attended)


class MPNetEncoder(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = [MPNetLayer(config) for _ in range(config.num_hidden_layers)]
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets, config.num_attention_heads
        )

    def compute_position_bias(self, length: int) -> Any:
        context = mx.arange(length)[:, None]
        memory = mx.arange(length)[None, :]
        buckets = relative_position_bucket(
            memory - context,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance,
        )
        return self.relative_attention_bias(buckets).transpose(2, 0, 1)[None, ...]

    def __call__(
        self, hidden: Any, mask: Any, output_hidden_states: bool
    ) -> tuple[Any, tuple[Any, ...] | None]:
        states = [hidden] if output_hidden_states else None
        position_bias = self.compute_position_bias(hidden.shape[1])
        for layer in self.layer:
            hidden = layer(hidden, mask, position_bias)
            if states is not None:
                states.append(hidden)
        return hidden, tuple(states) if states is not None else None


class MPNetPooler(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, hidden: Any) -> Any:
        return mx.tanh(self.dense(hidden[:, 0]))


class MPNetForSentenceEmbedding(nn.Module):
    def __init__(self, config: MPNetEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = MPNetEmbeddings(config)
        self.encoder = MPNetEncoder(config)
        self.pooler = MPNetPooler(config) if config.add_pooling_layer else None

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> EmbeddingModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.sentence_max_length:
            raise ValueError("input sequence exceeds sentence_max_length")
        attention_mask = (
            input_ids != self.config.pad_token_id
            if attention_mask is None
            else attention_mask.astype(mx.bool_)
        )
        hidden = self.embeddings(input_ids, position_ids)
        mask = bidirectional_attention_mask(attention_mask, hidden.dtype)
        hidden, states = self.encoder(hidden, mask, output_hidden_states)
        pooled = masked_mean_pool(hidden, attention_mask)
        embeddings = l2_normalize(pooled) if self.config.normalize_embeddings else pooled
        pooler_output = self.pooler(hidden) if self.pooler is not None else None
        return EmbeddingModelOutput(
            embeddings,
            hidden,
            attention_mask,
            pooler_output=pooler_output,
            hidden_states=states,
        )

    @staticmethod
    def similarity(left: Any, right: Any) -> Any:
        return cosine_similarity(left, right)
