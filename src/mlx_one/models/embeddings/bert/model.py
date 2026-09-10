"""Native MLX BERT encoder with Sentence Transformers pooling."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput
from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig
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


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self, input_ids: Any, token_type_ids: Any | None, position_ids: Any | None
    ) -> Any:
        batch, length = input_ids.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(mx.arange(length)[None, :], (batch, length))
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        hidden = self.word_embeddings(input_ids)
        hidden = hidden + self.position_embeddings(position_ids)
        hidden = hidden + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.LayerNorm(hidden))


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def __call__(self, hidden: Any, mask: Any) -> Any:
        batch, length, dim = hidden.shape
        shape = (batch, length, self.num_heads, self.head_dim)
        query = self.query(hidden).reshape(shape).transpose(0, 2, 1, 3)
        key = self.key(hidden).reshape(shape).transpose(0, 2, 1, 3)
        value = self.value(hidden).reshape(shape).transpose(0, 2, 1, 3)
        scores = (query @ key.transpose(0, 1, 3, 2)) * self.scale + mask
        attended = self.dropout(mx.softmax(scores, axis=-1, precise=True)) @ value
        return attended.transpose(0, 2, 1, 3).reshape(batch, length, dim)


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden: Any, residual: Any) -> Any:
        return self.LayerNorm(self.dropout(self.dense(hidden)) + residual)


class BertAttention(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(self, hidden: Any, mask: Any) -> Any:
        return self.output(self.self_attn(hidden, mask), hidden)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.activation = config.hidden_act

    def __call__(self, hidden: Any) -> Any:
        return _activation(self.dense(hidden), self.activation)


class BertOutput(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden: Any, residual: Any) -> Any:
        return self.LayerNorm(self.dropout(self.dense(hidden)) + residual)


class BertLayer(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(self, hidden: Any, mask: Any) -> Any:
        attended = self.attention(hidden, mask)
        return self.output(self.intermediate(attended), attended)


class BertEncoder(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, hidden: Any, mask: Any, output_hidden_states: bool
    ) -> tuple[Any, tuple[Any, ...] | None]:
        states = [hidden] if output_hidden_states else None
        for layer in self.layer:
            hidden = layer(hidden, mask)
            if states is not None:
                states.append(hidden)
        return hidden, tuple(states) if states is not None else None


class BertPooler(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, hidden: Any) -> Any:
        return mx.tanh(self.dense(hidden[:, 0]))


class BertForSentenceEmbedding(nn.Module):
    def __init__(self, config: BertEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if config.add_pooling_layer else None

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        token_type_ids: Any | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> EmbeddingModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.sentence_max_length:
            raise ValueError("input sequence exceeds sentence_max_length")
        attention_mask = (
            mx.ones_like(input_ids, dtype=mx.bool_)
            if attention_mask is None
            else attention_mask.astype(mx.bool_)
        )
        hidden = self.embeddings(input_ids, token_type_ids, position_ids)
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
