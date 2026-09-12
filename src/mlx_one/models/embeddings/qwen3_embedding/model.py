"""Native Qwen3 final-token sentence embedding model."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput
from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.language.qwen3.model import Qwen3Model
from mlx_one.models.shared.encoder import cosine_similarity, l2_normalize


def last_token_pool(hidden: Any, attention_mask: Any) -> Any:
    if hidden.ndim != 3 or attention_mask.ndim != 2 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError(
            "hidden states and attention mask must share batch and sequence dimensions"
        )
    mask = attention_mask.astype(mx.bool_)
    if mx.any(mx.sum(mask, axis=1) == 0).item():
        raise ValueError("each embedding input must contain at least one active token")
    positions = mx.arange(mask.shape[1])[None, :]
    indexes = mx.max(mx.where(mask, positions, -1), axis=1)
    return hidden[mx.arange(hidden.shape[0]), indexes]


class Qwen3ForEmbedding(nn.Module):
    def __init__(self, config: Qwen3EmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config.text_config)

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        dimensions: int | None = None,
        output_hidden_states: bool = False,
    ) -> EmbeddingModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_length:
            raise ValueError("input sequence exceeds max_seq_length")
        mask = mx.ones_like(input_ids, dtype=mx.bool_) if attention_mask is None else attention_mask
        hidden, _, states = self.model(
            input_ids,
            attention_mask=mask,
            output_hidden_states=output_hidden_states,
        )
        size = self.config.validate_dimensions(dimensions)
        embeddings = last_token_pool(hidden, mask)[..., :size]
        if self.config.normalize_embeddings:
            embeddings = l2_normalize(embeddings)
        return EmbeddingModelOutput(
            embeddings,
            hidden,
            mask,
            hidden_states=states,
        )

    @staticmethod
    def similarity(left: Any, right: Any) -> Any:
        return cosine_similarity(left, right)
