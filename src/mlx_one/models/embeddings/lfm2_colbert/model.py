"""Native MLX LFM2-ColBERT token encoder and MaxSim scorer."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput
from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
from mlx_one.models.language.lfm2.model import Lfm2Model


class Lfm2ColBERTModel(nn.Module):
    def __init__(self, config: Lfm2ColBERTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config.text_config)
        self.projection = nn.Linear(
            config.text_config.hidden_size, config.projection_dim, bias=False
        )

    def __call__(
        self, input_ids: Any, *, attention_mask: Any | None = None
    ) -> EmbeddingModelOutput:
        hidden, _, _ = self.model(input_ids, attention_mask=attention_mask)
        embeddings = self.projection(hidden)
        if self.config.normalize:
            norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
            embeddings = embeddings / mx.maximum(norm, 1e-12)
        if attention_mask is not None:
            embeddings = mx.where(attention_mask[..., None], embeddings, 0)
        return EmbeddingModelOutput(embeddings, hidden, attention_mask)

    @staticmethod
    def maxsim(
        query: Any,
        document: Any,
        *,
        query_mask: Any | None = None,
        document_mask: Any | None = None,
    ) -> Any:
        similarities = query @ document.transpose(0, 2, 1)
        if document_mask is not None:
            similarities = mx.where(document_mask[:, None, :], similarities, -float("inf"))
        maxima = mx.max(similarities, axis=-1)
        if query_mask is not None:
            maxima = mx.where(query_mask, maxima, 0)
        return mx.sum(maxima, axis=-1)
