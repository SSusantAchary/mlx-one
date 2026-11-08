"""Model: LFM2-ColBERT

Paper: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
Paper link: https://arxiv.org/abs/2004.12832
Architecture gist: LFM encoder produces contextual token embeddings; ColBERT late interaction scores
query and document tokens with MaxSim.
mlx-one implementation notes: Native MLX LFM2 token encoder and ColBERT-style MaxSim scorer.
"""

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
        self,
        input_ids: Any,
        *,
        attention_mask: Any | None = None,
        embedding_mask: Any | None = None,
    ) -> EmbeddingModelOutput:
        if self.config.query_prefix is None:
            hidden, _, _ = self.model(input_ids, attention_mask=attention_mask)
        else:
            # Expansion positions retain their EOS-token convolution states.
            # The publisher recipe masks only attention keys here. Exact-length
            # task groups avoid introducing unrelated document batch padding.
            hidden = self.model.embed_tokens(input_ids)
            length = input_ids.shape[1]
            causal = mx.triu(mx.full((length, length), -float("inf")), k=1)
            padding = (
                0
                if attention_mask is None
                else mx.where(attention_mask[:, None, None, :], 0.0, -float("inf"))
            )
            for layer in self.model.layers:
                hidden = layer(
                    hidden,
                    mask=causal + padding if layer.is_attention_layer else None,
                    cache=None,
                    position_ids=None,
                )
            hidden = self.model.embedding_norm(hidden)
        embeddings = self.projection(hidden)
        if self.config.normalize:
            norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
            embeddings = embeddings / mx.maximum(norm, 1e-12)
        keep = attention_mask if embedding_mask is None else embedding_mask
        if keep is not None:
            embeddings = mx.where(keep[..., None], embeddings, 0)
        return EmbeddingModelOutput(embeddings, hidden, keep)

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
