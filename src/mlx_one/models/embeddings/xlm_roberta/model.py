"""Native XLM-R post-norm encoder and checkpoint-specific retrieval heads.

XLM-R shares BERT's encoder equations. Its non-padding position IDs start after
the padding index; its sequence classifier uses tanh(dense(CLS)), not a BERT
pooler. BGE-M3 adds learned sparse and token-vector heads to the encoder.
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.outputs import EmbeddingModelOutput, RerankerModelOutput
from mlx_one.models.embeddings.bert.model import BertEmbeddings, BertEncoder
from mlx_one.models.embeddings.xlm_roberta.config import XLMRobertaConfig
from mlx_one.models.shared.encoder import bidirectional_attention_mask, l2_normalize


@dataclass
class MultiRepresentationOutput(EmbeddingModelOutput):
    sparse_weights: Any = None
    colbert_embeddings: Any = None


class ClassificationHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.dense = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, 1)

    def __call__(self, hidden: Any) -> Any:
        return self.out_proj(mx.tanh(self.dense(hidden[:, 0])))


class XLMRobertaForRetrieval(nn.Module):
    def __init__(self, config: XLMRobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.classifier = (
            ClassificationHead(config.hidden_size) if config.task == "reranking" else None
        )
        self.sparse_linear = (
            nn.Linear(config.hidden_size, 1) if config.multi_representation else None
        )
        self.colbert_linear = (
            nn.Linear(config.hidden_size, config.colbert_dim)
            if config.multi_representation
            else None
        )

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any = None,
        token_type_ids: Any = None,
        position_ids: Any = None,
        output_hidden_states: bool = False,
        representations: bool = False,
    ):
        if input_ids.ndim != 2 or not all(input_ids.shape):
            raise ValueError("XLM-R requires nonempty [batch, sequence] input_ids")
        if input_ids.shape[1] > self.config.sentence_max_length:
            raise ValueError("XLM-R input exceeds the configured sentence limit")
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position_ids is None:
            nonpad = (input_ids != self.config.pad_token_id).astype(mx.int32)
            position_ids = mx.cumsum(nonpad, axis=1) * nonpad + self.config.pad_token_id
        hidden = self.embeddings(input_ids, token_type_ids, position_ids)
        hidden, states = self.encoder(
            hidden, bidirectional_attention_mask(attention_mask, hidden.dtype), output_hidden_states
        )
        if self.classifier is not None:
            logits = self.classifier(hidden)
            return RerankerModelOutput(logits[:, 0], mx.sigmoid(logits[:, 0]), hidden, logits)
        dense = l2_normalize(hidden[:, 0])
        if representations:
            if self.sparse_linear is None:
                raise ValueError("this checkpoint has no learned sparse or multi-vector heads")
            sparse = mx.maximum(self.sparse_linear(hidden)[..., 0], 0) * attention_mask
            vectors = l2_normalize(self.colbert_linear(hidden[:, 1:]))
            vectors = vectors * attention_mask[:, 1:, None]
            return MultiRepresentationOutput(
                dense,
                hidden,
                attention_mask,
                hidden_states=states,
                sparse_weights=sparse,
                colbert_embeddings=vectors,
            )
        return EmbeddingModelOutput(dense, hidden, attention_mask, hidden_states=states)
