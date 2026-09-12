"""Native Qwen3 yes/no cross-encoder reranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
from mlx_one.models.language.qwen3.model import Qwen3Model


@dataclass
class RerankerModelOutput:
    scores: Any
    probabilities: Any
    last_hidden_state: Any
    logits: Any


class Qwen3ForReranking(nn.Module):
    def __init__(self, config: Qwen3RerankerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config.text_config)

    def __call__(
        self,
        input_ids: Any,
        *,
        attention_mask: Any,
        false_token_id: int,
        true_token_id: int,
    ) -> RerankerModelOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_length:
            raise ValueError("input sequence exceeds max_seq_length")
        if not 0 <= false_token_id < self.config.text_config.vocab_size:
            raise ValueError("false_token_id must be within the vocabulary")
        if not 0 <= true_token_id < self.config.text_config.vocab_size:
            raise ValueError("true_token_id must be within the vocabulary")
        hidden, _, _ = self.model(input_ids, attention_mask=attention_mask)
        logits = self.model.embed_tokens.as_linear(hidden[:, -1])
        selected = mx.stack((logits[:, false_token_id], logits[:, true_token_id]), axis=-1)
        probabilities = mx.softmax(selected, axis=-1, precise=True)[:, 1]
        return RerankerModelOutput(
            selected[:, 1] - selected[:, 0], probabilities, hidden, selected
        )
