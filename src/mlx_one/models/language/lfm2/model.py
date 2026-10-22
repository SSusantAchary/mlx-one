"""Model: LFM2 / LFM2.5

Paper: LFM2 Technical Report
Paper link: https://arxiv.org/abs/2511.23404
Architecture gist: Hybrid sequence model that interleaves full attention with short convolutional
state layers for efficient on-device inference.
mlx-one implementation notes: Native MLX dense LFM2 blocks and complete hybrid cache construction.
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mlx_one.core.cache import make_hybrid_caches
from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.lfm2.config import Lfm2Config
from mlx_one.models.language.lfm2.reference_math import Lfm2DecoderLayer
from mlx_one.models.shared.hybrid import sliding_causal_mask
from mlx_one.models.shared.reference_math import attention_mask as causal_mask
from mlx_one.models.shared.transformer import causal_mask as legacy_causal_mask
from mlx_one.ops.logits import project_logits


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

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
        layer_caches = cache or (None,) * len(self.layers)
        states = [hidden] if output_hidden_states else None
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            offset = 0 if layer_cache is None else layer_cache.offset
            if callable(getattr(layer_cache, "attend", None)):
                mask = None
            elif layer.layer_type == "sliding_attention":
                mask = sliding_causal_mask(hidden.shape[1], offset, int(self.config.sliding_window))
            elif layer.is_attention_layer:
                mask = causal_mask(hidden.shape[1], offset)
            else:
                mask = None
            if isinstance(layer, Lfm2DecoderLayer):
                hidden = layer(
                    hidden,
                    mask=mask if layer.is_attention_layer else (
                        None if attention_mask is None else attention_mask[:, -hidden.shape[1]:]
                    ),
                    cache=layer_cache,
                    position_ids=position_ids,
                )
            else:
                # Existing LFM2 MoE/multimodal subclasses retain their own blocks.
                if isinstance(mask, str):
                    mask = legacy_causal_mask(hidden.shape[1], offset)
                hidden = layer(
                    hidden, attention_mask=mask, padding_mask=attention_mask,
                    cache=layer_cache, position_ids=position_ids,
                )
            if states is not None:
                states.append(hidden)
        hidden = self.embedding_norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, cache, tuple(states) if states is not None else None


class Lfm2ForCausalLM(nn.Module):
    _mlx_one_reference_first_math = True

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        if not config.tie_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> tuple[Any, ...]:
        return make_hybrid_caches(tuple(self.config.layer_types), self.config.conv_L_cache)

    def __call__(
        self,
        input_ids: Any | None,
        *,
        cache: tuple[Any, ...] | None = None,
        input_embeddings: Any | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
        logits_mode=None,
    ) -> ModelOutput:
        hidden, cache, states = self.model(
            input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = project_logits(
            hidden,
            self.model.embed_tokens.as_linear if self.config.tie_embedding else self.lm_head,
            logits_mode, attention_mask=attention_mask,
        )
        return ModelOutput(logits, hidden, cache=cache, hidden_states=states)
