# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
"""Model: LFM2 mixture of experts

Paper: LFM2 Technical Report
Paper link: https://arxiv.org/abs/2511.23404
Architecture gist: LFM2 hybrid sequence layers pair attention or convolutional state with sparse
expert feed-forward routing.
mlx-one implementation notes: Native MLX LFM2-MoE routing, experts, and hybrid cache topology.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_hybrid_caches
from mlx_one.core.outputs import ModelOutput
from mlx_one.core.sliding_cache import SlidingKVCache
from mlx_one.models.language.lfm2.model import Lfm2Model
from mlx_one.models.language.lfm2.reference_math import MLP, Lfm2DecoderLayer
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
from mlx_one.models.shared.decoder import attention_mask_for
from mlx_one.models.shared.moe import router_auxiliary_loss
from mlx_one.models.shared.moe_decoder import validate_decoder_state
from mlx_one.models.shared.sparse import SelectedExperts


class LfmSparseMoe(nn.Module):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SelectedExperts(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.expert_bias = mx.zeros((config.num_experts,)) if config.use_expert_bias else None
        self._router_logits: Any | None = None
        self._top_indices: Any | None = None

    @property
    def router_logits(self):
        return self._router_logits

    @property
    def top_indices(self):
        return self._top_indices

    def __call__(self, value: Any) -> Any:
        self._router_logits = self.gate(value)
        probabilities = mx.sigmoid(self.router_logits)
        routed = probabilities
        if self.expert_bias is not None:
            routed = routed.astype(mx.float32) + self.expert_bias
        self._top_indices = mx.stop_gradient(
            mx.argpartition(-routed, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        scores = mx.take_along_axis(probabilities, self.top_indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
        scores = scores * self.routed_scaling_factor
        selected = self.switch_mlp(value, self.top_indices)
        return mx.sum(selected * scores[..., None].astype(value.dtype), axis=-2)


class Lfm2MoeModel(Lfm2Model):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for index in range(config.num_hidden_layers):
            layer = Lfm2DecoderLayer(config, index)
            layer.feed_forward = (
                MLP(
                    config.hidden_size,
                    config.intermediate_size,
                    config.block_multiple_of,
                    False,
                    None,
                )
                if index < config.num_dense_layers
                else LfmSparseMoe(config)
            )
            self.layers.append(layer)
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        attention_mask=None,
        position_ids=None,
        output_hidden_states=False,
    ):
        _, position_ids = validate_decoder_state(
            self, input_ids, cache, attention_mask, position_ids
        )
        hidden = self.embed_tokens(input_ids)
        states = [hidden] if output_hidden_states else None
        for layer, state in zip(self.layers, cache or (None,) * len(self.layers), strict=True):
            offset = 0 if state is None else state.offset
            if callable(getattr(state, "attend", None)):
                mask = None
            elif layer.is_attention_layer:
                retained = min(offset, getattr(state, "max_size", offset))
                window = (
                    self.config.sliding_window if layer.layer_type == "sliding_attention" else None
                )
                mask = attention_mask_for(
                    hidden.shape[1],
                    offset,
                    retained,
                    padding=attention_mask,
                    dtype=hidden.dtype,
                    window=window,
                )
            else:
                mask = None if attention_mask is None else attention_mask[:, -hidden.shape[1] :]
            hidden = layer(hidden, mask=mask, cache=state, position_ids=position_ids)
            if states is not None:
                states.append(hidden)
        hidden = self.embedding_norm(hidden)
        if states is not None:
            states[-1] = hidden
        return hidden, cache, tuple(states) if states is not None else None


class Lfm2MoeForCausalLM(nn.Module):
    _mlx_one_reference_first_math = True

    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2MoeModel(config)
        if not config.tie_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> tuple[Any, ...]:
        caches = make_hybrid_caches(tuple(self.config.layer_types), self.config.conv_L_cache)
        return tuple(
            SlidingKVCache(self.config.sliding_window) if kind == "sliding_attention" else cache
            for kind, cache in zip(self.config.layer_types, caches, strict=True)
        )

    def __call__(
        self,
        input_ids: Any,
        *,
        cache: tuple[Any, ...] | None = None,
        attention_mask: Any | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden, cache, states = self.model(
            input_ids,
            cache=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_embedding
            else self.lm_head(hidden)
        )
        all_routers = tuple(
            layer.feed_forward.router_logits
            for layer in self.model.layers[self.config.num_dense_layers :]
        )
        aux_loss = None
        if self.config.router_aux_loss_coef and all_routers:
            aux_loss = self.config.router_aux_loss_coef * router_auxiliary_loss(
                all_routers,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask=attention_mask,
            )
        routers = all_routers if self.config.output_router_logits else None
        return ModelOutput(
            logits,
            hidden,
            cache=cache,
            hidden_states=states,
            router_logits=routers,
            router_aux_loss=aux_loss,
        )
