"""Native MLX Liquid LFM2 mixture-of-experts model."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_one.core.cache import make_hybrid_caches
from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.lfm2.model import Lfm2Model
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
from mlx_one.models.shared.hybrid import HybridDecoderLayer, LfmMLP
from mlx_one.models.shared.moe import StackedExperts, router_auxiliary_loss


class LfmSparseMoe(nn.Module):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = StackedExperts(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.expert_bias = mx.zeros((config.num_experts,)) if config.use_expert_bias else None
        self.router_logits: Any | None = None
        self.top_indices: Any | None = None

    def __call__(self, value: Any) -> Any:
        self.router_logits = self.gate(value)
        probabilities = mx.softmax(self.router_logits, axis=-1, precise=True)
        routed = probabilities
        if self.expert_bias is not None:
            routed = routed + self.expert_bias
        self.top_indices = mx.stop_gradient(
            mx.argpartition(-routed, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        scores = mx.take_along_axis(routed, self.top_indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        expert_outputs = self.experts(value)
        selected = mx.take_along_axis(expert_outputs, self.top_indices[..., None], axis=-2)
        return mx.sum(selected * scores[..., None].astype(value.dtype), axis=-2)


class Lfm2MoeModel(Lfm2Model):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = []
        for index in range(config.num_hidden_layers):
            feed_forward = (
                LfmMLP(config.hidden_size, config.intermediate_size)
                if index < config.num_dense_layers
                else LfmSparseMoe(config)
            )
            self.layers.append(HybridDecoderLayer(config, index, feed_forward))
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class Lfm2MoeForCausalLM(nn.Module):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2MoeModel(config)
        if not config.tie_embedding:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> tuple[Any, ...]:
        return make_hybrid_caches(tuple(self.config.layer_types), self.config.conv_L_cache)

    def __call__(self, input_ids: Any, *, cache: tuple[Any, ...] | None = None) -> ModelOutput:
        hidden, cache, _ = self.model(input_ids, cache=cache)
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
                all_routers, self.config.num_experts, self.config.num_experts_per_tok
            )
        routers = all_routers if self.config.output_router_logits else None
        return ModelOutput(
            logits, hidden, cache=cache, router_logits=routers, router_aux_loss=aux_loss
        )
