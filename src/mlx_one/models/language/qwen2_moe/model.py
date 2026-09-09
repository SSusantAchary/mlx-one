"""Native MLX implementation of Qwen2-MoE."""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.qwen2_moe.config import Qwen2MoeConfig
from mlx_one.models.shared.moe import SparseMoeBlock, router_auxiliary_loss
from mlx_one.models.shared.transformer import GroupedQueryAttention, causal_mask


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            rope_theta=config.rope_theta,
            qkv_bias=True,
            qk_norm=False,
            norm_eps=config.rms_norm_eps,
            rope_scaling_factor=config.rope_scaling_factor,
        )
        self.mlp = SparseMoeBlock(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden: Any,
        *,
        mask: Any | None,
        cache: Any | None,
        position_ids: Any | None,
    ) -> tuple[Any, Any, Any]:
        attended = self.self_attn(
            self.input_layernorm(hidden),
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        hidden = hidden + attended
        routed, router_logits, indices = self.mlp(self.post_attention_layernorm(hidden))
        return hidden + routed, router_logits, indices


class Qwen2MoeModel(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2MoeDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: Any,
        *,
        cache: tuple[Any, ...] | None = None,
        position_ids: Any | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[Any, tuple[Any, ...] | None, tuple[Any, ...] | None, tuple[Any, ...]]:
        hidden = self.embed_tokens(input_ids)
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache count must match decoder layer count")
        offset = 0 if cache is None else cache[0].offset
        mask = causal_mask(hidden.shape[1], offset)
        states = [hidden] if output_hidden_states else None
        router_logits = []
        for layer, layer_cache in zip(
            self.layers, cache or (None,) * len(self.layers), strict=True
        ):
            hidden, logits, _ = layer(
                hidden,
                mask=mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
            router_logits.append(logits)
            if states is not None:
                states.append(hidden)
        hidden = self.norm(hidden)
        if states is not None:
            states[-1] = hidden
        return (
            hidden,
            cache,
            tuple(states) if states is not None else None,
            tuple(router_logits),
        )


class Qwen2MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen2MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2MoeModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: Any, **kwargs: Any) -> ModelOutput:
        hidden, cache, states, routers = self.model(input_ids, **kwargs)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.config.tie_word_embeddings
            else self.lm_head(hidden)
        )
        exposed = routers if self.config.output_router_logits else None
        auxiliary = (
            router_auxiliary_loss(
                routers, self.config.num_experts, self.config.num_experts_per_tok
            )
            * self.config.router_aux_loss_coef
            if self.config.output_router_logits
            else None
        )
        return ModelOutput(
            logits,
            hidden,
            cache=cache,
            hidden_states=states,
            router_logits=exposed,
            router_aux_loss=auxiliary,
        )
