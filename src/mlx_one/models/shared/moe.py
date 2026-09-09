"""Reusable sparse mixture-of-experts layers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class StackedExperts(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int) -> None:
        super().__init__()
        scale = hidden_size**-0.5
        self.gate_proj = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, intermediate_size, hidden_size)
        )
        self.up_proj = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, intermediate_size, hidden_size)
        )
        self.down_proj = mx.random.uniform(
            low=-scale, high=scale, shape=(num_experts, hidden_size, intermediate_size)
        )

    def __call__(self, value: Any) -> Any:
        gate = mx.einsum("...d,ehd->...eh", value, self.gate_proj)
        up = mx.einsum("...d,ehd->...eh", value, self.up_proj)
        activated = (gate * mx.sigmoid(gate)) * up
        return mx.einsum("...eh,edh->...ed", activated, self.down_proj)


class SparseMoeBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        from mlx_one.models.shared.transformer import SwiGLU

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = StackedExperts(
            config.hidden_size, config.moe_intermediate_size, config.num_experts
        )
        self.shared_expert = SwiGLU(
            config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def __call__(self, value: Any) -> tuple[Any, Any, Any]:
        router_logits = self.gate(value)
        probabilities = mx.softmax(router_logits, axis=-1, precise=True)
        indices = mx.stop_gradient(
            mx.argpartition(-probabilities, kth=self.top_k - 1, axis=-1)[..., : self.top_k]
        )
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        expert_outputs = self.experts(value)
        selected = mx.take_along_axis(expert_outputs, indices[..., None], axis=-2)
        sparse_output = mx.sum(selected * scores[..., None], axis=-2)
        shared_output = mx.sigmoid(self.shared_expert_gate(value)) * self.shared_expert(value)
        return sparse_output + shared_output, router_logits, indices


def router_auxiliary_loss(router_logits: tuple[Any, ...], num_experts: int, top_k: int) -> Any:
    if not router_logits:
        return mx.array(0.0)
    losses = []
    for logits in router_logits:
        probabilities = mx.softmax(logits.reshape(-1, num_experts), axis=-1, precise=True)
        indices = mx.argpartition(-probabilities, kth=top_k - 1, axis=-1)[:, :top_k]
        assignment = mx.sum(
            indices[..., None] == mx.arange(num_experts)[None, None, :], axis=1
        ) / top_k
        losses.append(
            num_experts
            * mx.sum(mx.mean(probabilities, axis=0) * mx.mean(assignment, axis=0))
        )
    return mx.mean(mx.stack(losses))
