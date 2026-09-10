"""Weight mapping for sparse LFM2 models."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.weights import sanitize_weights as sanitize_dense
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig


def sanitize_weights(weights: dict[str, Any], config: Lfm2MoeConfig) -> dict[str, Any]:
    import mlx.core as mx

    sanitized = sanitize_dense(weights, config)
    replacements = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    for layer in range(config.num_dense_layers, config.num_hidden_layers):
        prefix = f"model.layers.{layer}.feed_forward"
        for source, target in replacements.items():
            names = [
                f"{prefix}.experts.{expert}.{source}.weight" for expert in range(config.num_experts)
            ]
            if names[0] in sanitized:
                sanitized[f"{prefix}.experts.{target}"] = mx.stack(
                    [sanitized.pop(name) for name in names]
                )
    return sanitized


def weight_contract(config: Lfm2MoeConfig) -> WeightContract:
    from mlx_one.models.language.lfm2.weights import weight_contract as dense_contract

    expected = dict(dense_contract(config).expected)
    dim = config.hidden_size
    for layer in range(config.num_dense_layers, config.num_hidden_layers):
        prefix = f"model.layers.{layer}.feed_forward"
        for name in ("w1.weight", "w3.weight", "w2.weight"):
            expected.pop(f"{prefix}.{name}")
        expected.update(
            {
                f"{prefix}.gate.weight": (config.num_experts, dim),
                f"{prefix}.experts.gate_proj": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    dim,
                ),
                f"{prefix}.experts.up_proj": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    dim,
                ),
                f"{prefix}.experts.down_proj": (
                    config.num_experts,
                    dim,
                    config.moe_intermediate_size,
                ),
            }
        )
        if config.use_expert_bias:
            expected[f"{prefix}.expert_bias"] = (config.num_experts,)
    return WeightContract(expected)
