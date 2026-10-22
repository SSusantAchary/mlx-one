"""Weight mapping for sparse LFM2 models."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.weights import sanitize_weights as sanitize_dense
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig


def sanitize_weights(weights: dict[str, Any], config: Lfm2MoeConfig) -> dict[str, Any]:
    import mlx.core as mx

    # Native MLX checkpoints already have stacked switch_mlp tensors; HF layouts
    # have separate experts. Normalize both to the same quantizable module tree.
    renamed = {}
    for name, value in weights.items():
        for index in range(config.num_dense_layers):
            prefix = f"model.layers.{index}.feed_forward."
            if name.startswith(prefix):
                for source, target in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
                    name = name.replace(prefix + source + ".", prefix + target + ".")
        if name.endswith(".conv.conv.bias"):
            name = name.removesuffix(".bias") + "_bias"
        if config.tie_embedding and name.startswith("lm_head."):
            continue
        renamed[name] = value
    sanitized = sanitize_dense(renamed, config)
    replacements = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    for layer in range(config.num_dense_layers, config.num_hidden_layers):
        prefix = f"model.layers.{layer}.feed_forward"
        for source, target in replacements.items():
            legacy = f"{prefix}.experts.{target}"
            if legacy in sanitized:
                sanitized[f"{prefix}.switch_mlp.{target}.weight"] = sanitized.pop(legacy)
            for suffix in ("weight", "scales", "biases"):
                for spelling in (source, target):
                    names = [
                        f"{prefix}.experts.{expert}.{spelling}.{suffix}"
                        for expert in range(config.num_experts)
                    ]
                    if any(name in sanitized for name in names):
                        if not all(name in sanitized for name in names):
                            raise ValueError(f"incomplete expert tensors for {prefix}.{target}")
                        sanitized[f"{prefix}.switch_mlp.{target}.{suffix}"] = mx.stack(
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
                f"{prefix}.switch_mlp.gate_proj.weight": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    dim,
                ),
                f"{prefix}.switch_mlp.up_proj.weight": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    dim,
                ),
                f"{prefix}.switch_mlp.down_proj.weight": (
                    config.num_experts,
                    dim,
                    config.moe_intermediate_size,
                ),
            }
        )
        if config.use_expert_bias:
            expected[f"{prefix}.expert_bias"] = (config.num_experts,)
    return WeightContract(expected)
