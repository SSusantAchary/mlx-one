"""Weight conversion contract for Qwen2-MoE."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.qwen2_moe.config import Qwen2MoeConfig


def sanitize_weights(weights: dict[str, Any], config: Qwen2MoeConfig) -> dict[str, Any]:
    sanitized = {
        name: value
        for name, value in weights.items()
        if not name.endswith("self_attn.rotary_emb.inv_freq")
    }
    if config.tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            legacy = f"{prefix}.experts.{projection}"
            if legacy in sanitized:
                if legacy + ".weight" in sanitized:
                    raise ValueError("duplicate stacked expert weight layouts")
                sanitized[legacy + ".weight"] = sanitized.pop(legacy)
            for suffix in ("weight", "scales", "biases"):
                names = [
                    f"{prefix}.experts.{expert}.{projection}.{suffix}"
                    for expert in range(config.num_experts)
                ]
                if not any(name in sanitized for name in names):
                    continue
                if not all(name in sanitized for name in names):
                    raise ValueError(f"incomplete per-expert {projection}.{suffix} tensors")
                target = f"{legacy}.{suffix}"
                if target in sanitized:
                    raise ValueError("duplicate stacked and per-expert tensor layouts")
                values = [sanitized.pop(name) for name in names]
                if type(values[0]).__module__.startswith("mlx"):
                    import mlx.core as mx

                    stacked = mx.stack(values)
                else:
                    import numpy as np

                    stacked = np.stack(values)
                sanitized[target] = stacked
    return sanitized


def weight_contract(config: Qwen2MoeConfig) -> WeightContract:
    expected: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, config.hidden_size),
        "model.norm.weight": (config.hidden_size,),
    }
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (config.vocab_size, config.hidden_size)
    q_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        mlp = f"{prefix}.mlp"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (q_size, config.hidden_size),
                f"{prefix}.self_attn.q_proj.bias": (q_size,),
                f"{prefix}.self_attn.k_proj.weight": (kv_size, config.hidden_size),
                f"{prefix}.self_attn.k_proj.bias": (kv_size,),
                f"{prefix}.self_attn.v_proj.weight": (kv_size, config.hidden_size),
                f"{prefix}.self_attn.v_proj.bias": (kv_size,),
                f"{prefix}.self_attn.o_proj.weight": (config.hidden_size, q_size),
                f"{mlp}.gate.weight": (config.num_experts, config.hidden_size),
                f"{mlp}.experts.gate_proj.weight": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    config.hidden_size,
                ),
                f"{mlp}.experts.up_proj.weight": (
                    config.num_experts,
                    config.moe_intermediate_size,
                    config.hidden_size,
                ),
                f"{mlp}.experts.down_proj.weight": (
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                ),
                f"{mlp}.shared_expert.gate_proj.weight": (
                    config.shared_expert_intermediate_size,
                    config.hidden_size,
                ),
                f"{mlp}.shared_expert.up_proj.weight": (
                    config.shared_expert_intermediate_size,
                    config.hidden_size,
                ),
                f"{mlp}.shared_expert.down_proj.weight": (
                    config.hidden_size,
                    config.shared_expert_intermediate_size,
                ),
                f"{mlp}.shared_expert_gate.weight": (1, config.hidden_size),
            }
        )
    return WeightContract(expected)
