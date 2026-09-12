"""Weight contract for dense Qwen3."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.qwen3.config import Qwen3Config


def sanitize_weights(weights: dict[str, Any], config: Qwen3Config) -> dict[str, Any]:
    sanitized = {
        name: value
        for name, value in weights.items()
        if not name.endswith("rotary_emb.inv_freq")
    }
    if config.tie_word_embeddings:
        sanitized.pop("lm_head.weight", None)
    return sanitized


def weight_contract(config: Qwen3Config) -> WeightContract:
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
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                f"{prefix}.self_attn.q_proj.weight": (q_size, config.hidden_size),
                f"{prefix}.self_attn.k_proj.weight": (kv_size, config.hidden_size),
                f"{prefix}.self_attn.v_proj.weight": (kv_size, config.hidden_size),
                f"{prefix}.self_attn.o_proj.weight": (config.hidden_size, q_size),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.up_proj.weight": (
                    config.intermediate_size,
                    config.hidden_size,
                ),
                f"{prefix}.mlp.down_proj.weight": (
                    config.hidden_size,
                    config.intermediate_size,
                ),
            }
        )
    return WeightContract(expected)
