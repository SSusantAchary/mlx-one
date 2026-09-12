"""Strict Hugging Face safetensors contract for GPT-2."""

from __future__ import annotations

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.gpt2.config import GPT2Config


def sanitize_weights(weights: dict[str, Any], config: GPT2Config) -> dict[str, Any]:
    ignored_suffixes = (".attn.bias", ".attn.masked_bias")
    result: dict[str, Any] = {}
    for original, value in weights.items():
        if original.endswith(ignored_suffixes) or original == "lm_head.weight":
            continue
        name = original
        if not name.startswith("transformer.") and name.startswith(
            ("wte.", "wpe.", "ln_f.", "h.")
        ):
            name = f"transformer.{name}"
        if name in result:
            raise ValueError(f"duplicate GPT-2 tensor after normalization: {name}")
        result[name] = value
    for index in range(config.n_layer):
        prefix = f"transformer.h.{index}"
        for suffix in (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ):
            name = f"{prefix}.{suffix}"
            if name in result:
                result[name] = result[name].transpose(1, 0)
    return result


def weight_contract(config: GPT2Config) -> WeightContract:
    expected: dict[str, tuple[int, ...]] = {
        "transformer.wte.weight": (config.vocab_size, config.n_embd),
        "transformer.wpe.weight": (config.n_positions, config.n_embd),
        "transformer.ln_f.weight": (config.n_embd,),
        "transformer.ln_f.bias": (config.n_embd,),
    }
    for index in range(config.n_layer):
        prefix = f"transformer.h.{index}"
        expected.update(
            {
                f"{prefix}.ln_1.weight": (config.n_embd,),
                f"{prefix}.ln_1.bias": (config.n_embd,),
                f"{prefix}.attn.c_attn.weight": (3 * config.n_embd, config.n_embd),
                f"{prefix}.attn.c_attn.bias": (3 * config.n_embd,),
                f"{prefix}.attn.c_proj.weight": (config.n_embd, config.n_embd),
                f"{prefix}.attn.c_proj.bias": (config.n_embd,),
                f"{prefix}.ln_2.weight": (config.n_embd,),
                f"{prefix}.ln_2.bias": (config.n_embd,),
                f"{prefix}.mlp.c_fc.weight": (config.inner_dim, config.n_embd),
                f"{prefix}.mlp.c_fc.bias": (config.inner_dim,),
                f"{prefix}.mlp.c_proj.weight": (config.n_embd, config.inner_dim),
                f"{prefix}.mlp.c_proj.bias": (config.n_embd,),
            }
        )
    return WeightContract(expected)
