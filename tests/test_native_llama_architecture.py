"""Backend-free contracts for native Llama-family causal models."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.language.llama.config import LlamaConfig
from mlx_one.models.language.llama.weights import sanitize_weights, weight_contract


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def minicpm_config(size: str = "1b", **updates: object) -> LlamaConfig:
    values = {
        "model_type": "llama",
        "hidden_size": 1536 if size == "1b" else 2048,
        "num_hidden_layers": 24 if size == "1b" else 42,
        "intermediate_size": 4608 if size == "1b" else 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 130560,
        "max_position_embeddings": 131072,
        "rope_theta": 5_000_000,
        "tie_word_embeddings": False,
        "eos_token_id": [1, 130073],
    }
    values.update(updates)
    return LlamaConfig.from_dict(values)


def shaped(contract: object) -> dict[str, Shaped]:
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


def test_llama_registry_entry_is_lazy_and_text_generative() -> None:
    assert "llama" in registered_model_types()
    registration = get_registration("llama")
    assert registration.config_class() is LlamaConfig
    assert registration.modality == "text"
    assert registration.capabilities >= {"forward", "cache", "generate", "sample", "stream"}
    assert registration.loader_path == "mlx_one.text.loading:load_text_model"
    assert registration.supports("text-generation")


@pytest.mark.parametrize(
    ("size", "hidden", "layers", "intermediate"),
    [("1b", 1536, 24, 4608), ("2b", 2048, 42, 6144)],
)
def test_minicpm5_configs_preserve_explicit_attention_width(
    size: str, hidden: int, layers: int, intermediate: int
) -> None:
    config = minicpm_config(size)
    assert (config.hidden_size, config.num_hidden_layers, config.intermediate_size) == (
        hidden,
        layers,
        intermediate,
    )
    assert config.head_dim == 128
    assert config.eos_token_id == (1, 130073)
    assert config.num_attention_heads * config.head_dim == 2048


def test_llama_weight_contract_uses_explicit_head_dimension() -> None:
    config = minicpm_config("1b")
    contract = weight_contract(config)
    assert contract.expected["model.layers.0.self_attn.q_proj.weight"] == (2048, 1536)
    assert contract.expected["model.layers.0.self_attn.k_proj.weight"] == (256, 1536)
    assert contract.expected["model.layers.0.self_attn.o_proj.weight"] == (1536, 2048)
    contract.validate(shaped(contract))
    invalid = shaped(contract)
    invalid["unexpected.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def test_llama_sanitizer_removes_only_unused_or_tied_weights() -> None:
    config = minicpm_config("1b", tie_word_embeddings=True)
    cleaned = sanitize_weights(
        {
            "model.layers.0.self_attn.rotary_emb.inv_freq": Shaped((64,)),
            "model.embed_tokens.weight": Shaped((130560, 1536)),
            "lm_head.weight": Shaped((130560, 1536)),
        },
        config,
    )
    assert set(cleaned) == {"model.embed_tokens.weight"}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "qwen3"}, "model_type"),
        ({"num_key_value_heads": 3}, "divisible"),
        ({"head_dim": 127}, "even"),
        ({"hidden_act": "gelu"}, "SwiGLU"),
        ({"attention_bias": True}, "attention bias"),
        ({"mlp_bias": True}, "MLP bias"),
        ({"rope_traditional": True}, "traditional RoPE"),
        ({"rope_scaling": {"type": "dynamic", "factor": 2}}, "unsupported RoPE"),
        ({"eos_token_id": []}, "eos_token_id"),
    ],
)
def test_llama_rejects_unsupported_or_invalid_configs(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        minicpm_config("1b", **updates)


def test_llama_derives_standard_head_and_kv_dimensions() -> None:
    config = LlamaConfig.from_dict(
        {
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "vocab_size": 64,
        }
    )
    assert config.head_dim == 4
    assert config.num_key_value_heads == 4
