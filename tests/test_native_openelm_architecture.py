"""Backend-free contracts for the native Apple OpenELM architecture."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.language.openelm.config import OpenELMConfig, make_divisible
from mlx_one.models.language.openelm.weights import sanitize_weights, weight_contract


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def openelm_config(**updates: object) -> OpenELMConfig:
    values = {
        "model_type": "openelm",
        "vocab_size": 64,
        "max_context_length": 32,
        "rope_max_length": 64,
        "num_transformer_layers": 3,
        "model_dim": 16,
        "head_dim": 4,
        "qkv_multipliers": [0.5, 1.0],
        "num_gqa_groups": 2,
        "ffn_multipliers": [0.5, 1.0],
        "ffn_dim_divisor": 4,
        "normalize_qk_projections": True,
        "share_input_output_layers": True,
    }
    values.update(updates)
    return OpenELMConfig.from_dict(values)


def shaped_contract(config: OpenELMConfig) -> dict[str, Shaped]:
    contract = weight_contract(config)
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


def test_openelm_registry_is_lazy_and_backend_free() -> None:
    assert "openelm" in registered_model_types()
    registration = get_registration("openelm")
    assert registration.modality == "text"
    assert registration.capabilities == {"forward", "cache"}
    assert registration.config_class() is OpenELMConfig


def test_openelm_expands_layerwise_dimensions_deterministically() -> None:
    config = openelm_config()
    assert make_divisible(9, 8) == 16
    assert config.qkv_multipliers == (0.5, 0.75, 1.0)
    assert config.ffn_multipliers == (0.5, 0.75, 1.0)
    assert config.num_query_heads == (2, 4, 4)
    assert config.num_kv_heads == (1, 2, 2)
    assert config.ffn_dims == (8, 12, 16)


def test_openelm_accepts_serialized_per_layer_heads_and_multipliers() -> None:
    config = openelm_config(
        qkv_multipliers=[0.5, 0.75, 1.0],
        ffn_multipliers=[1.0, 1.5, 2.0],
        num_query_heads=[2, 4, 4],
        num_kv_heads=[1, 2, 2],
    )
    assert config.num_query_heads == (2, 4, 4)
    assert config.num_kv_heads == (1, 2, 2)
    assert config.ffn_dims == (16, 24, 32)


@pytest.mark.parametrize(
    (
        "layers",
        "model_dim",
        "head_dim",
        "first_query_heads",
        "last_query_heads",
        "first_ffn_dim",
        "last_ffn_dim",
    ),
    [
        (16, 1280, 64, 12, 20, 768, 5120),
        (20, 1536, 64, 12, 24, 768, 6144),
        (28, 2048, 64, 16, 32, 1024, 8192),
        (36, 3072, 128, 12, 24, 1536, 12288),
    ],
)
def test_openelm_released_profiles_expand_to_expected_boundaries(
    layers: int,
    model_dim: int,
    head_dim: int,
    first_query_heads: int,
    last_query_heads: int,
    first_ffn_dim: int,
    last_ffn_dim: int,
) -> None:
    config = OpenELMConfig(
        num_transformer_layers=layers,
        model_dim=model_dim,
        head_dim=head_dim,
        qkv_multipliers=(0.5, 1.0),
        num_gqa_groups=4,
        ffn_multipliers=(0.5, 4.0),
        normalize_qk_projections=True,
        share_input_output_layers=True,
    )
    assert len(config.num_query_heads) == layers
    assert config.num_query_heads[0] == first_query_heads
    assert config.num_query_heads[-1] == last_query_heads
    assert config.num_kv_heads[0] == first_query_heads // 4
    assert config.num_kv_heads[-1] == last_query_heads // 4
    assert config.ffn_dims[0] == first_ffn_dim
    assert config.ffn_dims[-1] == last_ffn_dim


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "qwen2"}, "model_type"),
        ({"head_dim": 3}, "even"),
        ({"model_dim": 16.5}, "positive integer"),
        ({"num_gqa_groups": None}, "positive integer"),
        ({"max_context_length": 65}, "cannot exceed"),
        ({"ffn_multipliers": [1.0, 2.0, 3.0, 4.0]}, "one value per layer"),
        ({"num_query_heads": [2, 4]}, "one value per layer"),
        (
            {"num_query_heads": [2, 4, 4], "num_kv_heads": [1, 1, 2]},
            "num_gqa_groups",
        ),
        ({"activation_fn_name": "gelu"}, "activation"),
        ({"normalization_layer_name": "layer_norm"}, "normalization"),
    ],
)
def test_openelm_rejects_unsupported_or_inconsistent_configs(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        openelm_config(**updates)


def test_openelm_weight_contract_tracks_layerwise_shapes_and_tied_head() -> None:
    config = openelm_config()
    contract = weight_contract(config)
    contract.validate(shaped_contract(config))
    assert contract.expected["transformer.layers.0.attn.qkv_proj.weight"] == (
        16,
        16,
    )
    assert contract.expected["transformer.layers.2.attn.qkv_proj.weight"] == (
        32,
        16,
    )
    assert "lm_head.weight" not in contract.expected
    invalid = shaped_contract(config)
    invalid["unexpected.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def test_openelm_untied_non_glu_contract_and_sanitizer() -> None:
    config = openelm_config(
        share_input_output_layers=False,
        ffn_with_glu=False,
    )
    contract = weight_contract(config)
    assert contract.expected["lm_head.weight"] == (config.vocab_size, config.model_dim)
    assert contract.expected["transformer.layers.0.ffn.proj_1.weight"] == (8, 16)
    sanitized = sanitize_weights(
        {
            "transformer.causal_mask": Shaped((32, 32)),
            "transformer.layers.0.attn.pos_embedding.inv_freq": Shaped((2,)),
            "lm_head.weight": Shaped((64, 16)),
        },
        openelm_config(),
    )
    assert sanitized == {}
