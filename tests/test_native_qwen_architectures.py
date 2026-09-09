"""Backend-free contracts for the native Qwen architecture families."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.weights import weight_contract as qwen2_weights
from mlx_one.models.language.qwen2_moe.config import Qwen2MoeConfig
from mlx_one.models.language.qwen2_moe.weights import (
    sanitize_weights as sanitize_moe_weights,
)
from mlx_one.models.language.qwen2_moe.weights import weight_contract as moe_weights
from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.language.qwen3.weights import weight_contract as qwen3_weights
from mlx_one.models.vision_language.qwen2_vl.config import Qwen2VLConfig
from mlx_one.models.vision_language.qwen2_vl.weights import (
    sanitize_weights as sanitize_vl_weights,
)
from mlx_one.models.vision_language.qwen2_vl.weights import (
    weight_contract as vl_weights,
)


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def qwen2_config(**updates: object) -> Qwen2Config:
    values = {
        "model_type": "qwen2",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
    }
    values.update(updates)
    return Qwen2Config.from_dict(values)


def qwen3_config(**updates: object) -> Qwen3Config:
    values = {
        "model_type": "qwen3",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 64,
    }
    values.update(updates)
    return Qwen3Config.from_dict(values)


def moe_config(**updates: object) -> Qwen2MoeConfig:
    values = {
        "model_type": "qwen2_moe",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 12,
        "shared_expert_intermediate_size": 24,
    }
    values.update(updates)
    return Qwen2MoeConfig.from_dict(values)


def vl_config() -> Qwen2VLConfig:
    return Qwen2VLConfig.from_dict(
        {
            "model_type": "qwen2_vl",
            "image_token_id": 60,
            "video_token_id": 61,
            "vision_start_token_id": 62,
            "vision_end_token_id": 63,
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [1, 1, 2],
                },
            },
            "vision_config": {
                "depth": 2,
                "embed_dim": 8,
                "hidden_size": 16,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "mlp_ratio": 2,
            },
        }
    )


def shaped_contract(contract: object) -> dict[str, Shaped]:
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


def test_registry_is_lazy_and_contains_all_qwen_families() -> None:
    assert registered_model_types() == (
        "openelm",
        "qwen2",
        "qwen2_moe",
        "qwen2_vl",
        "qwen3",
    )
    assert get_registration("qwen2_vl").modality == "vision-language"
    assert get_registration("qwen2_moe").capabilities >= {"forward", "router-outputs"}
    assert all(
        "training" not in get_registration(model_type).capabilities
        for model_type in registered_model_types()
    )


def test_dense_configs_validate_architecture_invariants() -> None:
    assert qwen2_config().head_dim == 4
    assert qwen3_config().head_dim == 4
    with pytest.raises(ConfigError, match="divisible"):
        qwen2_config(num_attention_heads=3)
    with pytest.raises(ConfigError, match="model_type"):
        qwen3_config(model_type="qwen2")
    with pytest.raises(ConfigError, match="unsupported RoPE"):
        qwen2_config(rope_scaling={"type": "unknown"})
    scaled = qwen2_config(rope_parameters={"rope_type": "linear", "factor": 2})
    assert scaled.rope_scaling_factor == 2
    with pytest.raises(ConfigError, match="not supported"):
        qwen3_config(
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 1]}
        )


def test_moe_config_validates_routing_and_explicit_limitations() -> None:
    assert moe_config().num_experts_per_tok == 2
    with pytest.raises(ConfigError, match="cannot exceed"):
        moe_config(num_experts_per_tok=5)
    with pytest.raises(ConfigError, match="sliding-window"):
        moe_config(use_sliding_window=True)


def test_moe_tied_head_cleanup_matches_weight_contract() -> None:
    config = moe_config(tie_word_embeddings=True)
    assert "lm_head.weight" not in moe_weights(config).expected
    assert sanitize_moe_weights(
        {"lm_head.weight": Shaped((64, 16))}, config
    ) == {}


def test_moe_sanitizer_stacks_hugging_face_expert_weights() -> None:
    config = moe_config(num_hidden_layers=1, num_experts=2)
    weights = {}
    shapes = {
        "gate_proj": (config.moe_intermediate_size, config.hidden_size),
        "up_proj": (config.moe_intermediate_size, config.hidden_size),
        "down_proj": (config.hidden_size, config.moe_intermediate_size),
    }
    for projection, shape in shapes.items():
        for expert in range(config.num_experts):
            weights[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"] = (
                np.full(shape, expert)
            )
    sanitized = sanitize_moe_weights(weights, config)
    for projection, shape in shapes.items():
        stacked = sanitized[f"model.layers.0.mlp.experts.{projection}"]
        assert stacked.shape == (config.num_experts, *shape)
        assert np.all(stacked[1] == 1)


def test_vl_config_validates_nested_dimensions_tokens_and_mrope() -> None:
    config = vl_config()
    assert config.text_config.mrope_section == (1, 1, 2)
    assert config.vision_config.spatial_merge_size == 2
    payload = {
        "model_type": "qwen2_vl",
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        },
        "vision_config": {"hidden_size": 32},
    }
    with pytest.raises(ConfigError, match="vision hidden_size"):
        Qwen2VLConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("config", "factory"),
    [(qwen2_config(), qwen2_weights), (qwen3_config(), qwen3_weights), (moe_config(), moe_weights)],
)
def test_language_weight_contracts_accept_exact_shapes(config: object, factory: object) -> None:
    contract = factory(config)
    contract.validate(shaped_contract(contract))
    invalid = shaped_contract(contract)
    invalid["unexpected.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def test_vl_weight_contract_and_patch_sanitizer() -> None:
    config = vl_config()
    contract = vl_weights(config)
    contract.validate(shaped_contract(contract))
    vision = config.vision_config
    convolution = np.zeros(
        (
            vision.embed_dim,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        )
    )
    sanitized = sanitize_vl_weights(
        {
            "visual.patch_embed.proj.weight": convolution,
            "model.language_model.embed_tokens.weight": Shaped((64, 16)),
            "visual.merger.mlp.0.weight": Shaped((32, 32)),
        },
        config,
    )
    assert sanitized["visual.patch_embed.proj.weight"].shape == (
        vision.embed_dim,
        vision.in_channels
        * vision.temporal_patch_size
        * vision.patch_size
        * vision.patch_size,
    )
    assert "language_model.embed_tokens.weight" in sanitized
    assert "visual.merger.mlp_0.weight" in sanitized
