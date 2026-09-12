"""Backend-free contracts for Qwen2.5-VL and Qwen3.5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
from mlx_one.models.vision_language.qwen2_5_vl.weights import (
    sanitize_weights as sanitize_qwen2_5_vl,
)
from mlx_one.models.vision_language.qwen2_5_vl.weights import (
    weight_contract as qwen2_5_vl_weights,
)
from mlx_one.models.vision_language.qwen3_5.config import Qwen3_5Config
from mlx_one.models.vision_language.qwen3_5.weights import (
    sanitize_weights as sanitize_qwen3_5,
)
from mlx_one.models.vision_language.qwen3_5.weights import (
    weight_contract as qwen3_5_weights,
)


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def qwen2_5_vl_config(**updates: object) -> Qwen2_5_VLConfig:
    values = {
        "model_type": "qwen2_5_vl",
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
            "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
        },
        "vision_config": {
            "model_type": "qwen2_5_vl_vision",
            "depth": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 8,
            "out_hidden_size": 16,
            "fullatt_block_indexes": [1],
        },
    }
    values.update(updates)
    return Qwen2_5_VLConfig.from_dict(values)


def qwen3_5_config(**updates: object) -> Qwen3_5Config:
    values = {
        "model_type": "qwen3_5",
        "image_token_id": 60,
        "video_token_id": 61,
        "vision_start_token_id": 62,
        "vision_end_token_id": 63,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "qwen3_5_text",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
            "tie_word_embeddings": True,
        },
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "depth": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
        },
    }
    values.update(updates)
    return Qwen3_5Config.from_dict(values)


def _shaped(contract: object) -> dict[str, Shaped]:
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


def test_registry_contains_both_families_without_eager_backend_import() -> None:
    assert {"qwen2_5_vl", "qwen3_5"} <= set(registered_model_types())
    assert get_registration("qwen2_5_vl").modality == "vision-language"
    assert "linear-attention" in get_registration("qwen3_5").capabilities


def test_qwen2_5_vl_nested_configuration_and_failures() -> None:
    config = qwen2_5_vl_config()
    assert config.vision_config.fullatt_block_indexes == (1,)
    assert config.text_config.mrope_section == (1, 1, 2)
    with pytest.raises(ConfigError, match="out_hidden_size"):
        qwen2_5_vl_config(
            vision_config={
                "model_type": "qwen2_5_vl_vision",
                "out_hidden_size": 32,
            }
        )
    with pytest.raises(ConfigError, match="window_size"):
        qwen2_5_vl_config(
            vision_config={
                "model_type": "qwen2_5_vl_vision",
                "window_size": 15,
            }
        )


def test_qwen3_5_hybrid_configuration_and_release_profiles() -> None:
    config = qwen3_5_config()
    assert config.text_config.layer_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    assert config.text_config.rotary_dim == 4
    for hidden, intermediate in ((1024, 3584), (2048, 6144)):
        profile = dict(config.text_config.__dict__)
        profile.update(
            hidden_size=hidden,
            intermediate_size=intermediate,
            num_hidden_layers=24,
            layer_types=None,
            extra={},
        )
        assert config.text_config.from_dict(profile).num_hidden_layers == 24
    with pytest.raises(ConfigError, match="one supported mixer"):
        qwen3_5_config(
            text_config={
                "model_type": "qwen3_5_text",
                "layer_types": ["full_attention"],
            }
        )


@pytest.mark.parametrize(
    ("config", "factory"),
    [(qwen2_5_vl_config(), qwen2_5_vl_weights), (qwen3_5_config(), qwen3_5_weights)],
)
def test_weight_contracts_accept_only_exact_synthetic_tensors(
    config: object, factory: object
) -> None:
    contract = factory(config)
    contract.validate(_shaped(contract))
    invalid = _shaped(contract)
    invalid["unexpected.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def test_qwen2_5_vl_sanitizes_nested_names_patch_conv_and_tied_head() -> None:
    config = qwen2_5_vl_config(tie_word_embeddings=True)
    patch = np.zeros((8, 3, 1, 2, 2))
    result = sanitize_qwen2_5_vl(
        {
            "model.visual.patch_embed.proj.weight": patch,
            "model.language_model.embed_tokens.weight": np.zeros((64, 16)),
            "lm_head.weight": np.zeros((64, 16)),
            "model.language_model.rotary_emb.inv_freq": np.zeros((4,)),
        },
        config,
    )
    assert result["visual.patch_embed.proj.weight"].shape == (8, 12)
    assert "language_model.embed_tokens.weight" in result
    assert "lm_head.weight" not in result


def test_qwen3_5_sanitizes_recurrent_conv_patch_and_ignored_buffers() -> None:
    config = qwen3_5_config()
    conv_dim = 24
    result = sanitize_qwen3_5(
        {
            "model.language_model.layers.0.linear_attn.conv1d.weight": np.zeros((conv_dim, 1, 4)),
            "model.visual.patch_embed.proj.weight": np.zeros((8, 3, 1, 2, 2)),
            "model.language_model.rotary_emb.inv_freq": np.zeros((2,)),
            "lm_head.weight": np.zeros((64, 16)),
        },
        config,
    )
    assert result["language_model.layers.0.linear_attn.conv1d.weight"].shape == (
        conv_dim,
        4,
    )
    assert result["visual.patch_embed.proj.weight"].shape == (8, 12)
    assert "lm_head.weight" not in result
