"""Backend-free contracts for native Liquid Foundation Models."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.audio.lfm2_audio.config import Lfm2AudioConfig
from mlx_one.models.audio.lfm2_audio.weights import weight_contract as audio_weight_contract
from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
from mlx_one.models.embeddings.lfm2_colbert.weights import (
    sanitize_weights as sanitize_colbert_weights,
)
from mlx_one.models.embeddings.lfm2_colbert.weights import (
    weight_contract as colbert_weight_contract,
)
from mlx_one.models.language.lfm2.config import Lfm2Config
from mlx_one.models.language.lfm2.weights import sanitize_weights, weight_contract
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
from mlx_one.models.language.lfm2_moe.weights import weight_contract as moe_weight_contract
from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig
from mlx_one.models.vision_language.lfm2_vl.weights import weight_contract as vl_weight_contract


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def dense_config(**updates: object) -> Lfm2Config:
    values = {
        "model_type": "lfm2",
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "conv_L_cache": 3,
        "intermediate_size": 24,
        "block_multiple_of": 4,
        "layer_types": ["conv", "full_attention", "conv"],
    }
    values.update(updates)
    return Lfm2Config.from_dict(values)


def moe_config(**updates: object) -> Lfm2MoeConfig:
    values = dict(dense_config().__dict__)
    values.update(
        {
            "model_type": "lfm2_moe",
            "num_dense_layers": 1,
            "moe_intermediate_size": 12,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
    )
    values.update(updates)
    return Lfm2MoeConfig.from_dict(values)


def shaped(contract: object) -> dict[str, Shaped]:
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


def test_lfm_registry_entries_are_lazy_and_backend_free() -> None:
    assert {"lfm2", "lfm2_moe", "lfm2_vl", "lfm2_colbert", "lfm2_audio"} <= set(
        registered_model_types()
    )
    dense = get_registration("lfm2")
    moe = get_registration("lfm2_moe")
    assert dense.config_class() is Lfm2Config
    assert moe.config_class() is Lfm2MoeConfig
    assert dense.capabilities == {"forward", "cache"}
    assert "router-outputs" in moe.capabilities
    assert get_registration("lfm2_vl").config_class() is Lfm2VLConfig
    assert get_registration("lfm2_colbert").config_class() is Lfm2ColBERTConfig
    assert get_registration("lfm2_audio").config_class() is Lfm2AudioConfig


def test_lfm_dense_derives_layer_layout_and_ffn_width() -> None:
    config = dense_config(layer_types=None, full_attn_idxs=[1])
    assert config.layer_types == ("conv", "full_attention", "conv")
    assert config.full_attn_idxs == (1,)
    assert config.adjusted_intermediate_size == 16


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "llama"}, "model_type"),
        ({"hidden_size": 15}, "divisible"),
        ({"num_attention_heads": 3}, "num_attention_heads"),
        ({"layer_types": ["conv"]}, "one entry"),
        ({"layer_types": ["conv", "conv", "conv"]}, "attention layer"),
        ({"layer_types": ["conv", "bad", "conv"]}, "unsupported"),
        ({"block_use_swiglu": False}, "SwiGLU"),
        ({"rope_parameters": {"rope_type": "linear"}}, "unsupported"),
    ],
)
def test_lfm_dense_rejects_invalid_configs(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        dense_config(**updates)


def test_lfm_dense_weight_contract_and_conv_sanitizer() -> None:
    config = dense_config()
    contract = weight_contract(config)
    contract.validate(shaped(contract))
    assert contract.expected["model.layers.0.conv.conv"] == (16, 3)
    cleaned = sanitize_weights(
        {
            "model.layers.0.conv.conv.weight": Shaped((16, 3)),
            "model.layers.1.self_attn.rotary_emb.inv_freq": Shaped((2,)),
            "lm_head.weight": Shaped((64, 16)),
        },
        config,
    )
    assert set(cleaned) == {"model.layers.0.conv.conv"}
    assert cleaned["model.layers.0.conv.conv"].shape == (16, 3)
    invalid = shaped(contract)
    invalid["unknown.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def test_lfm_moe_config_contract_and_expert_stacking() -> None:
    config = moe_config()
    contract = moe_weight_contract(config)
    contract.validate(shaped(contract))
    assert contract.expected["model.layers.1.feed_forward.gate.weight"] == (4, 16)
    source: dict[str, Shaped] = {}
    for expert in range(config.num_experts):
        source[f"model.layers.1.feed_forward.experts.{expert}.w1.weight"] = Shaped((12, 16))
        source[f"model.layers.1.feed_forward.experts.{expert}.w3.weight"] = Shaped((12, 16))
        source[f"model.layers.1.feed_forward.experts.{expert}.w2.weight"] = Shaped((16, 12))
    # Shape-only values deliberately cannot be stacked; real synthetic tensors are
    # exercised in the opt-in MLX suite.
    assert len(source) == 12


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "llama"}, "model_type"),
        ({"num_experts": 0}, "positive"),
        ({"num_experts_per_tok": 5}, "cannot exceed"),
        ({"num_dense_layers": 4}, "decoder layer"),
        ({"router_aux_loss_coef": -0.1}, "negative"),
    ],
)
def test_lfm_moe_rejects_invalid_configs(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        moe_config(**updates)


def test_lfm_vl_nested_config_and_validation() -> None:
    config = Lfm2VLConfig.from_dict(
        {
            "model_type": "lfm2-vl",
            "image_token_id": 63,
            "text_config": dense_config().__dict__,
            "vision_config": {
                "hidden_size": 12,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_channels": 3,
                "patch_size": 2,
                "image_size": 4,
                "downsample_factor": 2,
                "projection_dim": 16,
            },
        }
    )
    assert config.model_type == "lfm2_vl"
    assert config.vision_config.projection_dim == config.text_config.hidden_size
    contract = vl_weight_contract(config)
    contract.validate(shaped(contract))
    assert contract.expected["vision_model.projector.weight"] == (16, 48)
    with pytest.raises(ConfigError, match="projection_dim"):
        Lfm2VLConfig.from_dict(
            {
                "model_type": "lfm2_vl",
                "image_token_id": 63,
                "text_config": dense_config().__dict__,
                "vision_config": {"projection_dim": 17},
            }
        )


def test_lfm_colbert_profile_and_audio_nested_contracts() -> None:
    colbert = Lfm2ColBERTConfig.from_dict({**dense_config().__dict__, "model_type": "lfm2_colbert"})
    assert colbert.projection_dim == 128
    assert colbert.query_max_length == 32
    assert colbert.document_max_length == 512
    contract = colbert_weight_contract(colbert)
    contract.validate(shaped(contract))
    cleaned = sanitize_colbert_weights(
        {"1_Dense.linear.weight": Shaped((128, 16))}, colbert
    )
    assert set(cleaned) == {"projection.weight"}

    audio = Lfm2AudioConfig.from_dict(
        {
            "model_type": "lfm2_audio",
            "text_config": dense_config().__dict__,
            "audio_config": {
                "input_dim": 8,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "conv_kernel_size": 3,
                "subsampling_factor": 2,
            },
            "preprocessor_config": {
                "sampling_rate": 16000,
                "n_fft": 16,
                "window_length": 8,
                "hop_length": 4,
                "num_mel_bins": 8,
            },
            "depthformer_config": {
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "intermediate_size": 24,
            },
            "detokenizer_config": {
                "codebook_size": 17,
                "num_codebooks": 2,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 24,
                "upsample_factor": 2,
                "n_fft": 16,
                "hop_length": 4,
                "sliding_window": 4,
            },
            "audio_vocab_size": 17,
            "audio_eos_token_id": 16,
            "num_codebooks": 2,
        }
    )
    assert audio.audio_config.input_dim == audio.preprocessor_config.num_mel_bins
    assert audio.detokenizer_config.num_codebooks == audio.num_codebooks
    contract = audio_weight_contract(audio)
    contract.validate(shaped(contract))
    assert contract.expected["detokenizer.output.weight"] == (18, 16)
