"""Tiny native MLX execution tests for Qwen2.5-VL and Qwen3.5."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.models.vision_language.qwen2_5_vl.config import Qwen2_5_VLConfig
from mlx_one.models.vision_language.qwen2_5_vl.model import (
    Qwen2_5_VLForConditionalGeneration,
    _window_plan,
)
from mlx_one.models.vision_language.qwen2_5_vl.weights import (
    weight_contract as qwen2_5_vl_weights,
)
from mlx_one.models.vision_language.qwen3_5.config import Qwen3_5Config
from mlx_one.models.vision_language.qwen3_5.model import Qwen3_5ForConditionalGeneration
from mlx_one.models.vision_language.qwen3_5.weights import (
    weight_contract as qwen3_5_weights,
)


def qwen2_5_vl_config() -> Qwen2_5_VLConfig:
    return Qwen2_5_VLConfig.from_dict(
        {
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
    )


def qwen3_5_config() -> Qwen3_5Config:
    return Qwen3_5Config.from_dict(
        {
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
    )


def test_qwen2_5_vl_window_attention_merger_and_output_shapes() -> None:
    config = qwen2_5_vl_config()
    model = Qwen2_5_VLForConditionalGeneration(config)
    patch_dim = 3 * 1 * 2 * 2
    patches = mx.random.normal((16, patch_dim))
    indexes, lengths = _window_plan(((1, 4, 4),), config.vision_config)
    assert sorted(indexes) == list(range(4))
    assert lengths == (16,)
    tokens = mx.array(
        [
            [
                5,
                config.image_token_id,
                config.image_token_id,
                config.image_token_id,
                config.image_token_id,
                6,
            ]
        ]
    )
    output = model(tokens, pixel_values=patches, image_grid_thw=((1, 4, 4),))
    mx.eval(output.logits, output.vision_hidden_states)
    assert output.logits.shape == (1, 6, 64)
    assert output.vision_hidden_states.shape == (16, 8)
    assert output.position_ids.shape == (3, 1, 6)


def test_qwen3_5_hybrid_cache_vision_and_tied_logits() -> None:
    config = qwen3_5_config()
    model = Qwen3_5ForConditionalGeneration(config)
    text = mx.array([[1, 2, 3]])
    full = model(text)
    caches = model.make_cache()
    cached = mx.concatenate(
        [model(text[:, index : index + 1], cache=caches).logits for index in range(3)],
        axis=1,
    )
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 3, 64)
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4
    assert caches[0].conv_state.shape == (1, 3, 24)
    assert caches[0].recurrent_state.shape == (1, 2, 4, 4)
    patches = mx.random.normal((4, 12))
    multimodal = model(
        mx.array([[5, config.image_token_id, 6]]),
        pixel_values=patches,
        image_grid_thw=((1, 2, 2),),
    )
    mx.eval(multimodal.logits, multimodal.vision_hidden_states)
    assert multimodal.logits.shape == (1, 3, 64)
    assert multimodal.vision_hidden_states.shape == (4, 8)
    assert multimodal.position_ids.shape == (3, 1, 3)


@pytest.mark.parametrize(
    ("config", "model_type", "contract_factory"),
    [
        (qwen2_5_vl_config(), Qwen2_5_VLForConditionalGeneration, qwen2_5_vl_weights),
        (qwen3_5_config(), Qwen3_5ForConditionalGeneration, qwen3_5_weights),
    ],
)
def test_model_parameter_tree_exactly_matches_weight_contract(
    config: object, model_type: object, contract_factory: object
) -> None:
    model = model_type(config)
    parameters = dict(tree_flatten(model.parameters()))
    contract = contract_factory(config)
    assert set(parameters) == set(contract.expected)
    assert {name: tuple(value.shape) for name, value in parameters.items()} == contract.expected
