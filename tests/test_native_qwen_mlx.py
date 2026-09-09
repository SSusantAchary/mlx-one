"""Opt-in tiny-MLX execution tests for native Qwen architectures."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.core.cache import make_kv_caches
from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.model import Qwen2ForCausalLM
from mlx_one.models.language.qwen2_moe.config import Qwen2MoeConfig
from mlx_one.models.language.qwen2_moe.model import Qwen2MoeForCausalLM
from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.language.qwen3.model import Qwen3ForCausalLM
from mlx_one.models.shared.transformer import causal_mask
from mlx_one.models.vision_language.qwen2_vl.config import Qwen2VLConfig
from mlx_one.models.vision_language.qwen2_vl.model import (
    Qwen2VLForConditionalGeneration,
    _replace_features,
    multimodal_position_ids,
)


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
                "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
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


def test_qwen2_shapes_hidden_states_and_cache_equivalence() -> None:
    config = qwen2_config()
    model = Qwen2ForCausalLM(config)
    tokens = mx.array([[1, 2, 3]])
    full = model(tokens, output_hidden_states=True)
    caches = make_kv_caches(config.num_hidden_layers)
    pieces = [model(tokens[:, index : index + 1], cache=caches).logits for index in range(3)]
    cached = mx.concatenate(pieces, axis=1)
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 3, config.vocab_size)
    assert full.last_hidden_state.shape == (1, 3, config.hidden_size)
    assert len(full.hidden_states) == config.num_hidden_layers + 1
    assert causal_mask(3).shape == (3, 3)
    assert causal_mask(1, offset=2) is None
    assert caches[0].offset == 3
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4


def test_qwen3_shapes_qk_norm_and_cache_equivalence() -> None:
    config = qwen3_config()
    model = Qwen3ForCausalLM(config)
    assert model.model.layers[0].self_attn.q_norm is not None
    assert model.model.layers[0].self_attn.k_norm is not None
    tokens = mx.array([[2, 4, 6]])
    full = model(tokens)
    caches = make_kv_caches(config.num_hidden_layers)
    cached = mx.concatenate(
        [model(tokens[:, index : index + 1], cache=caches).logits for index in range(3)],
        axis=1,
    )
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 3, config.vocab_size)
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4


def test_qwen2_moe_exposes_router_shapes_and_shared_expert() -> None:
    config = moe_config(output_router_logits=True)
    model = Qwen2MoeForCausalLM(config)
    output = model(mx.array([[1, 2, 3]]), output_hidden_states=True)
    mx.eval(output.logits, output.router_aux_loss)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert len(output.router_logits) == config.num_hidden_layers
    assert output.router_logits[0].shape == (1, 3, config.num_experts)
    assert output.router_aux_loss.shape == ()
    block = model.model.layers[0].mlp
    routed, router_logits, top_k_indices = block(mx.random.normal((1, 3, 16)))
    mx.eval(routed, router_logits, top_k_indices)
    assert routed.shape == (1, 3, config.hidden_size)
    assert top_k_indices.shape == (1, 3, config.num_experts_per_tok)
    assert block.experts(mx.random.normal((1, 3, 16))).shape == (
        1,
        3,
        config.num_experts,
        config.hidden_size,
    )
    assert block.shared_expert_gate.weight.shape == (1, config.hidden_size)
    assert Qwen2MoeForCausalLM(moe_config())(mx.array([[1]])).router_logits is None


def test_qwen2_vl_shapes_positions_and_feature_insertion() -> None:
    config = vl_config()
    model = Qwen2VLForConditionalGeneration(config)
    tokens = mx.array([[5, config.image_token_id, 6]])
    patch_dim = (
        config.vision_config.in_channels
        * config.vision_config.temporal_patch_size
        * config.vision_config.patch_size**2
    )
    patches = mx.random.normal((4, patch_dim))
    merged, vision_states = model.visual(patches, ((1, 2, 2),))
    assert merged.shape == (1, config.text_config.hidden_size)
    assert vision_states.shape == (4, config.vision_config.embed_dim)
    replaced = _replace_features(
        mx.zeros((1, 3, config.text_config.hidden_size)),
        tokens == config.image_token_id,
        merged,
        "image",
    )
    mx.eval(replaced)
    assert mx.max(mx.abs(replaced[0, 1] - merged[0])).item() == 0
    positions, _ = multimodal_position_ids(
        tokens,
        mm_token_type_ids=mx.array([[0, 1, 0]]),
        image_grid_thw=((1, 2, 2),),
        video_grid_thw=None,
        spatial_merge_size=2,
        attention_mask=None,
    )
    assert positions.shape == (3, 1, 3)
    output = model(
        tokens,
        pixel_values=patches,
        image_grid_thw=((1, 2, 2),),
    )
    mx.eval(output.logits, output.vision_hidden_states, output.position_ids)
    assert output.logits.shape == (1, 3, config.text_config.vocab_size)
    assert output.vision_hidden_states.shape == (4, config.vision_config.embed_dim)
    assert output.position_ids.shape == (3, 1, 3)
    assert output.rope_deltas.shape == (1, 1)
