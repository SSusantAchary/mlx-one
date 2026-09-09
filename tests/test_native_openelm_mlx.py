"""Opt-in tiny-MLX execution tests for native Apple OpenELM."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.core.cache import make_kv_caches
from mlx_one.models.language.openelm.config import OpenELMConfig
from mlx_one.models.language.openelm.model import OpenELMForCausalLM


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


def test_openelm_layerwise_shapes_hidden_states_and_cache_equivalence() -> None:
    config = openelm_config()
    model = OpenELMForCausalLM(config)
    first = model.transformer.layers[0]
    last = model.transformer.layers[-1]
    assert first.attn.qkv_proj.weight.shape == (16, 16)
    assert last.attn.qkv_proj.weight.shape == (32, 16)
    assert first.ffn.proj_1.weight.shape == (16, 16)
    assert last.ffn.proj_1.weight.shape == (32, 16)
    assert first.attn.q_norm is not None
    assert first.attn.k_norm is not None

    tokens = mx.array([[1, 2, 3]])
    full = model(
        tokens,
        position_ids=mx.arange(tokens.shape[1])[None, :],
        output_hidden_states=True,
    )
    caches = make_kv_caches(config.num_transformer_layers)
    pieces = [
        model(tokens[:, index : index + 1], cache=caches).logits
        for index in range(tokens.shape[1])
    ]
    cached = mx.concatenate(pieces, axis=1)
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 3, config.vocab_size)
    assert full.last_hidden_state.shape == (1, 3, config.model_dim)
    assert len(full.hidden_states) == config.num_transformer_layers + 1
    assert [cache.offset for cache in caches] == [3, 3, 3]
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4


def test_openelm_untied_non_glu_and_optional_qk_norm_paths() -> None:
    config = openelm_config(
        ffn_with_glu=False,
        normalize_qk_projections=False,
        share_input_output_layers=False,
    )
    model = OpenELMForCausalLM(config)
    output = model(mx.array([[4, 5]]))
    mx.eval(output.logits)
    layer = model.transformer.layers[0]
    assert output.logits.shape == (1, 2, config.vocab_size)
    assert layer.ffn.proj_1.weight.shape == (config.ffn_dims[0], config.model_dim)
    assert layer.attn.q_norm is None
    assert layer.attn.k_norm is None
    assert model.lm_head.weight.shape == (config.vocab_size, config.model_dim)
