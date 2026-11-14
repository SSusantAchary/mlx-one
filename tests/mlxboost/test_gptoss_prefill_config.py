"""Public TTFT opt-in is scoped and remains backend-free until native execution."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.mlxboost import MLXBoostConfig
from mlx_one.server.config import ServerConfig


def test_gptoss_config_is_part_of_the_public_mlx_one_api():
    from mlx_one import MLXBoostConfig as PublicConfig

    assert PublicConfig is MLXBoostConfig
    assert not PublicConfig().enabled
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.2" in result.output


@pytest.fixture
def profile():
    affine4 = {"bits": 4, "group_size": 64, "mode": "affine"}
    q = {
        "bits": 4,
        "group_size": 32,
        "mode": "mxfp4",
        "model.embed_tokens": affine4.copy(),
        "lm_head": affine4.copy(),
    }
    for layer in range(24):
        q[f"model.layers.{layer}.mlp.router"] = {"bits": 8, "group_size": 64, "mode": "affine"}
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            q[f"model.layers.{layer}.self_attn.{projection}"] = affine4.copy()
    config = SimpleNamespace(
        hidden_size=2880,
        intermediate_size=2880,
        num_hidden_layers=24,
        num_local_experts=32,
        num_experts_per_tok=4,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        sliding_window=128,
        swiglu_limit=7,
    )
    return dict(
        family="gpt_oss",
        dtype="bfloat16",
        quantization=q,
        backend="dense",
        slots=1,
        cache_reuse=False,
        mlx_version="0.32.2",
        metal_available=True,
        model_config=config,
        cache_types=("bf16", "bf16"),
        device_info={"device_name": "Apple M4", "memory_size": 32 * 1024**3},
    )


@pytest.mark.parametrize("recipe", ["gptoss_prefill_sort", "gptoss_mlx_lm"])
def test_prefill_sort_is_explicit_and_defaults_off(profile, recipe):
    assert not MLXBoostConfig().gptoss_prefill_sort
    assert not MLXBoostConfig().gptoss_mlx_lm
    enabled = MLXBoostConfig(**{recipe: True})
    assert enabled.resolve(**profile) == (enabled, None)
    profile["cache_types"] = ("f16", "f16")
    assert enabled.resolve(**profile) == (enabled, None)
    with pytest.raises(ValueError, match="explicit"):
        ServerConfig(mlxboost_config=enabled)
    assert (
        ServerConfig(inference_mode="mlxboost_b1", mlxboost_config=enabled).mlxboost_config
        == enabled
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"family": "qwen2"},
        {"dtype": "float16"},
        {"mlx_version": "0.32.4"},
        {"slots": 2},
        {"cache_reuse": True},
        {"metal_available": False},
        {"backend": "paged"},
        {"cache_types": ("q4_0", "q4_0")},
        {"device_info": {"device_name": "Apple M4 Pro", "memory_size": 32 * 1024**3}},
        {"device_info": {"device_name": "Apple M4", "memory_size": 16 * 1024**3}},
    ],
)
@pytest.mark.parametrize("recipe", ["gptoss_prefill_sort", "gptoss_mlx_lm"])
def test_unqualified_execution_profiles_fall_back(profile, changes, recipe):
    profile.update(changes)
    resolved, reason = MLXBoostConfig(**{recipe: True}).resolve(**profile)
    assert not resolved.enabled and reason
    with pytest.raises(ValueError, match="strict_reference"):
        MLXBoostConfig(**{recipe: True}, strict_reference=True).resolve(**profile)


@pytest.mark.parametrize("recipe", ["gptoss_prefill_sort", "gptoss_mlx_lm"])
def test_q8_nonexpert_checkpoint_and_changed_geometry_do_not_inherit_qualification(profile, recipe):
    q8 = deepcopy(profile)
    q8["quantization"]["lm_head"]["bits"] = 8
    assert not MLXBoostConfig(**{recipe: True}).resolve(**q8)[0].enabled
    q8 = deepcopy(profile)
    q8["quantization"]["model.layers.23.self_attn.q_proj"]["bits"] = 8
    assert not MLXBoostConfig(**{recipe: True}).resolve(**q8)[0].enabled
    profile["model_config"].num_hidden_layers = 36
    assert not MLXBoostConfig(**{recipe: True}).resolve(**profile)[0].enabled


def test_unqualified_combinations_and_nonboolean_flags_are_rejected(profile):
    config, reason = MLXBoostConfig(gptoss_prefill_sort=True, fast_decode=True).resolve(**profile)
    assert not config.enabled and "fast_decode disabled" in reason
    config, reason = MLXBoostConfig(gptoss_prefill_sort=True, gptoss_mlx_lm=True).resolve(**profile)
    assert not config.enabled and "one GPT-OSS recipe" in reason
    with pytest.raises(TypeError, match="boolean"):
        MLXBoostConfig(gptoss_prefill_sort=1)


def test_cli_exposes_the_same_opt_in():
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--mlxboost-gptoss-prefill-sort" in result.output
    assert "--mlxboost-gptoss-mlx-lm" in result.output
