"""Opt-in pinned native dense/hybrid parity; never downloads checkpoint assets."""

import os

import pytest

if os.environ.get("MLX_ONE_RUN_INFERENCE_CHECKPOINTS") != "1":
    pytest.skip("set MLX_ONE_RUN_INFERENCE_CHECKPOINTS=1 on a Metal host", allow_module_level=True)

from mlx_one.engine.parity import forced_token_parity
from mlx_one.inference_baseline import CHECKPOINTS
from mlx_one.text import load_text_model


@pytest.mark.parametrize("model_id,revision", CHECKPOINTS.values())
def test_pinned_forced_token_parity_and_hybrid_isolation(model_id, revision):
    bundle = load_text_model(model_id, revision=revision, offline=True)
    result = forced_token_parity(bundle)
    assert result["status"] == "passed", result
    assert result["cancelled_request_isolated"]
    assert result["seeded_rng_isolation"]


@pytest.mark.parametrize("mode,size", [("experimental_block", 1),
                                      ("experimental_dense_batch", 1),
                                      ("experimental_dense_batch", 2),
                                      ("experimental_dense_batch", 4),
                                      ("experimental_dense_batch", 8)])
def test_pinned_qwen_fp16_cache_and_batched_parity(mode, size):
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.qualification import execution_parity

    model_id, revision = CHECKPOINTS["qwen"]
    bundle = load_text_model(model_id, revision=revision, offline=True)
    configure_execution_dtype(bundle, "float16")
    result = execution_parity(bundle, mode=mode, batch_size=size,
                              prompt="A shared prompt with enough tokens for sealed pages. " * 16)
    assert result["status"] == "passed", result
    assert result["seeded_rng_isolation"]
    assert all(row["reused_tokens"] > 0 for row in result["prefix_restoration"])


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("key", ["qwen", "qwen15", "qwen3", "lfm"])
def test_pinned_paged_and_hybrid_state_parity(key, size):
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.qualification import execution_parity

    model_id, revision = CHECKPOINTS[key]
    bundle = load_text_model(model_id, revision=revision, offline=True)
    hybrid = key == "lfm"
    if not hybrid:
        configure_execution_dtype(bundle, "float16")
    result = execution_parity(bundle, mode="experimental_hybrid_batch" if hybrid else "paged",
                              cache_type="bf16" if hybrid else "f16", batch_size=size, count=8,
                              prompt="Shared state and divergent suffixes on native execution. "
                              * 16)
    assert result["status"] == "passed", result
    assert result["prefix_isolation_passed"]
    assert result["cleanup"]["active_handles"] == 0
    assert result["component_comparisons"]


@pytest.mark.parametrize("page_tokens", [16, 32, 64, 128])
@pytest.mark.parametrize("cache_type,execution_dtype", [("f16", "float16"), ("bf16", "bfloat16")])
def test_pinned_direct_page_layout_specializations(page_tokens, cache_type, execution_dtype):
    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.qualification import execution_parity

    model_id, revision = CHECKPOINTS["qwen"]
    bundle = load_text_model(model_id, revision=revision, offline=True)
    configure_execution_dtype(bundle, execution_dtype)
    result = execution_parity(bundle, mode="paged", count=8, batch_size=2,
                              cache_type=cache_type, page_tokens=page_tokens,
                              prompt="Direct page reads across a partial final page. " * 32)
    assert result["status"] == "passed", result
