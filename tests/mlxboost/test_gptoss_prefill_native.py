"""Prefill assignment scheduling preserves native token/expert correspondence."""

import os

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("Requires opt-in Metal tests", allow_module_level=True)

import mlx.core as mx

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.mlxboost.gptoss_prefill import prefill_moe
from mlx_one.models.shared.sparse import SelectedExperts


def fixture(length):
    mx.random.seed(137)
    experts = SelectedExperts(64, 96, 8, bias=True, swiglu_limit=7)
    for name in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(experts, name)
        module.weight = (mx.random.normal(module.weight.shape) * 0.04).astype(mx.bfloat16)
        module.bias = (mx.random.normal(module.bias.shape) * 0.1).astype(mx.bfloat16)
        setattr(experts, name, module.to_quantized(group_size=32, bits=4, mode="mxfp4"))
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    indices = mx.argsort(mx.random.normal((1, length, 8)), axis=-1)[..., :4]
    scores = mx.softmax(mx.random.normal(indices.shape), axis=-1).astype(mx.bfloat16)
    return experts, x, indices, scores


@pytest.mark.parametrize("length", [16, 33, 512, 1024])
@pytest.mark.parametrize("grouped", [False, True])
def test_sorted_assignments_restore_original_token_and_expert_order(length, grouped):
    experts, x, indices, scores = fixture(length)
    expected = (experts(x, indices) * scores[..., None]).sum(axis=-2)
    actual = prefill_moe(experts, x, indices, scores, grouped=grouped)
    mx.eval(expected, actual)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    atol, rtol = TOLERANCES["bfloat16"]
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


def test_decode_short_chunks_batching_and_other_precision_stay_on_stock():
    experts, x, indices, scores = fixture(16)
    assert prefill_moe(experts, x[:, :1], indices[:, :1], scores[:, :1]) is None
    assert prefill_moe(experts, x[:, :15], indices[:, :15], scores[:, :15]) is None
    assert prefill_moe(experts, x.astype(mx.float32), indices, scores.astype(mx.float32)) is None
    assert (
        prefill_moe(
            experts,
            mx.concatenate([x, x]),
            mx.concatenate([indices, indices]),
            mx.concatenate([scores, scores]),
        )
        is None
    )


def test_unsupported_expert_format_returns_stock_fallback():
    experts, x, indices, scores = fixture(16)
    experts.gate_proj.mode = "affine"
    assert prefill_moe(experts, x, indices, scores) is None
