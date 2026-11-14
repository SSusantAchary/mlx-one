"""Sink normalization, split coverage and cache-view parity for native GQA."""

import os

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("Requires opt-in Metal tests", allow_module_level=True)

import mlx.core as mx

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.mlxboost.gptoss_attention import attention


def compare(actual, expected):
    atol, rtol = TOLERANCES[str(expected.dtype).rsplit(".", 1)[-1]]
    mx.eval(actual, expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 257, 8193])
def test_sink_aware_gqa_split_matches_stock_attention(dtype, length):
    mx.random.seed(309)
    q = mx.random.normal((1, 64, 1, 64)).astype(dtype)
    k = mx.random.normal((1, 8, length, 64)).astype(dtype)
    v = mx.random.normal(k.shape).astype(dtype)
    sinks = (mx.random.normal((64,)) * 3).astype(dtype)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, sinks=sinks)
    compare(attention(q, k, v, sinks), expected)


def test_sink_is_counted_once_across_all_splits_and_subchunks():
    q = mx.zeros((1, 64, 1, 64))
    k = mx.zeros((1, 8, 3, 64))
    v = mx.ones(k.shape)
    sinks = mx.full((64,), float(np.log(3)))
    for blocks in (32, 64, 128, 256):
        compare(attention(q, k, v, sinks, blocks=blocks), mx.full(q.shape, 0.5))


def test_capacity_slices_and_noncontiguous_query_and_sink_views():
    mx.random.seed(301)
    q = mx.random.normal((1, 64, 1, 128))[..., ::2]
    k = mx.random.normal((1, 8, 768, 64))[:, :, :513, :]
    v = mx.random.normal((1, 8, 768, 64))[:, :, :513, :]
    sinks = mx.random.normal((128,))[::2]
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, sinks=sinks)
    compare(attention(q, k, v, sinks), expected)


def test_unsupported_attention_layout_requests_stock_fallback():
    q = mx.zeros((1, 64, 1, 64))
    k = mx.zeros((1, 8, 7, 64))
    sinks = mx.zeros((64,))
    assert attention(q[:, :32], k, k, sinks) is None
    assert attention(q, k, k.astype(mx.float16), sinks) is None
    assert attention(q, k, k, sinks, blocks=63) is None
