"""Single-query mask removal must preserve full/sliding attention results."""
import os

import numpy as np
import pytest

if os.environ.get('MLX_ONE_RUN_MLX_TESTS') != '1':
    pytest.skip('Requires opt-in Metal', allow_module_level=True)

import mlx.core as mx

from mlx_one.mlxboost.gptoss import gptoss_execution, try_decode_attention
from mlx_one.models.shared.decoder import attention_mask_for


@pytest.mark.parametrize('length', [1, 7, 128, 129, 513, 4096, 8193])
@pytest.mark.parametrize('window', [None, 128])
def test_unmasked_stock_attention_matches_masked_reference(length, window):
    mx.random.seed(307)
    q = mx.random.normal((1, 64, 1, 64)).astype(mx.bfloat16)
    k = mx.random.normal((1, 8, length, 64)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    sinks = mx.random.normal((64,)).astype(mx.bfloat16)
    mask = attention_mask_for(1, length-1, length-1, window=window, dtype=q.dtype)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=.125, mask=mask, sinks=sinks)
    with gptoss_execution('mlx_lm_moe') as state:
        actual = try_decode_attention(q, k, v, sinks, window=window)
        if window is not None:
            assert actual is None  # Sliding-window arithmetic stays on the reference path.
            return
        assert state.dispatch['attention candidate: unmasked_sdpa'] == 1
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)),
                                  np.asarray(expected.astype(mx.float32)))


def test_other_recipes_and_unsupported_layouts_fall_back():
    q = mx.zeros((1, 64, 1, 64), dtype=mx.bfloat16)
    k = mx.zeros((1, 8, 4, 64), dtype=mx.bfloat16)
    sinks = mx.zeros((64,))
    assert try_decode_attention(q, k, k, sinks) is None
    with gptoss_execution('prefill_sort'):
        assert try_decode_attention(q, k, k, sinks) is None
    with gptoss_execution('mlx_lm_moe'):
        assert try_decode_attention(q.astype(mx.float32), k, k, sinks) is None
        assert try_decode_attention(q, k, k, sinks, window=0) is None
        assert try_decode_attention(mx.zeros((1, 64, 2, 64)), k, k, sinks) is None
