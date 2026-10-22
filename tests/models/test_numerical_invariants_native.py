"""Independent numerical invariants for proven low-precision defects."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("MLX_ONE_RUN_MLX_TESTS") != "1", reason="explicit native MLX opt-in required"
)


def test_oracle_diagnostic_accepts_absent_sinks_and_restores_hooks(monkeypatch):
    from pathlib import Path

    import mlx.core as mx
    import numpy as np

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/model_parity")
    )
    import mlx.nn as nn
    from four_path_oracle import attention_control, linear_control

    original_attention, original_linear = mx.fast.scaled_dot_product_attention, nn.Linear.__call__
    q = mx.ones((1, 1, 2, 64), dtype=mx.float16)
    for mode in ("rowwise", "contiguous_rowwise"):
        with pytest.raises(RuntimeError, match="cleanup probe"):
            with attention_control(mode), linear_control("rowwise"):
                result = mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125, sinks=None)
                mx.eval(result)
                np.testing.assert_array_equal(np.array(result), np.ones((1, 1, 2, 64)))
                raise RuntimeError("cleanup probe")
        assert mx.fast.scaled_dot_product_attention is original_attention
        assert nn.Linear.__call__ is original_linear


def test_gated_delta_decay_uses_fp32_exponential_before_state_update():
    import mlx.core as mx
    import numpy as np

    from mlx_one.models.vision_language.qwen3_5.model import _gated_delta_decay
    from mlx_one.ops.numerical_trace import compare_tensors

    a_log = mx.array([-1.5, -0.7, 0.3, 1.2], dtype=mx.bfloat16)
    a = mx.array([0.0, 1.0, 2.0, 3.0], dtype=mx.bfloat16)
    bias = mx.array([0.1, -0.2, 0.3, -0.4], dtype=mx.bfloat16)
    result = _gated_delta_decay(a, a_log, bias)
    mx.eval(result)
    arrays = [np.array(x.astype(mx.float32)).astype(np.float64) for x in (a_log, a, bias)]
    expected = np.exp(-np.exp(arrays[0]) * np.logaddexp(arrays[1] + arrays[2], 0))
    assert result.dtype == mx.float32
    # Decay multiplies FP32 recurrent state; apply its actual FP32 contract.
    metrics = compare_tensors(expected, np.array(result), dtype="float32")
    assert metrics["status"] == "PASS", metrics


def test_qwen35_depthwise_conv_accumulates_before_bfloat16_rounding():
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from mlx_one.models.vision_language.qwen3_5.model import DepthwiseCausalConv1d
    from mlx_one.ops.numerical_trace import compare_tensors

    values = mx.array([[[1.0078125], [1.0]]], dtype=mx.bfloat16)
    native = DepthwiseCausalConv1d(1, 2)
    native.weight = mx.array([[1032.0, -1040.0]], dtype=mx.bfloat16)
    reference = nn.Conv1d(1, 1, 2, bias=False)
    reference.weight = native.weight[..., None]
    output = native(values, None)
    expected = reference(mx.pad(values, ((0, 0), (1, 0), (0, 0))))
    mx.eval(output, expected)
    # Exact FP64 dot product is 0.0625. BF16 product rounding before the
    # reduction instead returns zero, violating the unchanged BF16 gate.
    assert float(expected[0, 1, 0].item()) == 0.0625
    result = compare_tensors(
        np.array(expected.astype(mx.float32)), np.array(output.astype(mx.float32)), dtype="bfloat16"
    )
    assert result["status"] == "PASS", result
    assert output.dtype == values.dtype


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_qwen35_conv_full_chunk_and_reset_preserve_history(dtype):
    import mlx.core as mx
    import numpy as np

    from mlx_one.models.vision_language.qwen3_5.model import (
        DepthwiseCausalConv1d,
        Qwen3_5LinearCache,
    )

    mx.random.seed(37)
    convolution = DepthwiseCausalConv1d(7, 4)
    convolution.weight = convolution.weight.astype(getattr(mx, dtype))
    values = mx.random.normal((1, 9, 7)).astype(getattr(mx, dtype))
    full = convolution(values, None)
    cache = Qwen3_5LinearCache(4)
    parts = [convolution(values[:, a:b], cache) for a, b in ((0, 3), (3, 8), (8, 9))]
    split = mx.concatenate(parts, axis=1)
    mx.eval(full, split)
    np.testing.assert_array_equal(
        np.array(full.astype(mx.float32)), np.array(split.astype(mx.float32))
    )
    assert cache.offset == 9
    np.testing.assert_array_equal(
        np.array(cache.conv_state.astype(mx.float32)), np.array(values[:, -3:].astype(mx.float32))
    )
    cache.reset()
    reset = convolution(values, cache)
    mx.eval(reset)
    np.testing.assert_array_equal(
        np.array(full.astype(mx.float32)), np.array(reset.astype(mx.float32))
    )
