"""MXFP4 kernels must match stock arithmetic under the unchanged project gates."""

import os

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("Requires opt-in Metal tests", allow_module_level=True)

import mlx.core as mx

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.mlxboost.gptoss_kernels import _HEADER, gate_up, project, swiglu, weighted_sum
from mlx_one.models.shared.sparse import ExpertLinear


def compare(actual, expected):
    key = str(expected.dtype).rsplit(".", 1)[-1]
    atol, rtol = TOLERANCES[key]
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


def projection(k, n, dtype):
    source = ExpertLinear(k, n, 4, bias=True)
    source.weight = (mx.random.normal((4, n, k)) * 0.04).astype(dtype)
    source.bias = (mx.random.normal((4, n)) * 0.1).astype(dtype)
    return source.to_quantized(group_size=32, bits=4, mode="mxfp4")


def test_e2m1_signed_values_and_e8m0_special_scale_bit_patterns():
    kernel = mx.fast.metal_kernel(
        name="gptoss_format_goldens",
        input_names=["codes", "scales"],
        output_names=["values", "scale_bits"],
        header=_HEADER,
        source="""
        uint i = thread_position_in_grid.x;
        if (i < 16) values[i] = gptoss_fp4(uint(codes[i]));
        if (i < 7) scale_bits[i] = as_type<uint>(gptoss_scale(scales[i]));
        """,
    )
    values, bits = kernel(
        inputs=[
            mx.arange(16, dtype=mx.uint8),
            mx.array([0, 1, 126, 127, 128, 254, 255], dtype=mx.uint8),
        ],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(16,), (7,)],
        output_dtypes=[mx.float32, mx.uint32],
    )
    expected = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
    )
    np.testing.assert_array_equal(np.asarray(values), expected)
    assert np.signbit(np.asarray(values)[8])
    np.testing.assert_array_equal(
        np.asarray(bits),
        [0x00400000, 0x00800000, 0x3F000000, 0x3F800000, 0x40000000, 0x7F000000, 0x7FC00000],
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("k,n", [(64, 17), (2880, 64)])
@pytest.mark.parametrize("words", [1, 4])
def test_mxfp4_projection_and_pair(dtype, k, n, words):
    mx.random.seed(91)
    gate, up = projection(k, n, dtype), projection(k, n, dtype)
    x = (mx.random.normal((1, 1, 1, 1, k)) * 0.2).astype(dtype)
    indices = mx.array([[[3, 0, 2, 1]]], dtype=mx.uint32)
    expected_gate, expected_up = gate(x, indices), up(x, indices)
    compare(project(gate, x, indices, words_per_lane=words), expected_gate)
    g, u = gate_up(gate, up, x, indices, activate=False, words_per_lane=words)
    compare(g, expected_gate)
    compare(u, expected_up)
    g = mx.minimum(expected_gate, 7)
    expected = (mx.clip(expected_up, -7, 7) + 1) * (g * mx.sigmoid(1.702 * g))
    compare(gate_up(gate, up, x, indices, words_per_lane=words), expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_clipping_activation_and_top4_weighted_reduction(dtype):
    gate = mx.array(
        [-100, -7.5, -7, -1.5, -0.0, 0.0, 0.5, 1.5, 7, 7.5, 100, 2], dtype=dtype
    ).reshape(1, 1, 4, 1, 3)
    up = mx.array([100, -100, -7, -1, 0, 0.5, 1, 1.5, 7, 7.5, -7.5, 2], dtype=dtype).reshape(
        gate.shape
    )
    g = mx.minimum(gate, 7)
    expected = (mx.clip(up, -7, 7) + 1) * (g * mx.sigmoid(1.702 * g))
    compare(swiglu(gate, up), expected)
    values = expected.squeeze(-2)
    scores = mx.array([[[0.1, 0.2, 0.3, 0.4]]], dtype=dtype)
    compare(weighted_sum(values, scores), (values * scores[..., None]).sum(axis=-2))


def test_invalid_expert_ids_produce_nan_without_out_of_bounds_access():
    module = projection(64, 17, mx.float32)
    result = project(module, mx.ones((1, 64)), mx.array([0, 1, 4, -1]))
    mx.eval(result)
    assert bool(mx.all(mx.isnan(result[2:])).item())
    assert bool(mx.all(mx.isfinite(result[:2])).item())


def test_wide_invalid_ids_do_not_wrap_into_valid_experts():
    module = projection(64, 17, mx.float32)
    indices = mx.array([0, 2**32, 2**32 + 4, -(2**32)], dtype=mx.int64)
    result = project(module, mx.ones((1, 64)), indices)
    mx.eval(result)
    assert bool(mx.all(mx.isfinite(result[:1])).item())
    assert bool(mx.all(mx.isnan(result[1:])).item())


@pytest.mark.parametrize("dtype,large", [(mx.bfloat16, 256), (mx.float16, 4096)])
def test_reduction_preserves_each_native_addition_rounding(dtype, large):
    values = mx.array([large, 1, -large, 1], dtype=dtype).reshape(1, 1, 4, 1)
    scores = mx.ones((1, 1, 4), dtype=dtype)
    expected = (values * scores[..., None]).sum(axis=-2)
    actual = weighted_sum(values, scores)
    mx.eval(actual, expected)
    assert expected.item() == 1
    assert bool(mx.array_equal(actual, expected).item())


def test_unsupported_formats_and_shapes_request_stock_fallback():
    source = ExpertLinear(64, 17, 4, bias=True)
    affine = source.to_quantized(group_size=32, bits=4, mode="affine")
    assert project(affine, mx.ones((1, 64)), mx.array([0, 1, 2, 3])) is None
    native = projection(64, 17, mx.float32)
    assert project(native, mx.ones((1, 64)), mx.array([0, 1])) is None
    assert project(native, mx.ones((2, 64)), mx.array([0, 1, 2, 3])) is None


def test_noncontiguous_packed_views_keep_their_row_and_bias_mapping():
    native = projection(64, 17, mx.float32)
    native.weight = native.weight[:, ::-1, :]
    native.scales = native.scales[:, ::-1, :]
    native.bias = native.bias[:, ::-1]
    x, indices = mx.ones((1, 64)), mx.array([3, 1, 0, 2])
    compare(project(native, x, indices), native(x, indices))
