import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.engine.kernel_dispatch import qmm_decision
from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
from mlx_one.ops.quantized_linear import try_quantized_linear


@pytest.mark.parametrize("bits", (4, 8))
@pytest.mark.parametrize("rows", (1, 2, 64, 127, 512))
def test_stock_qmm_receives_original_packed_payload_and_real_rows(monkeypatch, bits, rows):
    rng = np.random.default_rng(7)
    monkeypatch.setattr("mlx_one.ops.attention.mlx_version", lambda: (0, 32))
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    class Linear(dict):
        pass
    linear = Linear()
    linear.bits, linear.group_size, linear.mode = bits, 64, "affine"
    codes = rng.integers(0, 2**bits, size=(3, 128), dtype=np.uint32)
    lanes = 32 // bits
    shifts = np.arange(lanes, dtype=np.uint32) * bits
    linear.weight = np.bitwise_or.reduce(codes.reshape(3, -1, lanes) << shifts, axis=-1)
    linear.scales = np.full((3, 2), .015625, dtype=np.float16)
    linear.biases = np.full((3, 2), -.125, dtype=np.float16)
    before = linear.weight.copy()
    calls = []
    def qmm(value, weight, **settings):
        assert value.dtype == np.float16
        calls.append((value.shape, weight, settings))
        unpacked = ((weight[..., None] >> shifts) & (2**bits - 1)).reshape(3, 128)
        dense = (unpacked * np.repeat(settings["scales"], 64, axis=-1)
                 + np.repeat(settings["biases"], 64, axis=-1))
        return (value.astype(np.float64) @ dense.T).astype(value.dtype)
    mx.metal = SimpleNamespace(is_available=lambda: True)
    mx.gpu = "gpu"
    mx.default_device = lambda: "gpu"
    mx.quantized_matmul = qmm
    mx.float32 = np.float32
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    value = rng.normal(size=(1, rows, 128)).astype(np.float16)
    with math_execution(NativeMathPolicy(projection="mlx_qmm"), family="qwen3"):
        used, actual = try_quantized_linear(linear, value)
    if rows != 1:
        assert not used and actual is None and not calls
        np.testing.assert_array_equal(linear.weight, before)
        return
    expected_weight = codes * np.repeat(linear.scales, 64, axis=-1) + np.repeat(
        linear.biases, 64, axis=-1)
    expected = value.astype(np.float64) @ expected_weight.T
    assert used and len(calls) == 1
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)
    np.testing.assert_array_equal(linear.weight, before)
    assert calls[0][1] is linear.weight
    assert calls[0][2]["scales"] is linear.scales
    assert calls[0][2]["biases"] is linear.biases
    assert linear.weight.dtype == np.uint32


@pytest.mark.parametrize("change", ({"bits": 3}, {"bits": 8.0}, {"metal_available": False},
    {"mlx_version": (0, 31)}, {"group_size": 64.0}, {"mode": "mxfp4"}, {"weight_dtype": "float16"},
    {"activation_dtype": "float32"}, {"scales_shape": (2, 2)}, {"rows": 0},
    {"group_size": 16}, {"width": 127}, {"metadata_dtypes": ("float16", "float32")}))
def test_layout_mismatch_falls_back_without_execution(change):
    parameters = dict(family="qwen3", rows=1, width=128, weight_shape=(3, 16),
        scales_shape=(3, 2), biases_shape=(3, 2), bits=4, group_size=64, mode="affine",
        weight_dtype="uint32", activation_dtype="float16",
        metadata_dtypes=("float16", "float16"), mlx_version=(0, 32), metal_available=True)
    parameters.update(change)
    decision = qmm_decision(NativeMathPolicy(projection="mlx_qmm"), **parameters)
    assert decision.selected == "reference"


def test_native_matrix_qmm_failure_falls_back_before_cache_mutation():
    decision = qmm_decision(NativeMathPolicy(projection="mlx_qmm"), family="qwen3",
        rows=128, width=1024, weight_shape=(2048, 256), scales_shape=(2048, 16),
        biases_shape=(2048, 16), bits=8, group_size=64, mode="affine", weight_dtype="uint32",
        activation_dtype="float16", metadata_dtypes=("float16", "float16"),
        mlx_version=(0, 32), metal_available=True)
    assert decision.selected == "reference"
    assert "native" in decision.reason and "parity" in decision.reason
