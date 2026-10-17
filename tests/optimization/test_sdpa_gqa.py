import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.engine.optimization_policy import NativeMathPolicy, math_execution
from mlx_one.ops.attention import attention


def test_native_fp16_decode_failure_preserves_reference_before_cache_mutation():
    from mlx_one.engine.kernel_dispatch import sdpa_decision

    decision = sdpa_decision(NativeMathPolicy(attention="mlx_sdpa"), family="qwen3",
        q_shape=(1, 16, 1, 128), k_shape=(1, 8, 129, 128), v_shape=(1, 8, 129, 128),
        dtypes=("float16",) * 3, mask_kind="none", mlx_version=(0, 32), metal_available=True)
    assert decision.selected == "reference"
    assert "native" in decision.reason and "decode" in decision.reason


def oracle(q, k, v, scale, mask):
    k = np.repeat(k.astype(np.float64), q.shape[1] // k.shape[1], axis=1)
    v = np.repeat(v.astype(np.float64), q.shape[1] // v.shape[1], axis=1)
    scores = q.astype(np.float64) @ k.transpose(0, 1, 3, 2) * scale
    if isinstance(mask, str):
        valid = np.arange(k.shape[2])[None, :] <= (
            k.shape[2] - q.shape[2] + np.arange(q.shape[2])[:, None])
        scores = np.where(valid, scores, -np.inf)
    elif mask is not None:
        scores = np.where(mask, scores, -np.inf) if mask.dtype == bool else scores + mask
    probability = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability @ v


@pytest.fixture
def core(monkeypatch):
    monkeypatch.setattr("mlx_one.ops.attention.mlx_version", lambda: (0, 32))
    package, mx = ModuleType("mlx"), ModuleType("mlx.core")
    calls = []
    for name in ("arange", "repeat", "triu", "full", "where"):
        setattr(mx, name, getattr(np, name))
    mx.float32 = np.float32
    mx.gpu = "gpu"
    mx.default_device = lambda: "gpu"
    def gather_mm(left, right, *, lhs_indices, rhs_indices):
        return np.stack([left[i] @ right[j] for i, j in zip(lhs_indices, rhs_indices, strict=True)])
    mx.gather_mm = gather_mm
    def softmax(value, *, axis, precise):
        exponent = np.exp(value - value.max(axis=axis, keepdims=True))
        return exponent / exponent.sum(axis=axis, keepdims=True)
    mx.softmax = softmax
    def sdpa(q, k, v, *, scale, mask, force_fused):
        assert force_fused is True
        assert q.dtype == k.dtype == v.dtype == np.float32
        calls.append((q.shape, k.shape, v.shape))
        return oracle(q, k, v, scale, mask).astype(q.dtype)
    mx.fast = SimpleNamespace(scaled_dot_product_attention=sdpa)
    mx.metal = SimpleNamespace(is_available=lambda: True)
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    return mx, calls


@pytest.mark.parametrize("d", (64, 128, 256))
@pytest.mark.parametrize("gqa", (1, 2, 4, 8))
@pytest.mark.parametrize("queries,offset", ((1, 0), (1, 9), (2, 9), (16, 0), (16, 9)))
def test_gqa_and_lower_right_causality_match_independent_high_precision(core, d, gqa,
                                                                      queries, offset):
    rng = np.random.default_rng(301)
    q = rng.normal(size=(1, gqa, queries, d)).astype(np.float16)
    k = rng.normal(size=(1, 1, queries + offset, d)).astype(np.float16)
    v = rng.normal(size=k.shape).astype(np.float16)
    expected = oracle(q, k, v, d**-.5, "causal")
    actual = attention(q, k, v, scale=d**-.5, mask="causal", stable=True, promote=True)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        optimized = attention(q, k, v, scale=d**-.5, mask="causal", stable=True, promote=True)
    np.testing.assert_allclose(optimized, expected, atol=1e-2, rtol=1e-2)
    assert core[1] == ([] if queries == 1 else [(q.shape, k.shape, v.shape)])


@pytest.mark.parametrize("mask_kind", ("none", "bool", "additive"))
def test_array_masks_and_single_token_unmasked_history(core, mask_kind):
    rng = np.random.default_rng(88)
    q = rng.normal(size=(1, 4, 2, 64)).astype(np.float16)
    k = rng.normal(size=(1, 1, 11, 64)).astype(np.float16)
    v = rng.normal(size=k.shape).astype(np.float16)
    valid = np.ones((2, 11), dtype=bool)
    valid[:, 1::3] = False
    mask = None if mask_kind == "none" else (valid if mask_kind == "bool" else
                                            np.where(valid, 0.0, -np.inf).astype(np.float32))
    expected = oracle(q, k, v, .125, mask)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        actual = attention(q, k, v, scale=.125, mask=mask)
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)


def test_runtime_failure_is_not_retried_through_reference(core, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("injected kernel failure")
    monkeypatch.setattr(core[0].fast, "scaled_dot_product_attention", broken)
    q = np.ones((1, 1, 2, 64), dtype=np.float16)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        with pytest.raises(RuntimeError, match="injected"):
            attention(q, q, q, scale=.125, mask="causal")


def test_float32_soft_bias_is_not_silently_rounded(core):
    q = np.ones((1, 1, 2, 64), dtype=np.float16)
    mask = np.array([[0.00012345, -np.inf], [0.00019345, 0.0]], dtype=np.float32)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        actual = attention(q, q, q, scale=.125, mask=mask, stable=True, promote=True)
    np.testing.assert_allclose(actual, oracle(q, q, q, .125, mask), atol=1e-4, rtol=1e-4)
    assert core[1] == []


def test_cpu_backend_never_selects_fused_gpu_route(core):
    core[0].default_device = lambda: "cpu"
    q = np.ones((1, 1, 2, 64), dtype=np.float16)
    with math_execution(NativeMathPolicy(attention="mlx_sdpa"), family="lfm2"):
        attention(q, q, q, scale=.125, mask="causal", stable=True, promote=True)
    assert core[1] == []
