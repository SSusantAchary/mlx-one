"""CPU mapping and arithmetic checks for the isolated stock-MLX diagnostic."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from mlx_one.ops.attention import attention_reference


@pytest.fixture
def candidate(monkeypatch):
    core, package = ModuleType("mlx.core"), ModuleType("mlx")
    for name in ("arange", "concatenate", "where", "triu"):
        setattr(core, name, getattr(np, name))
    core.float32, core.int32 = np.float32, np.int32
    core.full = lambda shape, value: np.full(shape, value, dtype=np.float32)
    calls = []
    core.contiguous_calls = []
    core.index_repeat_calls = []

    def contiguous(value, *, allow_col_major=False):
        assert allow_col_major is False
        core.contiguous_calls.append({"shape": value.shape, "dtype": value.dtype})
        return np.ascontiguousarray(value)

    def gather_mm(left, right, *, lhs_indices, rhs_indices):
        calls.append({"lhs_shape": left.shape, "rhs_shape": right.shape,
                      "lhs_indices": lhs_indices.copy(), "rhs_indices": rhs_indices.copy(),
                      "lhs_dtype": left.dtype, "rhs_dtype": right.dtype,
                      "rhs_strides": tuple(stride // right.itemsize for stride in right.strides)})
        return np.stack([left[i] @ right[j] for i, j in
                         zip(lhs_indices, rhs_indices, strict=True)])

    def softmax(value, *, axis, precise):
        assert precise is True
        probabilities = np.exp(value - value.max(axis=axis, keepdims=True))
        return probabilities / probabilities.sum(axis=axis, keepdims=True)

    def repeat_indices(value, repeats, axis=None):
        if value.ndim != 1 or value.dtype != np.int32 or axis is not None:
            raise AssertionError("candidate may repeat only one-dimensional int32 indices")
        core.index_repeat_calls.append({"shape": value.shape, "repeats": repeats})
        return np.repeat(value, repeats)

    core.gather_mm, core.softmax, core.repeat = gather_mm, softmax, repeat_indices
    core.contiguous = contiguous
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    path = Path(__file__).parents[2] / "scripts/qualification/reference_tiled_attention.py"
    spec = importlib.util.spec_from_file_location("reference_tiled_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_compatible_attention, core, calls


def _inputs(batch=2, heads=8, kv_heads=2, queries=7, history=5, width=8, value_width=5,
            dtype=np.float16):
    rng = np.random.default_rng(921)
    q = rng.normal(size=(batch, heads, queries, width)).astype(dtype)
    k = rng.normal(size=(batch, kv_heads, queries + history, width)).astype(dtype)
    v = rng.normal(size=(batch, kv_heads, queries + history, value_width)).astype(dtype)
    return q, k, v


def _reference(core, q, k, v, scale, mask):
    no_repeat = core.repeat
    core.repeat = np.repeat
    try:
        return attention_reference(q, k, v, scale=scale, mask=mask, stable=True, promote=True)
    finally:
        core.repeat = no_repeat


@pytest.mark.parametrize("batch,heads,kv_heads", [(1, 8, 8), (2, 8, 2), (4, 8, 1)])
@pytest.mark.parametrize("query_tile", [1, 3, 32])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "boolean", "additive"])
def test_candidate_exactly_preserves_stable_reference_rows_and_fp16_casts(
        candidate, batch, heads, kv_heads, query_tile, mask_kind):
    run, core, calls = candidate
    q, k, v = _inputs(batch, heads, kv_heads)
    allowed = np.arange(k.shape[2])[None, :] <= 5 + np.arange(q.shape[2])[:, None]
    mask = {"none": None, "causal": "causal", "boolean": allowed,
            "additive": np.where(allowed, np.float32(.000123), -np.inf).astype(np.float32)
            }[mask_kind]
    expected = _reference(core, q, k, v, 8**-.5, mask)
    calls.clear()
    actual = run(q, k, v, 8**-.5, mask=mask, query_tile=query_tile)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))
    assert actual.dtype == np.float32
    assert all(call["lhs_dtype"] == call["rhs_dtype"] == np.float32 for call in calls)
    assert all(call["rhs_shape"][0] == batch * kv_heads for call in calls)
    assert all(call["lhs_shape"][0] <= batch * heads * query_tile for call in calls)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_native_gqa_indices_preserve_batch_head_and_query_alignment(candidate, batch):
    run, core, calls = candidate
    q, k, v = _inputs(batch=batch, heads=8, kv_heads=2, queries=7)
    run(q, k, v, .25, query_tile=3)
    assert len(calls) == 6  # Three tiles, one QK and one PV gathered product per tile.
    for qk, pv, tile in zip(calls[::2], calls[1::2], (3, 3, 1), strict=True):
        expected = np.array([row * 2 + head // 4
                             for row in range(batch) for head in range(8) for _ in range(tile)])
        np.testing.assert_array_equal(qk["rhs_indices"], expected)
        np.testing.assert_array_equal(pv["rhs_indices"], expected)
        np.testing.assert_array_equal(qk["lhs_indices"], np.arange(batch * 8 * tile))
        assert qk["lhs_shape"] == (batch * 8 * tile, 1, 8)
        assert qk["rhs_shape"] == (batch * 2, 8, 12)
        assert pv["lhs_shape"] == (batch * 8 * tile, 1, 12)
        assert pv["rhs_shape"] == (batch * 2, 12, 5)
    assert core.index_repeat_calls == [{"shape": (batch * 2,), "repeats": 12},
                                       {"shape": (batch * 2,), "repeats": 12},
                                       {"shape": (batch * 2,), "repeats": 4}]


@pytest.mark.parametrize("queries,history", [(1, 0), (1, 31), (7, 19), (64, 0)])
def test_causal_alignment_uses_total_query_extent_across_tile_boundaries(
        candidate, queries, history):
    run, core, calls = candidate
    q, k, v = _inputs(batch=1, queries=queries, history=history)
    expected = _reference(core, q, k, v, .125, "causal")
    actual = run(q, k, v, .125, mask="causal", query_tile=7)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mask_shape", [(), (12,), (1, 12), (7, 12), (1, 1, 7, 12),
                                       (2, 8, 7, 12), (2, 8, 1, 12)])
@pytest.mark.parametrize("boolean", [False, True])
def test_broadcast_masks_are_sliced_only_when_the_query_dimension_is_present(
        candidate, mask_shape, boolean):
    run, core, calls = candidate
    q, k, v = _inputs()
    mask = np.ones(mask_shape, dtype=bool) if boolean else np.full(
        mask_shape, .000193, dtype=np.float32)
    expected = _reference(core, q, k, v, .125, mask)
    actual = run(q, k, v, .125, mask=mask, query_tile=3)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_candidate_uses_fp32_operators_for_each_supported_input_precision(candidate, dtype):
    run, core, calls = candidate
    q, k, v = _inputs(dtype=dtype)
    actual = run(q, k, v, .25, query_tile=3)
    expected = _reference(core, q, k, v, .25, None)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def _strided(value, layout):
    if layout == "sequence_major":
        return np.ascontiguousarray(value.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    if layout == "padded_cache":
        allocation = np.zeros((*value.shape[:2], value.shape[2] + 19, value.shape[3]),
                              dtype=value.dtype)
        allocation[:, :, :value.shape[2], :] = value
        return allocation[:, :, :value.shape[2], :]
    allocation = np.zeros((*value.shape[:-1], value.shape[-1] * 2), dtype=value.dtype)
    allocation[..., ::2] = value
    return allocation[..., ::2]


@pytest.mark.parametrize("layout", ["sequence_major", "padded_cache", "width_stride"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_strided_native_gqa_kv_keeps_reference_matrix_layout(candidate, layout, dtype):
    run, core, calls = candidate
    q, k, v = (_strided(value, layout) for value in _inputs(dtype=dtype))
    assert not k.flags.c_contiguous and not v.flags.c_contiguous
    expected = _reference(core, q, k, v, .125, "causal")
    calls.clear()
    actual = run(q, k, v, .125, mask="causal", query_tile=3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.astype(np.float16), expected.astype(np.float16))
    assert core.contiguous_calls == [{"shape": k.shape, "dtype": np.dtype(np.float32)},
                                     {"shape": v.shape, "dtype": np.dtype(np.float32)}]
    for qk, pv in zip(calls[::2], calls[1::2], strict=True):
        # RHS K stays column-major [D,K], selecting the reference's GEMV route.
        # Each V matrix remains row-major [K,Dv]. Neither repeats native heads.
        assert qk["rhs_strides"][-2:] == (1, k.shape[-1])
        assert pv["rhs_strides"][-2:] == (v.shape[-1], 1)
        assert qk["rhs_shape"][0] == pv["rhs_shape"][0] == k.shape[0] * k.shape[1]


def test_mha_retains_original_cast_order_without_gqa_normalization(candidate):
    run, core, calls = candidate
    q, k, v = (_strided(value, "padded_cache") for value in _inputs(heads=2, kv_heads=2))
    expected = _reference(core, q, k, v, .125, "causal")
    actual = run(q, k, v, .125, mask="causal", query_tile=3)
    np.testing.assert_array_equal(actual, expected)
    assert core.contiguous_calls == []


@pytest.mark.parametrize("change", ["rank", "zero", "batch", "heads", "width", "value_length",
                                    "dtype", "mixed_dtype", "mask_rank", "mask_shape",
                                    "mask_dtype", "mask_string", "causal_alignment",
                                    "tile_zero", "tile_bool", "scale_nan", "scale_bool"])
def test_invalid_options_raise_before_any_candidate_product(candidate, change):
    run, core, calls = candidate
    q, k, v = _inputs()
    options = {"scale": .125, "mask": None, "query_tile": 3}
    if change == "rank":
        q = q[0]
    elif change == "zero":
        q = q[:, :, :0]
    elif change == "batch":
        k = k[:1]
    elif change == "heads":
        q = q[:, :3]
    elif change == "width":
        k = k[..., :4]
    elif change == "value_length":
        v = v[:, :, :-1]
    elif change == "dtype":
        q, k, v = (value.astype(np.float64) for value in (q, k, v))
    elif change == "mixed_dtype":
        q = q.astype(np.float32)
    elif change == "mask_rank":
        options["mask"] = np.ones((1, 1, 1, 1, 12), bool)
    elif change == "mask_shape":
        options["mask"] = np.ones((7, 13), bool)
    elif change == "mask_dtype":
        options["mask"] = np.ones((7, 12), np.int32)
    elif change == "mask_string":
        options["mask"] = "upper-left"
    elif change == "causal_alignment":
        k, v = k[:, :, :3], v[:, :, :3]
        options["mask"] = "causal"
    elif change == "tile_zero":
        options["query_tile"] = 0
    elif change == "tile_bool":
        options["query_tile"] = True
    elif change == "scale_nan":
        options["scale"] = float("nan")
    elif change == "scale_bool":
        options["scale"] = True
    with pytest.raises(ValueError):
        run(q, k, v, **options)
    assert calls == []
