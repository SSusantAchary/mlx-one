"""Independent analytical checks, separate from whole-model roundoff gates."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("MLX_ONE_RUN_MLX_TESTS") != "1", reason="real Metal qualification required"
)


@pytest.mark.parametrize(
    "family,local",
    [(f, False) for f in ("gpt2", "qwen2", "qwen3", "llama", "lfm2", "phi3", "gemma3")]
    + [("gemma3", True)],
)
def test_analytical_projection_rotary_positions_visibility_and_reset(monkeypatch, family, local):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "benchmarks/model_parity")
    )
    from alpha_controls import cache_controls

    result = cache_controls(family, local=local)
    assert result["status"] == "PASS", result


@pytest.mark.parametrize("window", [512, 2047])
def test_exact_window_boundaries_keep_absolute_positions_and_chunk_visibility(window):
    import mlx.core as mx
    import numpy as np

    from mlx_one.core.sliding_cache import SlidingKVCache
    from mlx_one.models.shared.decoder import attention_mask_for

    cache = SlidingKVCache(window)
    cache.storage_dtype = "float16"
    total = 0
    for length in [window - 1, 1, 1, 7, 2]:
        positions = np.arange(total, total + length)
        data = mx.array(positions.reshape(1, 1, -1, 1), dtype=mx.float32)
        retained = min(total, window)
        k, v = cache.update(data, -data)
        expected_positions = np.arange(total - retained, total + length)
        np.testing.assert_array_equal(np.array(k).ravel(), expected_positions.astype(np.float16))
        np.testing.assert_array_equal(np.array(v).ravel(), -expected_positions.astype(np.float16))
        mask = attention_mask_for(length, total, retained, window=window)
        expected = (expected_positions[None, :] <= positions[:, None]) & (
            expected_positions[None, :] > positions[:, None] - window
        )
        np.testing.assert_array_equal(np.isfinite(np.array(mask)), expected)
        assert cache.offset == total + length
        assert cache.keys.shape[2] == min(total + length, window)
        total += length
    cache.reset()
    assert cache.offset == 0 and cache.keys is None


def test_lfm_convolution_state_and_output_against_numpy_cross_correlation():
    from types import SimpleNamespace

    import mlx.core as mx
    import numpy as np

    from mlx_one.core.cache import ConvCache
    from mlx_one.models.language.lfm2.reference_math import ShortConv

    mx.random.seed(771)
    block = ShortConv(SimpleNamespace(hidden_size=64, conv_L_cache=4, conv_bias=False), 0)
    x = mx.array(np.random.default_rng(77).normal(size=(1, 23, 64)) / 8, dtype=mx.float32)
    x64, wi, wo, kernel = [
        np.array(y).astype(np.float64)
        for y in (x, block.in_proj.weight, block.out_proj.weight, block.conv)
    ]
    b, c, h = np.split(x64 @ wi.T, 3, axis=-1)
    history = np.pad(b * h, [(0, 0), (3, 0), (0, 0)])
    conv = np.stack([np.sum(history[:, i : i + 4] * kernel.T, axis=1) for i in range(23)], axis=1)
    expected = (c * conv) @ wo.T
    state, offset, outputs = ConvCache(4), 0, []
    for n in [3, 1, 8, 2, 9]:
        outputs.append(block(x[:, offset : offset + n], cache=state))
        offset += n
        np.testing.assert_allclose(
            np.array(state.values), (b * h)[:, max(0, offset - 3) : offset], atol=1e-6, rtol=1e-5
        )
        assert state.offset == offset
    actual = np.array(mx.concatenate(outputs, axis=1))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    state.reset()
    assert state.values is None and state.offset == 0
    np.testing.assert_array_equal(np.array(block(x, cache=state)), np.array(block(x)))


def test_fp16_kv_representation_append_reset_and_request_isolation():
    import mlx.core as mx
    import numpy as np

    from mlx_one.core.cache import QuantizedKVCache

    a, b = [QuantizedKVCache(None, None) for _ in range(2)]
    values = np.random.default_rng(1984).normal(size=(1, 2, 9, 16)).astype(np.float32)
    for first, last in [(0, 3), (3, 4), (4, 9)]:
        k, v = a.update(mx.array(values[:, :, first:last]), mx.array(-values[:, :, first:last]))
        assert k.dtype == v.dtype == mx.float16
        np.testing.assert_array_equal(np.array(k), values[:, :, :last].astype(np.float16))
        np.testing.assert_array_equal(np.array(v), (-values[:, :, :last]).astype(np.float16))
        assert a.offset == last and b.offset == 0
    a.reset()
    k, _ = a.update(mx.ones((1, 2, 1, 16)), mx.ones((1, 2, 1, 16)))
    assert a.offset == 1 and k.shape[2] == 1 and b.offset == 0


def test_gpt2_learned_positions_follow_absolute_cache_offset(monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from mlx_one.models.language.gpt2.config import GPT2Config
    from mlx_one.models.language.gpt2.model import GPT2LMHeadModel

    mx.random.seed(81)
    model = GPT2LMHeadModel(
        GPT2Config.from_dict(
            {
                "model_type": "gpt2",
                "n_embd": 64,
                "n_layer": 1,
                "n_head": 4,
                "n_positions": 64,
                "n_ctx": 64,
                "vocab_size": 64,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
    )
    model.eval()
    target = model.transformer.h[0].attn.c_attn
    norm = model.transformer.h[0].ln_1
    observed = []
    original = nn.Linear.__call__

    def capture(layer, value):
        if layer is target:
            observed.append(np.array(value).astype(np.float64))
        return original(layer, value)

    monkeypatch.setattr(nn.Linear, "__call__", capture)
    cache = model.make_cache()
    offset = 0
    for ids in [[3, 7, 11], [2], [5, 9, 1, 4]]:
        output = model(mx.array([ids]), cache=cache).logits
        mx.eval(output)
        raw = np.array(model.transformer.wte.weight)[ids].astype(np.float64) + np.array(
            model.transformer.wpe.weight
        )[offset : offset + len(ids)].astype(np.float64)
        mean = raw.mean(axis=-1, keepdims=True)
        expected = (raw - mean) / np.sqrt(
            np.mean((raw - mean) ** 2, axis=-1, keepdims=True) + norm.eps
        )
        expected = expected * np.array(norm.weight) + np.array(norm.bias)
        np.testing.assert_allclose(observed[-1][0], expected, atol=1e-5, rtol=1e-5)
        offset += len(ids)
        assert all(c.offset == offset for c in cache)
