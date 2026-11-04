"""Real MLX primitive and cache checks, separate from checkpoint qualification."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MLXBOOST_NATIVE_TESTS") != "1", reason="requires an explicit Metal test session"
)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 128])
def test_quantized_linear_matches_pinned_stock_operation(bits, rows):
    import mlx.core as mx
    import mlx.nn as nn

    # Actual Qwen2.5 Q-projection dimensions. Q4 is a primitive diagnostic only;
    # it does not qualify a converted model or change any stored checkpoint.
    mx.random.seed(7)
    linear = nn.QuantizedLinear.from_linear(
        nn.Linear(896, 896, bias=True), group_size=64, bits=bits
    )
    x = mx.random.normal((1, rows, 896)).astype(mx.float16)
    linear.set_dtype(mx.float16)
    actual = linear(x)
    expected = (
        mx.quantized_matmul(
            x,
            linear.weight,
            scales=linear.scales,
            biases=linear.biases,
            transpose=True,
            group_size=64,
            bits=bits,
        )
        + linear.bias
    )
    mx.eval(actual, expected)
    assert bool(mx.all(actual == expected).item())
    assert linear.weight.dtype == mx.uint32


def test_dense_cache_growth_boundary_offsets_and_no_cross_request_state():
    import mlx.core as mx

    from mlx_one.core.cache import KVCache

    cache, other = KVCache(), KVCache()
    values = mx.arange(258 * 64).reshape(1, 1, 258, 64).astype(mx.float16)
    for start, stop in [(0, 255), (255, 256), (256, 257), (257, 258)]:
        keys, vals = cache.update(values[:, :, start:stop], -values[:, :, start:stop])
        mx.eval(keys, vals)
        assert cache.offset == stop
        assert bool(mx.all(keys == values[:, :, :stop]).item())
        assert bool(mx.all(vals == -values[:, :, :stop]).item())
        assert other.offset == 0 and other.keys is None
    assert cache.keys.shape[2] == 512
    cache.reset()
    assert cache.offset == 0 and cache.keys is None


def test_greedy_sampling_matches_reference_tokens_on_near_ties():
    import random

    import mlx.core as mx

    from mlx_one.text.sampling import select_token
    from mlx_one.text.schemas import TextGenerationOptions

    # Exact ties and neighboring representable values near the winning logits.
    for dtype in [mx.float32, mx.float16, mx.bfloat16]:
        for values in [[0, 0, -1], [1, 1.02, -1], [0, 0.02, -10]]:
            logits = mx.array(values, dtype=dtype)
            expected = int(mx.argmax(logits - mx.logsumexp(logits)).item())
            assert (
                select_token(logits, TextGenerationOptions(temperature=0), random.Random(0))
                == expected
            )
