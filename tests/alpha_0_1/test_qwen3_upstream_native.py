"""Explicitly authorized tiny native B1 contracts for the parallel Qwen3 port."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("explicit Metal authorization required", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn

from mlx_one.inference_baseline import TOLERANCES
from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.language.qwen3.upstream_model import Qwen3UpstreamForCausalLM


@pytest.mark.parametrize("quantized", [False, True])
def test_candidate_full_vs_split_logits_and_kv(quantized):
    mx.random.seed(23)
    config = Qwen3Config.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": 128,
            "intermediate_size": 256,
            "head_dim": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "tie_word_embeddings": True,
        }
    )
    model = Qwen3UpstreamForCausalLM(config)
    dtype = "float16" if quantized else "float32"
    if quantized:
        nn.quantize(model, group_size=64, bits=8)
        model.set_dtype(mx.float16, predicate=lambda dtype: dtype != mx.uint32)
    ids = mx.array([[i % 64 for i in range(33)]])
    full_cache, split_cache = model.make_cache(), model.make_cache()
    full = model(ids, cache=full_cache).logits[:, -1:]
    mx.eval(model(ids[:, :-1], cache=split_cache).logits)
    split = model(ids[:, -1:], cache=split_cache).logits
    mx.eval(full, split)
    atol, rtol = TOLERANCES[dtype]
    assert bool(mx.allclose(full, split, atol=atol, rtol=rtol).item())
    for left, right in zip(full_cache, split_cache, strict=True):
        assert left.offset == right.offset == 33
        for x, y in zip(left.state(), right.state(), strict=True):
            assert x.dtype == y.dtype == getattr(mx, dtype)
            assert bool(mx.allclose(x, y, atol=atol, rtol=rtol).item())
    for cache in full_cache:
        cache.reset()
    reset = model(ids, cache=full_cache).logits[:, -1:]
    mx.eval(reset)
    assert bool(mx.array_equal(full, reset).item())
