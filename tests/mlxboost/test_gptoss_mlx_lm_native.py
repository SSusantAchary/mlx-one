"""Compare the adaptation with actual pinned upstream definitions on Metal."""

import os
import runpy
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("Requires opt-in Metal tests", allow_module_level=True)

import mlx.core as mx

from mlx_one.mlxboost.gptoss_mlx_lm import mlx_lm_moe
from mlx_one.models.shared.sparse import SelectedExperts

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "local/gptoss20b/upstreams/mlx-lm").exists():
    pytest.skip("Requires the pinned development source checkout", allow_module_level=True)


@pytest.mark.parametrize("length", [1, 15, 16, 33, 512, 1024, 2048])
def test_matches_actual_upstream_at_decode_and_sort_threshold(length):
    reference, _, compatibility = runpy.run_path(
        str(ROOT / "benchmarks/gptoss20b_upstream_moe.py")
    )["load_reference"]()
    mx.random.seed(181)
    experts = SelectedExperts(64, 96, 8, bias=True, swiglu_limit=7)
    for name in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(experts, name)
        module.weight = mx.random.normal(module.weight.shape).astype(mx.bfloat16)
        module.bias = mx.random.normal(module.bias.shape).astype(mx.bfloat16)
        setattr(experts, name, module.to_quantized(group_size=32, bits=4, mode="mxfp4"))
    x = mx.random.normal((1, length, 64)).astype(mx.bfloat16)
    indices = mx.argsort(mx.random.normal((1, length, 8)))[..., :4]
    scores = mx.softmax(mx.random.normal(indices.shape), precise=True).astype(mx.bfloat16)
    expected = reference(experts, x, indices, scores)
    actual = mlx_lm_moe(experts, x, indices, scores)
    mx.eval(expected, actual)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert bool(mx.all(actual == expected).item())
    assert len(compatibility) == 3
