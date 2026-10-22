"""Isolated tensor checks against read-only local native reference sources."""

import importlib.util
import os
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.core.cache import KVCache
from mlx_one.models.shared.transformer import GroupedQueryAttention, apply_rope, causal_mask
from mlx_one.models.vision_language.qwen3_5.model import Qwen3_5RMSNorm, _l2_normalize


@pytest.fixture(scope="module")
def local_reference():
    root = Path(__file__).resolve().parents[2]
    if not (root / "references_libs/mlx-lm-main").is_dir():
        pytest.skip("required local reference snapshot is unavailable")
    spec = importlib.util.spec_from_file_location(
        "revamp_math_reference", root / "benchmarks/model_parity/local_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("offset", [0, 1, 17])
def test_rope_positions_and_cached_offsets_match_local_reference(local_reference, offset):
    reference = local_reference.reference_module("qwen3")
    mx.random.seed(41)
    values = mx.random.normal((1, 4, 3, 8))
    rope = reference.initialize_rope(8, base=10000, traditional=False,
                                    scaling_config=None, max_position_embeddings=128)
    result = local_reference.compare(rope(values, offset=offset),
                                     apply_rope(values, offset=offset, theta=10000), "float32")
    assert result["status"] == "PASS", result


def test_qwen3_qk_norm_gqa_mask_and_cached_attention_match_local_reference(local_reference):
    reference = local_reference.reference_module("qwen3")
    args = reference.ModelArgs(model_type="qwen3", hidden_size=16, num_hidden_layers=1,
                               intermediate_size=32, num_attention_heads=4,
                               rms_norm_eps=1e-6, vocab_size=64, num_key_value_heads=2,
                               max_position_embeddings=128, head_dim=8,
                               tie_word_embeddings=True, rope_theta=10000)
    native = GroupedQueryAttention(16, 4, 2, 8, rope_theta=10000, qkv_bias=False,
                                   qk_norm=True, norm_eps=1e-6)
    upstream = reference.Attention(args)
    upstream.load_weights(tree_flatten(native.parameters()), strict=True)
    hidden = mx.random.normal((1, 5, 16))
    expected = upstream(hidden, mask="causal")
    actual = native(hidden, mask=causal_mask(5))
    assert local_reference.compare(expected, actual, "float32")["status"] == "PASS"
    cache = KVCache()
    pieces = [native(hidden[:, i:i + 1], mask=causal_mask(1, i), cache=cache)
              for i in range(5)]
    assert local_reference.compare(actual, mx.concatenate(pieces, axis=1),
                                   "float32")["status"] == "PASS"


def test_qwen35_l2_epsilon_and_readout_scale_match_local_reference(local_reference):
    reference = local_reference.reference_module("gated_delta")
    values = mx.array([[[[0., 0., 0., 0.], [1e-5, -1e-5, 1e-5, -1e-5],
                         [0.1, -0.2, 0.3, -0.4]]]])
    query, key = reference.normalize_qk(values, values, inv_scale=0.5, eps=1e-6)
    native = _l2_normalize(values)
    assert local_reference.compare(query, native * 0.5, "float32")["status"] == "PASS"
    assert local_reference.compare(key, native, "float32")["status"] == "PASS"


def test_qwen35_offset_norm_matches_folded_publisher_scale(local_reference):
    reference = local_reference.reference_module("qwen3_5")
    native = Qwen3_5RMSNorm(8, 1e-6)
    native.weight = mx.linspace(-0.5, 0.5, 8)
    upstream = reference.nn.RMSNorm(8, eps=1e-6)
    upstream.weight = native.weight + 1
    values = mx.random.normal((1, 4, 8))
    result = local_reference.compare(upstream(values), native(values), "float32")
    assert result["status"] == "PASS", result
