"""Tiny new-family logits/state checks against the provided MLX-LM snapshot."""

import importlib.util
import os
from dataclasses import asdict
from pathlib import Path

import pytest
from test_new_language_contracts import tiny_config

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.core.registry import get_registration
from mlx_one.models.language.gemma3.model import Gemma3RMSNorm
from mlx_one.models.shared.transformer import apply_rope


@pytest.fixture(scope="module")
def reference():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "new_family_local_reference", root / "benchmarks/model_parity/local_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def upstream_snapshot():
    root = Path(__file__).resolve().parents[2]
    if not (root / "references_libs/mlx-lm-main/mlx_lm").is_dir():
        pytest.skip("optional local MLX-LM reference checkout is not installed")


@pytest.mark.usefixtures("upstream_snapshot")
@pytest.mark.parametrize("family", ["phi3", "gemma3_text"])
def test_new_family_full_incremental_chunked_reset_and_local_reference(reference, family):
    mx.random.seed(11)
    config = tiny_config(family)
    native = get_registration(family).model_class()(config)
    native.eval()
    module = reference.reference_module(family)
    upstream = module.Model(module.ModelArgs.from_dict(asdict(config)))
    weights = dict(tree_flatten(native.parameters()))
    if hasattr(upstream, "sanitize"):
        weights = upstream.sanitize(weights)
    upstream.load_weights(list(weights.items()), strict=True)
    upstream.eval()
    tokens = mx.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]])
    actual, expected = native(tokens).logits, upstream(tokens)
    report = reference.compare(expected, actual, "float32")
    assert report["status"] == "PASS", report
    cache = native.make_cache()
    incremental = mx.concatenate(
        [native(tokens[:, i : i + 1], cache=cache).logits for i in range(tokens.shape[1])], axis=1
    )
    assert reference.compare(actual, incremental, "float32")["status"] == "PASS"
    cache = native.make_cache()
    chunks = mx.concatenate(
        [
            native(tokens[:, start:end], cache=cache).logits
            for start, end in [(0, 3), (3, 8), (8, 11)]
        ],
        axis=1,
    )
    assert reference.compare(actual, chunks, "float32")["status"] == "PASS"
    assert all(c.offset == 11 for c in cache)
    if family == "gemma3_text":
        assert cache[0].keys.shape[2] == 4 and cache[-1].state()[0].shape[2] == 11
    for c in cache:
        c.reset()
    assert (
        reference.compare(actual, native(tokens, cache=cache).logits, "float32")["status"] == "PASS"
    )
    with pytest.raises(ValueError, match="positions"):
        native(mx.zeros((1, 65), dtype=mx.int32))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_rope_preserves_input_precision(dtype):
    values = mx.ones((1, 2, 3, 8), dtype=dtype)
    assert apply_rope(values, offset=5, theta=10000).dtype == dtype


@pytest.mark.usefixtures("upstream_snapshot")
@pytest.mark.parametrize("offset", [0, 4096, 8192, 32768])
@pytest.mark.parametrize("factor", [8.0, 32.0])
def test_llama3_frequency_bands_match_local_reference(reference, offset, factor):
    scaling = dict(
        rope_type="llama3",
        factor=factor,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )
    module = reference.reference_module("rope_utils")
    rope = module.Llama3RoPE(64, base=500000, scaling_config=scaling)
    values = mx.random.normal((1, 2, 3, 64))
    actual = apply_rope(values, offset=offset, theta=500000, rope_scaling=scaling)
    assert reference.compare(rope(values, offset=offset), actual, "float32")["status"] == "PASS"


def test_gemma_offset_norm_identity_and_nonidentity():
    values = mx.array([[1.0, 2.0, 3.0, 4.0]])
    norm = Gemma3RMSNorm(4, 1e-6)
    expected = values * mx.rsqrt(mx.mean(values**2, axis=-1, keepdims=True) + 1e-6)
    assert mx.max(mx.abs(norm(values) - expected)).item() < 1e-6
    norm.weight = mx.array([0.0, 0.25, -0.5, 1.0])
    assert mx.max(mx.abs(norm(values) - expected * (1 + norm.weight))).item() < 1e-6


def test_phi3_sliding_boundary_full_incremental_chunked_and_bounded_storage(reference):
    mx.random.seed(31)
    model = get_registration("phi3").model_class()(tiny_config("phi3", sliding_window=4))
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    full = model(tokens).logits
    for pieces in ([(i, i + 1) for i in range(9)], [(0, 3), (3, 7), (7, 9)]):
        cache = model.make_cache()
        actual = mx.concatenate([model(tokens[:, a:b], cache=cache).logits for a, b in pieces], 1)
        assert reference.compare(full, actual, "float32")["status"] == "PASS"
        assert all(c.offset == 9 and c.keys.shape[2] == 4 for c in cache)
    # Perturbing tokens outside a one-layer receptive window cannot affect its final output.
    shallow = get_registration("phi3").model_class()(
        tiny_config("phi3", num_hidden_layers=1, sliding_window=4)
    )
    shallow.eval()
    changed = mx.array([[10, 11, 12, 13, 14, 6, 7, 8, 9]])
    assert (
        reference.compare(shallow(tokens).logits[:, -1], shallow(changed).logits[:, -1], "float32")[
            "status"
        ]
        == "PASS"
    )


@pytest.mark.usefixtures("upstream_snapshot")
@pytest.mark.parametrize("partial", [1.0, 0.5])
def test_phi3_longrope_explicit_discontinuous_positions(reference, partial):
    from mlx_one.models.language.phi3.model import Phi3Attention

    count = int(2 * partial)
    scaling = {"type": "longrope", "short_factor": [1, 2][:count], "long_factor": [3, 4][:count]}
    attention = Phi3Attention(
        tiny_config(
            "phi3",
            max_position_embeddings=64,
            original_max_position_embeddings=32,
            rope_scaling=scaling,
            partial_rotary_factor=partial,
        )
    )
    values = mx.random.normal((1, 2, 3, 4))
    positions = mx.array([[1, 4, 9]])
    actual = attention.rotary(values, 0, positions)
    expected = mx.concatenate(
        [attention.rotary(values[:, :, i : i + 1], p, None) for i, p in enumerate((1, 4, 9))],
        axis=2,
    )
    assert reference.compare(expected, actual, "float32")["status"] == "PASS"
    upstream = reference.reference_module("rope_utils").SuScaledRoPE(
        int(4 * partial),
        base=10000,
        max_position_embeddings=64,
        original_max_position_embeddings=32,
        short_factor=scaling["short_factor"],
        long_factor=scaling["long_factor"],
    )
    expected = mx.concatenate(
        [upstream(values[:, :, i : i + 1], offset=p) for i, p in enumerate((1, 4, 9))], axis=2
    )
    assert reference.compare(expected, actual, "float32")["status"] == "PASS"


def test_quantized_projection_trace_covers_bypassed_modules_and_restores(monkeypatch):
    import mlx.nn as nn

    from mlx_one.models.shared import execution

    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks/model_parity"))
    spec = importlib.util.spec_from_file_location(
        "packed_trace_test", root / "benchmarks/model_parity/quantized_projection_trace.py"
    )
    tracing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracing)

    class TinyPacked(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
            self.o_proj = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)

        def __call__(self, value):
            value = execution.project(self.q_proj, value, execution_dtype="float16")
            return execution.project(self.o_proj, value, execution_dtype="float16")

    model = TinyPacked()
    model.set_dtype(mx.float16, predicate=lambda dtype: dtype != mx.uint32)
    values = mx.ones((1, 3, 64), dtype=mx.float16)
    expected = model(values)
    original = execution._project
    with tracing.capture_with_projections(model, "qwen2") as events:
        actual = model(values)
    assert [name for name, _ in events] == ["q_proj#1", "o_proj.input#1", "o_proj#1"]
    assert mx.array_equal(actual, expected).item()
    assert execution._project is original
    with pytest.raises(RuntimeError), tracing.capture_with_projections(model, "qwen2"):
        raise RuntimeError("diagnostic failure")
    assert execution._project is original
