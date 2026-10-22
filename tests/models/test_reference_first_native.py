"""Whole-port differential and independent controls, isolated from the Engine."""

import importlib.util
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.core.registry import get_registration

FAMILIES = ("qwen3", "gpt2", "qwen2", "llama", "lfm2", "qwen3_5", "gemma3_text", "phi3")


@pytest.fixture(scope="module")
def reference():
    root = Path(__file__).resolve().parents[2]
    if not (root / "references_libs/mlx-lm-main").is_dir():
        pytest.skip("local reference snapshot unavailable")
    spec = importlib.util.spec_from_file_location(
        "reference_first_test", root / "benchmarks/model_parity/local_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_for(family):
    from test_native_gpt2_architecture import tiny_config as gpt2
    from test_native_lfm_architectures import dense_config as lfm2
    from test_native_llama_mlx import tiny_config as llama
    from test_native_qwen_architectures import qwen2_config, qwen3_config
    from test_native_qwen_multimodal_architectures import qwen3_5_config
    from test_new_language_contracts import tiny_config

    return {
        "gpt2": gpt2,
        "qwen2": qwen2_config,
        "qwen3": qwen3_config,
        "llama": llama,
        "lfm2": lfm2,
        "qwen3_5": qwen3_5_config,
        "gemma3_text": lambda: tiny_config("gemma3_text", sliding_window=16),
        "phi3": lambda: tiny_config("phi3", sliding_window=None),
    }[family]()


def pair(family, reference):
    config = config_for(family)
    native = get_registration(family).model_class()(config)
    module = reference.reference_module(family)
    upstream = module.Model(module.ModelArgs.from_dict(asdict(config)))
    weights = dict(tree_flatten(native.parameters()))
    if family == "gemma3_text":
        upstream.sanitize(dict(weights))
    translated = []
    for name, expected in tree_flatten(upstream.parameters()):
        source = name
        if family == "gpt2":
            source = source.replace("model.", "transformer.", 1)
        elif family == "lfm2":
            source = source.replace(".conv.conv.weight", ".conv.conv")
        elif family == "qwen3_5":
            source = source.replace("language_model.model.", "language_model.")
        value = weights[source]
        if value.ndim == 2 and expected.ndim == 3:
            value = value[..., None]
        if family == "qwen3_5" and (
            source.endswith(
                (
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                )
            )
            or source == "language_model.norm.weight"
        ):
            value = 1 + value
        translated.append((name, value))
    upstream.load_weights(translated, strict=True)
    native.eval()
    upstream.eval()
    return native, upstream


@pytest.mark.parametrize("family", FAMILIES)
def test_whole_port_full_cached_chunks_and_reset(family, reference):
    mx.random.seed(817)
    native, upstream = pair(family, reference)
    if family == "llama":
        from mlx_one.models.shared.reference_math import initialize_rope

        rope_source = reference.reference_module("rope_utils")
        values = mx.random.normal((1, 2, 3, 64))
        for factor in (8.0, 32.0):
            scaling = dict(
                rope_type="llama3",
                factor=factor,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                original_max_position_embeddings=8192,
            )
            ported = initialize_rope(64, base=500000, scaling_config=scaling)
            original = rope_source.Llama3RoPE(64, base=500000, scaling_config=scaling)
            for offset in (0, 4096, 8192, 32768):
                assert (
                    reference.compare(
                        original(values, offset=offset), ported(values, offset=offset), "float32"
                    )["status"]
                    == "PASS"
                )
    ids = mx.array([[2, 3, 4, 5, 6, 7, 8]])
    full = native(ids).logits
    expected = upstream(ids)
    assert reference.compare(expected, full, "float32")["status"] == "PASS"
    cache_module = reference.reference_module("cache")
    for partition in ((7,), (6, 1), (1, 6), (3, 2, 2), (1,) * 7):
        nc, rc = native.make_cache(), cache_module.make_prompt_cache(upstream)
        start, native_parts, ref_parts = 0, [], []
        for length in partition:
            token_slice = ids[:, start : start + length]
            left, right = native(token_slice, cache=nc).logits, upstream(token_slice, cache=rc)
            mx.eval(left, right)
            native_parts.append(left)
            ref_parts.append(right)
            start += length
        actual, ref = mx.concatenate(native_parts, 1), mx.concatenate(ref_parts, 1)
        for left, right in ((ref, actual), (full, actual), (expected, ref)):
            result = reference.compare(left, right, "float32")
            assert result["status"] == "PASS", (family, partition, result)
        assert all(c.offset == 7 for c in nc)
        for c in nc:
            c.reset()
        assert reference.compare(full, native(ids, cache=nc).logits, "float32")["status"] == "PASS"


@pytest.mark.parametrize("family", [f for f in FAMILIES if f != "qwen3_5"])
def test_attention_contraction_against_numpy_float64(family, reference, monkeypatch):
    native, _ = pair(family, reference)
    original = mx.fast.scaled_dot_product_attention
    comparisons = []

    def checked(q, k, v, *, scale, mask=None, **kwargs):
        actual = original(q, k, v, scale=scale, mask=mask, **kwargs)
        mx.eval(q, k, v, actual)
        cq, ck, cv = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, v))
        ck, cv = (np.repeat(x, cq.shape[1] // ck.shape[1], axis=1) for x in (ck, cv))
        scores = (cq @ ck.swapaxes(-1, -2)) * scale
        if isinstance(mask, str):
            allowed = (
                np.arange(ck.shape[2])[None]
                <= np.arange(ck.shape[2] - cq.shape[2], ck.shape[2])[:, None]
            )
            scores = np.where(allowed, scores, -np.inf)
        elif mask is not None:
            cm = np.array(mask)
            scores = np.where(cm, scores, -np.inf) if cm.dtype == bool else scores + cm
        p = np.exp(scores - scores.max(-1, keepdims=True))
        expected = p / p.sum(-1, keepdims=True) @ cv
        comparisons.append(
            reference.compare(mx.array(expected.astype(np.float32)), actual, "float32")
        )
        return actual

    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", checked)
    mx.eval(native(mx.array([[2, 3, 4]])).logits)
    assert comparisons and all(row["status"] == "PASS" for row in comparisons)


def test_delta_recurrence_independently_matches_numpy_float64():
    from mlx_one.models.vision_language.qwen3_5.reference_delta import gated_delta_ops

    rng = np.random.default_rng(917)
    q, k, v = (rng.normal(size=(1, 5, 2, 4)).astype(np.float32) * 0.2 for _ in range(3))
    g = rng.uniform(0.5, 0.9, (1, 5, 2)).astype(np.float32)
    beta = rng.uniform(0.1, 0.8, (1, 5, 2)).astype(np.float32)
    state = np.zeros((1, 2, 4, 4), np.float64)
    outputs = []
    for t in range(5):
        state *= g[:, t, :, None, None]
        memory = np.sum(state * k[:, t, :, None, :], axis=-1)
        delta = (v[:, t] - memory) * beta[:, t, :, None]
        state += k[:, t, :, None, :] * delta[..., None]
        outputs.append(np.sum(state * q[:, t, :, None, :], axis=-1))
    result, actual_state = gated_delta_ops(*(mx.array(x) for x in (q, k, v, g, beta)))
    mx.eval(result, actual_state)
    np.testing.assert_allclose(np.array(result), np.stack(outputs, 1), atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(np.array(actual_state), state, atol=1e-7, rtol=1e-5)


def test_ported_decay_preserves_the_fp32_precision_fix():
    from mlx_one.models.vision_language.qwen3_5.reference_delta import compute_g
    from mlx_one.ops.numerical_trace import compare_tensors

    a_log = mx.array([-1.5, -0.7, 0.3, 1.2], dtype=mx.bfloat16)
    a = mx.array([0.0, 1.0, 2.0, 3.0], dtype=mx.bfloat16)
    bias = mx.array([0.1, -0.2, 0.3, -0.4], dtype=mx.bfloat16)
    result = compute_g(a_log, a, bias)
    mx.eval(result)
    ca, cb, cc = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (a_log, a, bias))
    expected = np.exp(-np.exp(ca) * np.logaddexp(cb + cc, 0))
    assert result.dtype == mx.float32
    assert compare_tensors(expected, np.array(result), dtype="float32")["status"] == "PASS"
