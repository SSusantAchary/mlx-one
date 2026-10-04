"""Opt-in Metal Qwen architecture, request cache and lifecycle checks."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from benchmarks.qwen_audio_rounding_candidate import (
    AudioConv2d,
    AudioLayerNorm,
    install_candidate,
)
from benchmarks.qwen_audio_rounding_candidate import (
    candidate_attention as attention,
)
from mlx_one.audio import open_audio_stream, transcribe
from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig, audio_token_length
from mlx_one.models.audio.qwen3_asr.loading import LoadedQwenAudio
from mlx_one.models.audio.qwen3_asr.model import QwenASRModel


def model():
    return QwenASRModel(
        QwenASRConfig.from_dict(
            {
                "model_type": "qwen3_asr",
                "timestamp_token_id": 6,
                "thinker_config": {
                    "model_type": "qwen3_asr",
                    "audio_token_id": 3,
                    "audio_start_token_id": 4,
                    "audio_end_token_id": 5,
                    "audio_config": {
                        "d_model": 8,
                        "encoder_layers": 2,
                        "encoder_attention_heads": 2,
                        "encoder_ffn_dim": 16,
                        "output_dim": 8,
                        "downsample_hidden_size": 2,
                    },
                    "text_config": {
                        "model_type": "qwen3",
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 4,
                        "vocab_size": 32,
                        "rope_theta": 1000000.0,
                        "tie_word_embeddings": True,
                    },
                },
            }
        )
    )


def bundle():
    return LoadedQwenAudio(
        model(), SimpleNamespace(), {"eos_token_id": [1]}, "", Path("/fixture"), "pin"
    )


def test_audio_bf16_rounds_after_bias_and_affine_without_changing_weights():
    mx.random.seed(17)
    conv = AudioConv2d(1, 3, 3, stride=2, padding=1)
    conv.weight = conv.weight.astype(mx.bfloat16)
    conv.bias = mx.array([0.123, -0.321, 0.456], dtype=mx.bfloat16)
    x = mx.random.normal((2, 8, 9, 1)).astype(mx.bfloat16)
    expected = (
        mx.conv2d(x.astype(mx.float32), conv.weight.astype(mx.float32), stride=2, padding=1)
        + conv.bias.astype(mx.float32)
    ).astype(mx.bfloat16)
    assert mx.array_equal(conv(x), expected).item()
    assert conv.weight.dtype == conv.bias.dtype == mx.bfloat16
    norm = AudioLayerNorm(9)
    norm.weight = mx.random.uniform(shape=(9,)).astype(mx.bfloat16)
    norm.bias = mx.random.normal((9,)).astype(mx.bfloat16)
    value = mx.random.normal((2, 5, 9)).astype(mx.bfloat16)
    expected = mx.fast.layer_norm(
        value.astype(mx.float32),
        norm.weight.astype(mx.float32),
        norm.bias.astype(mx.float32),
        norm.eps,
    ).astype(mx.bfloat16)
    assert mx.array_equal(norm(value), expected).item()
    assert norm.weight.dtype == norm.bias.dtype == mx.bfloat16


def test_bf16_attention_key_blocks_and_grouped_heads():
    # Cross the pinned reference's 512-key boundary with exactly representable
    # values. Each KV head must be replicated contiguously into its query group.
    q = mx.zeros((1, 4, 2, 8), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 513, 8), dtype=mx.bfloat16)
    v = mx.concatenate((mx.full((1, 1, 513, 8), 1.5), mx.full((1, 1, 513, 8), 2.0)), axis=1)
    v = v.astype(mx.bfloat16)
    result = attention(q, k, v, scale=8**-0.5, cpu_flash=True)
    expected = mx.concatenate((mx.full((1, 2, 2, 8), 1.5), mx.full((1, 2, 2, 8), 2.0)), axis=1)
    assert result.dtype == mx.bfloat16
    assert mx.array_equal(result, expected.astype(mx.bfloat16)).item()
    mask = mx.zeros((2, 513))
    mask[0, -1] = -float("inf")
    assert mx.array_equal(
        attention(q, k, v, scale=8**-0.5, mask=mask, cpu_flash=True), result
    ).item()


def test_rounding_candidate_is_opt_in_and_request_model_local():
    first, second = model(), model()
    original_conv, original_norm = (
        type(second.audio_tower.conv2d1),
        type(second.audio_tower.ln_post),
    )
    install_candidate(first)
    assert isinstance(first.audio_tower.conv2d1, AudioConv2d)
    assert isinstance(first.audio_tower.ln_post, AudioLayerNorm)
    assert type(second.audio_tower.conv2d1) is original_conv
    assert type(second.audio_tower.ln_post) is original_norm
    assert original_conv is not AudioConv2d and original_norm is not AudioLayerNorm


def test_text_math_candidate_gqa_mask_and_dtype():
    from benchmarks.qwen_audio_text_candidate import math_attention

    q = mx.zeros((1, 4, 2, 8), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 3, 8), dtype=mx.bfloat16)
    v = mx.concatenate((mx.full((1, 1, 3, 8), 1.5), mx.full((1, 1, 3, 8), 2.0)), axis=1)
    mask = mx.array([[0.0, -float("inf"), -float("inf")], [0.0, 0.0, -float("inf")]])
    result = math_attention(q, k, v.astype(mx.bfloat16), scale=8**-0.5, mask=mask)
    assert result.dtype == mx.bfloat16
    assert mx.array_equal(result[:, :2], mx.full((1, 2, 2, 8), 1.5)).item()
    assert mx.array_equal(result[:, 2:], mx.full((1, 2, 2, 8), 2.0)).item()
    empty = math_attention(q, k, v, scale=8**-0.5, mask=mx.full((2, 3), -float("inf")))
    assert mx.array_equal(empty, mx.zeros_like(empty)).item()


def test_text_candidate_does_not_modify_shared_classes_or_other_models():
    from benchmarks.qwen_audio_text_candidate import CandidateTextAttention
    from benchmarks.qwen_audio_text_candidate import install_candidate as install_text

    first, second = model(), model()
    original = type(second.model.layers[0].self_attn)
    install_text(first)
    assert isinstance(first.model.layers[0].self_attn, CandidateTextAttention)
    assert type(second.model.layers[0].self_attn) is original
    assert original is not CandidateTextAttention


def test_compensated_linear_preserves_checkpoint_dtype_and_crosses_tiles():
    from benchmarks.qwen_audio_text_candidate import CompensatedLinear

    layer = CompensatedLinear(513, 2)
    layer.weight = mx.ones((2, 513), dtype=mx.bfloat16)
    layer.bias = mx.array([0.0, 2.0], dtype=mx.bfloat16)
    x = mx.ones((1, 513), dtype=mx.bfloat16)
    actual = layer(x)
    assert actual.dtype == layer.weight.dtype == layer.bias.dtype == mx.bfloat16
    expected = mx.array([[513.0, 515.0]]).astype(mx.bfloat16)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(layer(x.astype(mx.float32)), mx.array([[513.0, 515.0]])).item()


def test_features_cached_full_decoder_and_isolation():
    native = model()
    features = mx.zeros((128, 101))
    ids = mx.array([[4] + [3] * audio_token_length(101) + [5, 7, 8]])
    full = native(ids, features=features)
    caches = native.make_cache()
    prefill = native(ids[:, :-1], features=features, cache=caches)
    tail = native(ids[:, -1:], cache=caches)
    assert mx.allclose(full, mx.concatenate((prefill, tail), axis=1), atol=1e-4, rtol=1e-4).item()
    other = native.make_cache()
    native(mx.array([[1]]), cache=other)
    for state in caches:
        state.reset()
    assert other[0].offset == 1
    with pytest.raises(ValueError, match="exactly match"):
        native(mx.array([[4, 3, 5]]), features=features)
    with pytest.raises(ValueError, match="only at prefill"):
        native(ids, features=features, cache=other)


def test_empty_waveform_and_rate_validation():
    loaded = bundle()
    result = transcribe(loaded, [], sample_rate=16000)
    assert result.text == "" and result.language is None and result.timestamps is None
    for audio, rate in (([0.0], None), ([0.0], 8000), ([float("nan")], 16000), ([[0.0]], 16000)):
        with pytest.raises(ValueError):
            transcribe(loaded, audio, sample_rate=rate)


def test_stream_owns_pcm_not_shared_model(monkeypatch):
    loaded = bundle()
    seen = []

    def decode(bundle, waveform, **kwargs):
        seen.append(waveform.shape[0])
        return "language English<asr_text>text", ()

    monkeypatch.setattr("mlx_one.audio.streaming.decode_qwen", decode)
    with open_audio_stream(loaded, sample_rate=16000) as first:
        with open_audio_stream(loaded, sample_rate=16000) as second:
            assert first.push([0.0] * 32000)[0].text == "text"
            first.close()
            second.push([0.0] * 1000)
            assert second.finish().text == "text"
    assert seen == [32000, 1000] and loaded.model is not None
