"""Opt-in Metal component, decoder isolation and offline boundary checks."""

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.audio.moonshine import AudioResourceLimitError
from mlx_one.audio.parakeet import greedy_decode
from mlx_one.models.audio.parakeet.config import ParakeetConfig
from mlx_one.models.audio.parakeet.model import ParakeetTDT, RelativeAttention
from mlx_one.models.audio.parakeet.processing import ParakeetProcessor, preprocess
from mlx_one.models.audio.parakeet.weights import weight_contract


def tiny_model():
    config = ParakeetConfig(
        vocab_size=9,
        blank_token_id=8,
        decoder_hidden_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_mel_bins=16,
        subsampling_conv_channels=2,
    )
    return ParakeetTDT(config)


def test_model_parameters_match_contract_and_mask_lengths():
    model = tiny_model()
    weight_contract(model.config).validate(dict(tree_flatten(model.parameters())))
    output, lengths = model.encode(mx.ones((1, 25, 16)), mx.array([24]))
    assert output.shape == (1, 4, 4)
    assert lengths.tolist() == [3]
    assert mx.all(mx.isfinite(output)).item()


def test_relative_shift_matches_reference_reshape():
    x = mx.arange(30).reshape(1, 2, 3, 5)
    padded = mx.concatenate((mx.zeros((1, 2, 3, 1)), x), axis=-1)
    expected = padded.reshape(1, 2, 6, 3)[:, :, 1:].reshape(1, 2, 3, 5)
    assert mx.array_equal(RelativeAttention.relative_shift(x), expected).item()


def test_predictor_state_is_request_local():
    model = tiny_model()
    for index in range(model.config.num_decoder_layers):
        for side in ("ih", "hh"):
            setattr(model.decoder.lstm, f"weight_{side}_l{index}", mx.random.normal((16, 4)))
    first, state = model.decoder(2)
    second, _ = model.decoder(4, state)
    repeated, repeated_state = model.decoder(2)
    assert mx.array_equal(first, repeated).item()
    assert mx.array_equal(state[0], repeated_state[0]).item()
    assert not mx.array_equal(first, second).item()


def test_silent_preprocessing_is_finite_and_last_frame_masked():
    features, count = preprocess(mx.zeros(1600), ParakeetProcessor(feature_size=16))
    assert features.shape == (1, 11, 16)
    assert count == 10
    assert mx.all(mx.isfinite(features)).item()
    assert mx.array_equal(features, mx.zeros_like(features)).item()
    assert mx.array_equal(features[:, -1], mx.zeros((1, 16))).item()
    with pytest.raises(ValueError, match="320"):
        preprocess(mx.ones(319), ParakeetProcessor(feature_size=16))


def test_blank_duration_zero_advances_and_budget_refuses_truncation():
    model = tiny_model()
    model.joint.head.weight = mx.zeros_like(model.joint.head.weight)
    model.joint.head.bias = mx.zeros_like(model.joint.head.bias)
    model.joint.head.bias[8] = 20.0
    model.joint.head.bias[9] = 20.0
    encoded = mx.zeros((1, 3, 4))
    tokens, durations = greedy_decode(model, encoded, 3, {})
    assert tokens == (8, 8, 8, 8)
    assert durations == (0, 1, 1, 1)
    with pytest.raises(AudioResourceLimitError, match="refusing truncation"):
        greedy_decode(model, encoded, 3, {}, limit=1)


def test_zero_duration_lexical_loop_has_explicit_limit_and_cancellation():
    model = tiny_model()
    model.joint.head.weight = mx.zeros_like(model.joint.head.weight)
    model.joint.head.bias = mx.zeros_like(model.joint.head.bias)
    model.joint.head.bias[1] = 20.0
    model.joint.head.bias[9] = 20.0
    with pytest.raises(AudioResourceLimitError):
        greedy_decode(model, mx.zeros((1, 2, 4)), 2, {}, limit=3)

    def cancelled():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        greedy_decode(model, mx.zeros((1, 2, 4)), 2, {}, check=cancelled)
