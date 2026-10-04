"""Opt-in native Moonshine execution, session isolation and cancellation tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from mlx_one.audio import ASRDecodeOptions, SpeechTranscript, transcribe
from mlx_one.audio.moonshine import AudioResourceLimitError
from mlx_one.models.audio.moonshine.config import MoonshineConfig
from mlx_one.models.audio.moonshine.loading import LoadedMoonshine
from mlx_one.models.audio.moonshine.model import MoonshineForConditionalGeneration


def model():
    return MoonshineForConditionalGeneration(
        MoonshineConfig(
            hidden_size=8,
            intermediate_size=16,
            encoder_num_hidden_layers=2,
            decoder_num_hidden_layers=2,
            encoder_num_attention_heads=2,
            decoder_num_attention_heads=2,
            partial_rotary_factor=1.0,
            vocab_size=8,
        )
    )


def bundle(native=None):
    return LoadedMoonshine(
        native or model(),
        Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>")),
        {"max_length": 194},
        Path("/fixture"),
        "pin",
    )


def test_cached_decode_matches_full_and_cross_cache_isolation() -> None:
    native = model()
    encoded = native.encode(mx.zeros((1, 4000)))
    tokens = mx.array([[1, 3, 4, 5]])
    full = native.decode(tokens, encoded).logits
    cache = native.make_cache()
    pieces = [native.decode(tokens[:, i : i + 1], encoded, cache=cache).logits for i in range(4)]
    cached = mx.concatenate(pieces, axis=1)
    mx.eval(full, cached)
    assert mx.max(mx.abs(full - cached)).item() < 1e-4
    previous = cache[0].cross_keys
    native.decode(mx.array([[6]]), encoded, cache=cache)
    assert cache[0].cross_keys is previous
    other = native.make_cache()
    native.decode(mx.array([[1]]), encoded, cache=other)
    cache[0].reset()
    assert other[0].offset == 1 and other[0].cross_keys is not None


def test_empty_waveform_has_no_fabricated_timestamps() -> None:
    result = transcribe(bundle(), [], sample_rate=16000)
    assert isinstance(result, SpeechTranscript)
    assert result.text == "" and result.duration == 0
    assert result.timestamps is None and result.confidence is None and result.language is None


def test_nonfinite_stereo_and_short_waveforms_are_rejected() -> None:
    loaded = bundle()
    for audio in ([[0.0, 0.0]], [float("nan")] * 900, [0.0] * 894):
        with pytest.raises(ValueError):
            transcribe(loaded, audio, sample_rate=16000)


def test_decoder_limit_and_cancellation_release_only_request_caches(monkeypatch) -> None:
    native = model()
    cache = native.make_cache()
    monkeypatch.setattr(native, "make_cache", lambda: cache)

    def decode(input_ids, encoded, **kwargs):
        # Exercise allocation, then force a non-EOS token.
        native.model.decoder(input_ids, encoded, kwargs["cache"])
        return SimpleNamespace(logits=mx.array([[[0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]]]))

    monkeypatch.setattr(native, "decode", decode)
    loaded = bundle(native)
    with pytest.raises(AudioResourceLimitError, match="truncated"):
        transcribe(
            loaded, mx.zeros(4000), sample_rate=16000, options=ASRDecodeOptions(max_tokens=1)
        )
    assert all(item.offset == 0 and item.cross_keys is None for item in cache)
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) > 2

    with pytest.raises(RuntimeError, match="cancelled"):
        transcribe(loaded, mx.zeros(4000), sample_rate=16000, cancelled=cancelled)
    assert all(item.offset == 0 and item.cross_keys is None for item in cache)
    assert loaded.model.model.encoder.conv1.weight.shape == (8, 127, 1)
