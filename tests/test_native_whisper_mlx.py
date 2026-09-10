"""Opt-in tiny MLX execution tests for native Whisper."""

from __future__ import annotations

import os
import struct
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.audio.decoding import align_words
from mlx_one.models.audio.whisper.config import WhisperConfig
from mlx_one.models.audio.whisper.model import WhisperForConditionalGeneration
from mlx_one.models.audio.whisper.processing import (
    decode_audio,
    log_mel_spectrogram,
    mel_filterbank,
    pad_or_trim,
)
from mlx_one.models.audio.whisper.tokenizer import WhisperTokenizer, bytes_to_unicode


def config(**updates: object) -> WhisperConfig:
    values = {
        "vocab_size": 300,
        "num_mel_bins": 80,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "decoder_layers": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 16,
        "decoder_ffn_dim": 16,
        "d_model": 8,
        "max_source_positions": 4,
        "max_target_positions": 12,
        "pad_token_id": 1,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "decoder_start_token_id": 2,
        "begin_suppress_tokens": [3, 4],
    }
    values.update(updates)
    return WhisperConfig.from_dict(values)


def test_whisper_encoder_decoder_attention_and_hidden_shapes() -> None:
    model = WhisperForConditionalGeneration(config())
    features = mx.zeros((1, 80, 8))
    output = model(
        features,
        mx.array([[2, 5, 6]]),
        output_hidden_states=True,
        output_attentions=True,
    )
    mx.eval(output.logits, output.encoder_last_hidden_state, *output.cross_attentions)
    assert output.logits.shape == (1, 3, 300)
    assert output.encoder_last_hidden_state.shape == (1, 4, 8)
    assert output.decoder_last_hidden_state.shape == (1, 3, 8)
    assert len(output.encoder_hidden_states) == 3
    assert len(output.decoder_hidden_states) == 3
    assert output.decoder_attentions[0].shape == (1, 2, 3, 3)
    assert output.cross_attentions[0].shape == (1, 2, 3, 4)


def test_whisper_cached_decode_matches_full_sequence_and_reuses_cross_cache() -> None:
    model = WhisperForConditionalGeneration(config())
    features = mx.zeros((1, 80, 8))
    encoder, _ = model.encode(features)
    tokens = mx.array([[2, 5, 6, 7]])
    full = model(None, tokens, encoder_outputs=encoder)
    caches = model.make_cache()
    pieces = [
        model(
            None,
            tokens[:, index : index + 1],
            encoder_outputs=encoder,
            cache=caches,
            use_cache=True,
        ).logits
        for index in range(tokens.shape[1])
    ]
    cached = mx.concatenate(pieces, axis=1)
    first_cross = caches[0].cross_keys
    model(
        None,
        mx.array([[8]]),
        encoder_outputs=encoder,
        cache=caches,
        use_cache=True,
    )
    mx.eval(full.logits, cached)
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4
    assert [item.offset for item in caches] == [5, 5]
    assert caches[0].cross_keys is first_cross


@pytest.mark.parametrize("mel_bins", [80, 128])
def test_whisper_mel_profiles_and_conv_subsampling(mel_bins: int) -> None:
    model = WhisperForConditionalGeneration(config(num_mel_bins=mel_bins))
    encoded, _ = model.encode(mx.zeros((1, mel_bins, 8)))
    filters = mel_filterbank(mel_bins)
    mx.eval(encoded, filters)
    assert encoded.shape == (1, 4, 8)
    assert filters.shape == (mel_bins, 201)


def test_whisper_log_mel_padding_and_dynamic_range_are_finite() -> None:
    waveform = pad_or_trim(mx.zeros((200,)), 800)
    features = log_mel_spectrogram(waveform, num_mel_bins=80)
    mx.eval(features)
    assert waveform.shape == (800,)
    assert features.shape[0] == 80
    assert mx.all(mx.isfinite(features)).item()


def test_whisper_word_alignment_returns_monotonic_boundaries() -> None:
    mapping = bytes_to_unicode()
    tokenizer = WhisperTokenizer(
        {character: byte for byte, character in mapping.items()}, [], {}
    )
    tokens = tuple(tokenizer.encode(" hi"))
    attention = mx.zeros((1, 2, len(tokens), 8))
    diagonal = mx.arange(len(tokens)) % 8
    attention = attention.at[0, 0, mx.arange(len(tokens)), diagonal].add(5.0)
    attention = attention.at[0, 1, mx.arange(len(tokens)), diagonal].add(5.0)
    words = align_words(
        tokens,
        (attention,),
        tokenizer,
        alignment_heads=[[0, 0], [0, 1]],
        median_width=3,
        token_logprobs=tuple([-0.1] * len(tokens)),
    )
    assert words
    assert all(word.start <= word.end for word in words)
    assert all(0 < word.probability < 1 for word in words)


def test_whisper_ffmpeg_decode_uses_argument_array(monkeypatch, tmp_path) -> None:
    source = tmp_path / "sample.any"
    source.write_bytes(b"fixture")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=struct.pack("<3f", 0.0, 0.5, -0.5),
            stderr=b"",
        )

    monkeypatch.setattr("shutil.which", lambda value: "/fake/ffmpeg")
    monkeypatch.setattr("subprocess.run", fake_run)
    waveform = decode_audio(source)
    mx.eval(waveform)
    assert waveform.shape == (3,)
    assert captured["command"] == [
        "/fake/ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(source),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "16000",
        "-",
    ]
    assert captured["kwargs"] == {"check": False, "capture_output": True}
