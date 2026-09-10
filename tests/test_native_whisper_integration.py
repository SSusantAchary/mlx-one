"""Opt-in pinned real-checkpoint smoke gates for native Whisper."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_WHISPER_INTEGRATION") != "1":
    pytest.skip(
        "set MLX_ONE_RUN_WHISPER_INTEGRATION=1 to run pinned Whisper checkpoints",
        allow_module_level=True,
    )

import mlx.core as mx

from mlx_one.models.audio.whisper.loading import load_whisper

TINY_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
TURBO_REVISION = "60be3615a4d667e1258e8ad29130467587c489aa"


@pytest.mark.parametrize(
    ("model_id", "revision", "dimensions"),
    [
        ("openai/whisper-tiny", TINY_REVISION, (384, 4, 4, 80)),
        ("openai/whisper-large-v3-turbo", TURBO_REVISION, (1280, 32, 4, 128)),
    ],
)
def test_pinned_whisper_checkpoint_contracts(
    model_id: str, revision: str, dimensions: tuple[int, int, int, int]
) -> None:
    bundle = load_whisper(model_id, revision=revision)
    config = bundle.model.config
    assert (
        config.d_model,
        config.encoder_layers,
        config.decoder_layers,
        config.num_mel_bins,
    ) == dimensions


@pytest.mark.parametrize(
    ("model_id", "revision", "mel_bins", "hidden", "vocabulary"),
    [
        ("openai/whisper-tiny", TINY_REVISION, 80, 384, 51865),
        ("openai/whisper-large-v3-turbo", TURBO_REVISION, 128, 1280, 51866),
    ],
)
def test_pinned_whisper_executes_encoder_and_decoder(
    model_id: str,
    revision: str,
    mel_bins: int,
    hidden: int,
    vocabulary: int,
) -> None:
    bundle = load_whisper(model_id, revision=revision)
    encoded, _ = bundle.model.encode(mx.zeros((1, mel_bins, 3000)))
    output = bundle.model(
        None,
        mx.array([[bundle.model.config.decoder_start_token_id]]),
        encoder_outputs=encoded,
    )
    mx.eval(encoded, output.logits)
    assert encoded.shape == (1, 1500, hidden)
    assert output.logits.shape == (1, 1, vocabulary)
