"""Backend-free contracts for the native Whisper ASR vertical slice."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from click.testing import CliRunner

from mlx_one.audio.decoding import (
    compression_ratio,
    dynamic_time_warp,
    initial_tokens,
    median_filter,
    timestamp_segments,
)
from mlx_one.audio.evaluation import (
    WHISPER_CER_PROFILE,
    WHISPER_WER_PROFILE,
    character_error_rate,
    load_asr_records,
    normalize_transcript,
    score_asr_records,
    word_error_rate,
)
from mlx_one.audio.schemas import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    WhisperDecodeOptions,
)
from mlx_one.cli import main
from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration
from mlx_one.models.audio.whisper.config import WhisperConfig
from mlx_one.models.audio.whisper.loading import WhisperLoadError, load_whisper
from mlx_one.models.audio.whisper.processing import AudioProcessingError, decode_audio
from mlx_one.models.audio.whisper.tokenizer import WhisperTokenizer, bytes_to_unicode
from mlx_one.models.audio.whisper.weights import sanitize_weights, weight_contract


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]

    def transpose(self, *axes: int) -> Shaped:
        return Shaped(tuple(self.shape[index] for index in axes))


def tiny_config(**updates: object) -> WhisperConfig:
    values = {
        "vocab_size": 64,
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


def test_official_tiny_and_turbo_profiles_share_one_config() -> None:
    tiny = WhisperConfig()
    assert (tiny.d_model, tiny.encoder_layers, tiny.decoder_layers) == (384, 4, 4)
    assert (tiny.num_mel_bins, tiny.input_frames) == (80, 3000)
    turbo = WhisperConfig.from_dict(
        {
            "vocab_size": 51866,
            "num_mel_bins": 128,
            "d_model": 1280,
            "encoder_layers": 32,
            "encoder_attention_heads": 20,
            "decoder_layers": 4,
            "decoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "decoder_ffn_dim": 5120,
            "pad_token_id": 50257,
            "bos_token_id": 50257,
            "eos_token_id": 50257,
            "decoder_start_token_id": 50258,
            "begin_suppress_tokens": [220, 50256],
        }
    )
    assert (turbo.encoder_layers, turbo.decoder_layers, turbo.num_mel_bins) == (32, 4, 128)


def test_whisper_registry_is_lazy_and_declares_asr_capabilities() -> None:
    registration = get_registration("whisper")
    assert registration.config_class() is WhisperConfig
    assert registration.modality == "asr"
    assert registration.capabilities >= {
        "forward",
        "cache",
        "transcribe",
        "language-detection",
        "segment-timestamps",
        "word-timestamps",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "wav2vec2"}, "model_type"),
        ({"num_mel_bins": 96}, "80 or 128"),
        ({"d_model": 7}, "divisible"),
        ({"activation_function": "silu"}, "GELU"),
        ({"dropout": 1.0}, r"\[0, 1\)"),
        ({"encoder_layerdrop": 0.1}, "LayerDrop"),
        ({"apply_spec_augment": True}, "SpecAugment"),
        ({"tie_word_embeddings": False}, "tied"),
        ({"is_encoder_decoder": False}, "encoder-decoder"),
        ({"median_filter_width": 4}, "odd"),
        ({"eos_token_id": 64}, "vocabulary"),
        ({"begin_suppress_tokens": [64]}, "vocabulary"),
    ],
)
def test_whisper_config_rejects_unsupported_values(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        tiny_config(**updates)


def test_whisper_weight_contract_is_exact_and_transposes_convolutions() -> None:
    config = tiny_config()
    contract = weight_contract(config)
    values = {name: Shaped(shape) for name, shape in contract.expected.items()}
    contract.validate(values)
    assert contract.expected["model.encoder.conv1.weight"] == (8, 3, 80)
    assert contract.expected["model.decoder.layers.0.encoder_attn.k_proj.weight"] == (8, 8)
    source = {
        "model.encoder.conv1.weight": Shaped((8, 80, 3)),
        "model.encoder.conv2.weight": Shaped((8, 8, 3)),
        "proj_out.weight": Shaped((64, 8)),
    }
    cleaned = sanitize_weights(source, config)
    assert cleaned["model.encoder.conv1.weight"].shape == (8, 3, 80)
    assert "proj_out.weight" not in cleaned
    invalid = dict(values)
    invalid["unknown.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)


def tokenizer() -> WhisperTokenizer:
    mapping = bytes_to_unicode()
    vocabulary = {character: byte for byte, character in mapping.items()}
    special = {
        "<|endoftext|>": 256,
        "<|startoftranscript|>": 257,
        "<|en|>": 258,
        "<|fr|>": 259,
        "<|transcribe|>": 260,
        "<|translate|>": 261,
        "<|notimestamps|>": 262,
        "<|nospeech|>": 263,
        "<|startofprev|>": 264,
        "<|0.00|>": 265,
        "<|0.02|>": 266,
        "<|0.04|>": 267,
    }
    return WhisperTokenizer(vocabulary, [], special)


def test_whisper_byte_bpe_special_tokens_and_word_splitting() -> None:
    value = tokenizer()
    text = " hello café!"
    encoded = value.encode(text)
    assert value.decode(encoded) == text
    assert value.encode("<|en|>", allow_special=True) == [258]
    words, groups = value.split_to_word_tokens(encoded)
    assert "".join(words) == text
    assert sum(len(group) for group in groups) == len(encoded)
    punctuated_words, _ = value.split_to_word_tokens(value.encode(" hello! world"))
    assert punctuated_words[0].endswith("!")
    assert initial_tokens(value, language="en", task="transcribe", without_timestamps=True) == [
        257,
        258,
        260,
        262,
    ]


def test_timestamp_dtw_median_and_compression_helpers() -> None:
    value = tokenizer()
    segments = timestamp_segments((265, *value.encode(" hi"), 267), value)
    assert segments == [(0.0, 0.04, tuple(value.encode(" hi")))]
    assert median_filter([[9.0, 1.0, 5.0]], 3) == [[9.0, 5.0, 5.0]]
    text, time = dynamic_time_warp([[0.0, 2.0], [2.0, 0.0]])
    assert (text, time) == ([0, 1], [0, 1])
    assert compression_ratio("repeat " * 30) > compression_ratio("short")


def test_decode_options_and_transcription_result_are_json_serializable() -> None:
    with pytest.raises(ValueError, match="temperatures"):
        WhisperDecodeOptions(temperatures=())
    word = TranscriptionWord(" hello", 0.0, 0.4, 0.9)
    segment = TranscriptionSegment(0, 0, 0.0, 0.4, "hello", (1,), 0.0, -0.1, 1.0, 0.0, (word,))
    result = TranscriptionResult("hello", "en", (segment,), 0.4, "openai/whisper-tiny")
    payload = json.loads(result.to_json())
    assert payload["segments"][0]["words"][0]["word"] == " hello"


def test_asr_normalization_wer_cer_and_dataset_paths(tmp_path) -> None:
    assert normalize_transcript("  HéLLO,  WORLD! ") == "héllo world"
    assert word_error_rate("one two three", "one four three") == pytest.approx(1 / 3)
    assert character_error_rate("abc", "adc") == pytest.approx(1 / 3)
    dataset = tmp_path / "asr.json"
    dataset.write_text(json.dumps([{"audio": "clip.wav", "reference": "hello"}]))
    records = load_asr_records(dataset)
    assert records[0]["audio"] == str((tmp_path / "clip.wav").resolve())
    assert score_asr_records(records, ["hello"], WHISPER_WER_PROFILE)["value"] == 0
    assert score_asr_records(records, ["hallo"], WHISPER_CER_PROFILE)["value"] > 0


def test_loader_rejects_pickle_only_checkpoint(tmp_path) -> None:
    config = tiny_config()
    (tmp_path / "config.json").write_text(json.dumps(config.__dict__))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"feature_size": 80}))
    (tmp_path / "vocab.json").write_text(json.dumps({"a": 0}))
    (tmp_path / "added_tokens.json").write_text("{}")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    (tmp_path / "pytorch_model.bin").write_bytes(b"unsafe")
    with pytest.raises(WhisperLoadError, match="pickle"):
        load_whisper(tmp_path)


def test_decode_audio_rejects_missing_path() -> None:
    with pytest.raises(AudioProcessingError, match="does not exist"):
        decode_audio("missing-audio.wav")


def test_transcribe_cli_renders_plain_text_and_json(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fixture")
    result = TranscriptionResult("hello", "en", (), 1.0, "model")
    monkeypatch.setattr("mlx_one.cli.transcribe", lambda *args, **kwargs: result)
    runner = CliRunner()
    plain = runner.invoke(main, ["transcribe", "model", str(audio)])
    structured = runner.invoke(main, ["transcribe", "model", str(audio), "--json-output"])
    assert plain.exit_code == 0 and plain.output.strip() == "hello"
    assert structured.exit_code == 0 and json.loads(structured.output)["language"] == "en"
