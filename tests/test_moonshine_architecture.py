"""Network-free original Moonshine configuration, assets and API contracts."""

import json
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from mlx_one.audio import ASRDecodeOptions, SpeechTranscript, WhisperDecodeOptions
from mlx_one.audio.moonshine import transcribe_moonshine
from mlx_one.cli import main
from mlx_one.core.config import ConfigError
from mlx_one.core.registry import get_registration
from mlx_one.core.weights import WeightContractError
from mlx_one.inspection import _infer_modality
from mlx_one.models.audio.moonshine.config import MoonshineConfig, encoded_length
from mlx_one.models.audio.moonshine.loading import (
    MoonshineLoadError,
    load_moonshine,
    validate_generation,
    validate_processor,
)
from mlx_one.models.audio.moonshine.weights import sanitize_weights, weight_contract
from mlx_one.schemas import Modality


def test_import_and_registry_are_backend_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mlx_one.audio; "
                "from mlx_one.core.registry import get_registration; "
                "r=get_registration('moonshine'); r.config_class(); r.loader(); "
                "assert 'mlx.core' not in sys.modules; assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    registration = get_registration("moonshine")
    assert registration.modality == "asr"
    assert registration.supports("transcribe")
    assert not registration.supports("train")
    assert not registration.supports("align")
    assert _infer_modality(None, {"model_type": "moonshine"}, (), None) == Modality.ASR


def test_tiny_base_configuration_and_valid_convolution_lengths() -> None:
    tiny = MoonshineConfig()
    base = MoonshineConfig(
        hidden_size=416,
        intermediate_size=1664,
        encoder_num_hidden_layers=8,
        decoder_num_hidden_layers=8,
        partial_rotary_factor=0.62,
    )
    assert int(tiny.hidden_size / 8 * tiny.partial_rotary_factor) == 32
    assert int(base.hidden_size / 8 * base.partial_rotary_factor) == 32
    assert encoded_length(894) == 0
    assert encoded_length(895) == 1
    assert encoded_length(16000) == 40
    assert encoded_length(0) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "moonshine_streaming"},
        {"hidden_size": 289},
        {"encoder_num_key_value_heads": 2},
        {"partial_rotary_factor": 0.53},
        {"partial_rotary_factor": float("nan")},
        {"quantization_config": {"bits": 4}},
        {"rope_scaling": {"type": "linear", "factor": 2}},
        {"encoder_hidden_act": "silu"},
        {"attention_bias": True},
        {"tie_word_embeddings": False},
        {"attention_dropout": 0.1},
        {"eos_token_id": 40000},
        {"decoder_num_hidden_layers": True},
    ],
)
def test_unsupported_architectures_fail_explicitly(changes) -> None:
    with pytest.raises(ConfigError):
        MoonshineConfig.from_dict(changes)


@dataclass
class Shaped:
    shape: tuple[int, ...]

    def transpose(self, *axes):
        return Shaped(tuple(self.shape[axis] for axis in axes))


def test_exact_tensor_contract_and_conv_transpose() -> None:
    config = MoonshineConfig()
    contract = weight_contract(config)
    assert len(contract.expected) == 160
    assert contract.expected["model.decoder.layers.0.mlp.fc1.weight"] == (2304, 288)
    assert "model.encoder.conv1.bias" not in contract.expected
    values = {name: Shaped(shape) for name, shape in contract.expected.items()}
    contract.validate(values)
    mapped = sanitize_weights({"model.encoder.conv1.weight": Shaped((288, 1, 127))}, config)
    assert mapped["model.encoder.conv1.weight"].shape == (288, 127, 1)
    with pytest.raises(WeightContractError, match="unexpected"):
        contract.validate({**values, "bad.weight": Shaped((1,))})
    values.pop("model.decoder.norm.weight")
    with pytest.raises(WeightContractError, match="missing"):
        contract.validate(values)


def test_processor_and_generation_contracts_reject_altered_settings() -> None:
    data = {
        "feature_extractor_type": "Wav2Vec2FeatureExtractor",
        "sampling_rate": 16000,
        "feature_size": 1,
        "padding_side": "right",
        "padding_value": 0.0,
        "do_normalize": False,
        "return_attention_mask": True,
    }
    validate_processor(data)
    with pytest.raises(MoonshineLoadError, match="preprocessing"):
        validate_processor({**data, "do_normalize": True})
    validate_generation({"max_length": 194}, MoonshineConfig())
    for change in (
        {"eos_token_id": 3},
        {"num_beams": 5},
        {"forced_decoder_ids": [[0, 1]]},
        {"max_length": True},
    ):
        with pytest.raises(MoonshineLoadError):
            validate_generation(change, MoonshineConfig())


def test_missing_assets_do_not_initialize_backend(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "moonshine"}))
    with pytest.raises(OSError):
        load_moonshine(tmp_path, offline=True)
    with pytest.raises(MoonshineLoadError):
        load_moonshine(tmp_path / "missing", offline=True)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"sample_rate": None}, "explicit sample_rate"),
        ({"sample_rate": 8000}, "16000"),
        ({"sample_rate": True}, "16000"),
        ({"options": WhisperDecodeOptions()}, "ASRDecodeOptions"),
        ({"task": "translate"}, "English transcription"),
        ({"word_timestamps": True}, "timestamps"),
        ({"language": "zh"}, "English transcription"),
    ],
)
def test_input_options_validated_before_backend(updates, match) -> None:
    kwargs = {
        "model_name": "test",
        "sample_rate": 16000,
        "language": None,
        "task": "transcribe",
        "word_timestamps": False,
        "options": None,
        "cancelled": None,
        "started": 0,
        "loaded_at": 0,
        **updates,
    }
    with pytest.raises(ValueError, match=match):
        transcribe_moonshine(SimpleNamespace(), [], **kwargs)


def test_cancellation_before_backend_and_separate_json_schema() -> None:
    with pytest.raises(RuntimeError, match="cancelled"):
        transcribe_moonshine(
            SimpleNamespace(),
            [],
            model_name="test",
            sample_rate=16000,
            language=None,
            task="transcribe",
            word_timestamps=False,
            options=None,
            cancelled=lambda: True,
            started=0,
            loaded_at=0,
        )
    result = SpeechTranscript("hello", 1.0, "model", "pin", 16000)
    data = json.loads(result.to_json())
    assert data["language"] is None and data["timestamps"] is None and data["confidence"] is None
    assert "avg_logprob" not in data
    for value in (0, True, 1.5):
        with pytest.raises(ValueError):
            ASRDecodeOptions(max_tokens=value)


def test_cli_default_family_options_and_explicit_whisper_flags(monkeypatch, tmp_path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    captured = []

    def fake(*args, **kwargs):
        captured.append(kwargs)
        return SpeechTranscript("hello", 1.0, "model", "pin", 16000)

    monkeypatch.setattr("mlx_one.cli.transcribe", fake)
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "model", str(audio), "--json-output"])
    assert result.exit_code == 0 and json.loads(result.output)["text"] == "hello"
    assert captured[-1]["options"] is None
    result = runner.invoke(main, ["transcribe", "model", str(audio), "--beam-size", "1"])
    assert result.exit_code == 0 and captured[-1]["options"].beam_size == 1
