"""Network- and backend-free audio snapshot contracts."""

import json
from pathlib import Path

import pytest

from benchmarks.audio_checkpoint_contract import contract
from mlx_one.audio.weights import AudioWeightsError, select_safetensors, validate_tensor_index
from mlx_one.models.audio.whisper.config import WhisperConfig
from mlx_one.models.audio.whisper.loading import WhisperLoadError, _validate_generation_config
from mlx_one.models.audio.whisper.weights import weight_contract


def test_distil_large_v3_asymmetric_contract() -> None:
    config = WhisperConfig.from_dict(
        {
            "d_model": 1280,
            "num_mel_bins": 128,
            "vocab_size": 51866,
            "encoder_layers": 32,
            "decoder_layers": 2,
            "encoder_attention_heads": 20,
            "decoder_attention_heads": 20,
            "encoder_ffn_dim": 5120,
            "decoder_ffn_dim": 5120,
        }
    )
    expected = weight_contract(config).expected
    assert "model.encoder.layers.31.fc1.weight" in expected
    assert "model.decoder.layers.1.fc1.weight" in expected
    assert "model.decoder.layers.2.fc1.weight" not in expected
    assert expected["model.encoder.conv1.weight"] == (1280, 3, 128)


def test_precision_variants_are_not_merged(tmp_path: Path) -> None:
    for name in ("model.safetensors", "model.fp32.safetensors", "consolidated.safetensors"):
        (tmp_path / name).touch()
    files, mapping = select_safetensors(tmp_path)
    assert files == (tmp_path / "model.safetensors",)
    assert mapping is None


def test_index_is_authoritative_over_alternate_export(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"},
            }
        )
    )
    files, mapping = select_safetensors(tmp_path)
    assert [path.name for path in files] == ["part-1.safetensors", "part-2.safetensors"]
    validate_tensor_index(
        files, mapping, {"part-1.safetensors": ("a",), "part-2.safetensors": ("b",)}
    )


def test_explicit_layout_is_authoritative_and_never_falls_back(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "part.safetensors"}})
    )
    assert select_safetensors(tmp_path, layout="canonical") == (
        (tmp_path / "model.safetensors",),
        None,
    )
    assert select_safetensors(tmp_path, layout="indexed")[0] == (tmp_path / "part.safetensors",)
    with pytest.raises(AudioWeightsError, match="layout"):
        select_safetensors(tmp_path, layout="unknown")


@pytest.mark.parametrize("layout", ["canonical", "indexed"])
def test_required_layout_missing_is_rejected(tmp_path: Path, layout: str) -> None:
    (tmp_path / "alternate.safetensors").touch()
    with pytest.raises(AudioWeightsError, match="required"):
        select_safetensors(tmp_path, layout=layout)


@pytest.mark.parametrize("asset", ["../escape", "/escape", "a\\b", "a:b", "missing.json", ""])
def test_required_family_assets_are_validated_before_tensor_loading(tmp_path, asset):
    with pytest.raises(ValueError, match="asset"):
        contract(tmp_path, "publisher/model", "a" * 40, "license", required_assets=(asset,))


def test_required_nested_assets_retain_unique_relative_names(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    (tmp_path / "config.json").write_text('{"model_type":"test"}')
    for family in ("text", "audio"):
        directory = tmp_path / family
        directory.mkdir()
        (directory / "tokenizer.json").write_text(json.dumps({"family": family}))
    save_file({"x": np.zeros(1, dtype=np.float32)}, tmp_path / "model.safetensors")
    result = contract(
        tmp_path,
        "publisher/model",
        "a" * 40,
        "license",
        required_assets=("text/tokenizer.json", "audio/tokenizer.json"),
        weight_layout="canonical",
    )
    assert result["assets"]["text/tokenizer.json"] != result["assets"]["audio/tokenizer.json"]


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        {},
        {"a": "../escape.safetensors"},
        {"a": "/escape.safetensors"},
        {"a": "a\\b.safetensors"},
        {"a": "model.bin"},
        {"a": 3},
    ],
)
def test_invalid_index_is_rejected(tmp_path: Path, mapping: object) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(AudioWeightsError):
        select_safetensors(tmp_path)


def test_duplicate_json_tensor_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"part-1.safetensors","a":"part-2.safetensors"}}'
    )
    with pytest.raises(AudioWeightsError, match="duplicate"):
        select_safetensors(tmp_path)


@pytest.mark.parametrize(
    "names",
    [
        {"part-1.safetensors": ("a",), "part-2.safetensors": ("a", "b")},
        {"part-1.safetensors": ("b",), "part-2.safetensors": ("a",)},
        {"part-1.safetensors": ("a",), "part-2.safetensors": ()},
        {"part-1.safetensors": ("a",), "part-2.safetensors": ("b", "extra")},
    ],
)
def test_misassigned_missing_and_duplicate_tensors_are_rejected(names: dict) -> None:
    files = (Path("part-1.safetensors"), Path("part-2.safetensors"))
    with pytest.raises(AudioWeightsError):
        validate_tensor_index(files, {"a": files[0].name, "b": files[1].name}, names)


def test_local_single_export_and_ambiguous_exports(tmp_path: Path) -> None:
    (tmp_path / "converted.safetensors").touch()
    assert select_safetensors(tmp_path)[0] == (tmp_path / "converted.safetensors",)
    (tmp_path / "alternate.safetensors").touch()
    with pytest.raises(AudioWeightsError, match="ambiguous"):
        select_safetensors(tmp_path)


def test_pickle_only_checkpoint_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.bin").touch()
    with pytest.raises(AudioWeightsError, match="pickle"):
        select_safetensors(tmp_path)


@pytest.mark.parametrize(
    "generation",
    [
        {"alignment_heads": [[2, 0]]},
        {"alignment_heads": [[1, 20]]},
        {"alignment_heads": [[True, 0]]},
        {"alignment_heads": []},
        {"suppress_tokens": [-1]},
        {"suppress_tokens": [51866]},
        {"begin_suppress_tokens": [True]},
    ],
)
def test_distil_generation_metadata_rejects_wrong_heads_and_ids(generation: dict) -> None:
    config = WhisperConfig(
        decoder_layers=2,
        decoder_attention_heads=20,
        d_model=1280,
        encoder_attention_heads=20,
        vocab_size=51866,
    )
    with pytest.raises(WhisperLoadError):
        _validate_generation_config(generation, config)


def test_distil_generation_metadata_accepts_second_decoder_layer_heads() -> None:
    config = WhisperConfig(
        decoder_layers=2,
        decoder_attention_heads=20,
        d_model=1280,
        encoder_attention_heads=20,
        vocab_size=51866,
    )
    _validate_generation_config(
        {"alignment_heads": [[1, head] for head in range(20)], "suppress_tokens": [1, 50363]},
        config,
    )
