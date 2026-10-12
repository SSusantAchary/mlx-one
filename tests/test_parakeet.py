"""Network-free Parakeet contracts, archive safety and task dispatch."""

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.parakeet_conversion import convert, inspect_archive
from mlx_one.audio.loading import AudioModelLoadError, load_audio_model
from mlx_one.audio.parakeet import timestamp_tokens, transcribe_parakeet
from mlx_one.audio.schemas import QwenASROptions
from mlx_one.audio.streaming import open_audio_stream
from mlx_one.core.config import ConfigError
from mlx_one.core.registry import get_registration
from mlx_one.models.audio.parakeet.config import ParakeetConfig, encoded_length
from mlx_one.models.audio.parakeet.loading import (
    LoadedParakeet,
    validate_generation,
    validate_headers,
)
from mlx_one.models.audio.parakeet.processing import ParakeetProcessor
from mlx_one.models.audio.parakeet.weights import weight_contract


def test_registration_is_offline_only_and_task_validation_precedes_backend(tmp_path):
    registration = get_registration("parakeet_tdt")
    assert registration.supports("transcribe")
    assert not registration.supports("align")
    assert not registration.supports("synthesize")
    assert "stream" not in registration.capabilities
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "parakeet_tdt"}))
    for task in ("align", "synthesize"):
        with pytest.raises(AudioModelLoadError, match="does not support"):
            load_audio_model(root, task=task, offline=True)
    with pytest.raises(AudioModelLoadError, match="NeMo archive"):
        load_audio_model("nvidia/parakeet-tdt-0.6b-v2", offline=True)


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_size": 7},
        {"durations": (0, 1, 3)},
        {"durations": (0.0, 1, 2, 3, 4)},
        {"blank_token_id": 10},
        {"num_hidden_layers": True},
        {"conv_kernel_size": 8},
        {"num_mel_bins": 127},
    ],
)
def test_config_rejects_unsupported_dimensions(change):
    with pytest.raises(ConfigError):
        replace(ParakeetConfig(), **change)


def test_pinned_v3_contract_matches_all_frozen_headers():
    path = (
        Path(__file__).parents[1] / "qualification/results/parakeet-tdt-v3-checkpoint-contract.json"
    )
    contract = json.loads(path.read_text())
    config = ParakeetConfig.from_dict(contract["config"])
    weights = {
        name: SimpleNamespace(shape=tensor["shape"]) for name, tensor in contract["tensors"].items()
    }
    weight_contract(config, native=False).validate(weights)
    assert len(weights) == 723


def test_subsampling_uses_ceil_and_final_mask_length():
    assert [encoded_length(n) for n in (1, 7, 8, 9, 25)] == [1, 1, 1, 2, 4]


def test_generation_settings_are_validated():
    config = ParakeetConfig()
    data = {
        "decoder_start_token_id": config.blank_token_id,
        "pad_token_id": config.pad_token_id,
        "suppress_tokens": list(range(config.vocab_size, config.vocab_size + 5)),
    }
    validate_generation(data, config)
    for change in ({"decoder_start_token_id": 0}, {"do_sample": True}, {"suppress_tokens": []}):
        with pytest.raises(ValueError):
            validate_generation({**data, **change}, config)


def test_wrong_weight_precision_is_rejected_before_metal(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    save_file({"unexpected": np.zeros(1, dtype=np.float16)}, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="dtype"):
        validate_headers(tmp_path, ParakeetConfig())


def tar_inventory(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as out:
        for name, kind, size in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = size if kind == tarfile.REGTYPE else 0
            member.linkname = "/escape"
            out.addfile(member, io.BytesIO(b"a" * member.size) if member.isfile() else None)
    data.seek(0)
    return tarfile.open(fileobj=data, mode="r")


@pytest.mark.parametrize(
    "entries",
    [
        [("../escape", tarfile.REGTYPE, 1)],
        [("/escape", tarfile.REGTYPE, 1)],
        [("safe", tarfile.SYMTYPE, 0)],
        [("safe", tarfile.LNKTYPE, 0)],
        [("safe", tarfile.FIFOTYPE, 0)],
        [("a\\b", tarfile.REGTYPE, 1)],
        [("same", tarfile.REGTYPE, 1), ("./same", tarfile.REGTYPE, 1)],
    ],
)
def test_archive_rejects_paths_links_special_files_and_duplicates(entries):
    with tar_inventory(entries) as archive, pytest.raises(ValueError):
        inspect_archive(archive)


def test_archive_size_limit_and_regular_entries():
    with tar_inventory([("./safe", tarfile.REGTYPE, 5)]) as archive:
        assert set(inspect_archive(archive)) == {"safe"}
    with tar_inventory([("./safe", tarfile.REGTYPE, 5)]) as archive, pytest.raises(ValueError):
        inspect_archive(archive, max_bytes=4)


def test_conversion_refuses_existing_output_before_loading_checkpoint(tmp_path):
    with pytest.raises(FileExistsError):
        convert(Path("/missing"), Path("/missing"), tmp_path)


def test_conversion_refuses_unpinned_archive_without_importing_torch(tmp_path):
    archive = tmp_path / "bad.nemo"
    archive.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="checksum"):
        convert(archive, Path("/missing"), tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_token_timestamps_keep_repeated_tokens_and_attach_punctuation():
    from tokenizers import Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel

    tokenizer = Tokenizer(WordLevel({"a": 0, ".": 1, "<blank>": 2}, unk_token="a"))
    tokenizer.decoder = Fuse()
    spans = timestamp_tokens(tokenizer, (2, 0, 0, 1), (0, 1, 2, 1), blank=2, pad=2)
    assert len(spans) == 3
    assert spans[0]["start"] == 0
    assert spans[1]["end"] == pytest.approx(0.24)
    assert spans[2]["start"] == spans[2]["end"] == spans[1]["end"]


def test_processor_requires_family_settings():
    with pytest.raises(ValueError):
        ParakeetProcessor.from_dict({"feature_extractor_type": "WhisperFeatureExtractor"})


def test_loaded_parakeet_streaming_is_rejected_without_materializing_weights():
    bundle = LoadedParakeet(None, None, ParakeetProcessor(), {}, Path("/model"), None, "v3")
    with pytest.raises(ValueError, match="offline-only"):
        open_audio_stream(bundle, sample_rate=16000)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"sample_rate": None}, "explicit"),
        ({"sample_rate": 8000}, "16000"),
        ({"sample_rate": True}, "16000"),
        ({"word_timestamps": True}, "word timestamps"),
        ({"task": "translate"}, "offline transcription"),
        ({"language": "fr"}, "language conditioning"),
        ({"options": QwenASROptions()}, "Whisper/Qwen"),
    ],
)
def test_invalid_transcription_options_fail_before_backend(change, message):
    bundle = LoadedParakeet(None, None, ParakeetProcessor(), {}, Path("/model"), None, "v3")
    kwargs = dict(
        model_name="test",
        sample_rate=16000,
        language=None,
        task="transcribe",
        word_timestamps=False,
        options=None,
        cancelled=None,
        started=0.0,
        loaded_at=0.0,
    )
    with pytest.raises(ValueError, match=message):
        transcribe_parakeet(bundle, [], **{**kwargs, **change})
