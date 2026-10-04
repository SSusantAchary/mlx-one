"""Backend-free task dispatch; future architecture names are not support claims."""

import json
from types import SimpleNamespace

import pytest

from mlx_one.audio import AudioModelLoadError, load_audio_model
from mlx_one.audio.loading import audio_model_type


def test_qwen_aligner_is_resolved_from_nested_config() -> None:
    assert (
        audio_model_type(
            {
                "model_type": "qwen3_asr",
                "thinker_config": {
                    "model_type": "qwen3_forced_aligner",
                },
            }
        )
        == "qwen3_forced_aligner"
    )
    assert audio_model_type({"model_type": "qwen3_asr"}) == "qwen3_asr"
    with pytest.raises(AudioModelLoadError, match="model_type"):
        audio_model_type({})


def test_task_dispatch_forwards_loader_options_without_loading_backend(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    captured = {}

    def loader(model, **kwargs):
        captured.update(kwargs)
        return "loaded"

    monkeypatch.setattr(
        "mlx_one.audio.loading.get_registration",
        lambda _: SimpleNamespace(
            modality="asr",
            supports=lambda task: task == "transcribe",
            loader=lambda: loader,
        ),
    )
    assert load_audio_model(tmp_path, revision="pin", offline=True, cache_dir=tmp_path) == "loaded"
    assert captured == {"revision": "pin", "offline": True, "cache_dir": tmp_path}


@pytest.mark.parametrize("task", ["align", "synthesize"])
def test_whisper_refuses_other_audio_tasks_before_backend_loading(tmp_path, task) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}))
    with pytest.raises(AudioModelLoadError, match="does not support"):
        load_audio_model(tmp_path, task=task)


def test_non_audio_and_unimplemented_models_are_not_accepted(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(AudioModelLoadError, match="not an audio"):
        load_audio_model(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_asr"}))
    with pytest.raises(AudioModelLoadError, match="unsupported native"):
        load_audio_model(tmp_path)


def test_invalid_task_and_missing_local_assets_are_rejected(tmp_path) -> None:
    with pytest.raises(AudioModelLoadError, match="unsupported audio task"):
        load_audio_model(tmp_path, task="generate")
    with pytest.raises(AudioModelLoadError, match="does not exist"):
        load_audio_model(tmp_path / "missing")
