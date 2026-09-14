from types import SimpleNamespace

import pytest

from mlx_one.utils import model_loader
from mlx_one.utils.model_loader import (
    ModelLoadError,
    load_audio_model,
    load_embedding_model,
    load_model,
    load_vlm_model,
)


def test_load_model_uses_native_text_loader(monkeypatch) -> None:
    calls = []
    def fake_load(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return SimpleNamespace(model="model", tokenizer="tokenizer")

    monkeypatch.setattr("mlx_one.text.load_text_model", fake_load)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    model, tokenizer = load_model("openbmb/MiniCPM5-1B-MLX", revision="main")

    assert model is not None
    assert tokenizer is not None
    assert calls == [("openbmb/MiniCPM5-1B-MLX", {"revision": "main"})]


def test_load_model_uses_native_vlm_loader(monkeypatch) -> None:
    calls = []
    def fake_load(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return SimpleNamespace(model="model", processor="processor")

    monkeypatch.setattr("mlx_one.vision.loading.load_vlm_model", fake_load)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    loaded = load_model("mlx-community/Qwen2-VL-2B-Instruct-4bit", modality="vlm", lazy=True)

    assert loaded == ("model", "processor")
    assert calls == [("mlx-community/Qwen2-VL-2B-Instruct-4bit", {"lazy": True})]


def test_load_vlm_model_convenience(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlx_one.vision.loading.load_vlm_model",
        lambda model_ref, **_kwargs: SimpleNamespace(model=model_ref, processor="processor"),
    )
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    assert load_vlm_model("mlx-community/example-vlm") == (
        "mlx-community/example-vlm",
        "processor",
    )


def test_load_model_uses_native_whisper_loader(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlx_one.models.audio.whisper.loading.load_whisper",
        lambda *_args, **_kwargs: SimpleNamespace(model="whisper"),
    )
    assert load_model("openai/whisper-tiny", modality="audio-stt") == "whisper"


@pytest.mark.parametrize("task", ["tts", "sts"])
def test_unsupported_audio_tasks_fail_fast(task) -> None:
    with pytest.raises(ModelLoadError, match="planned but not implemented"):
        load_audio_model("example/audio", task=task)


def test_load_model_uses_native_retrieval_loader(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlx_one.retrieval.load_retrieval_model",
        lambda model_ref, **_kwargs: SimpleNamespace(model=model_ref, tokenizer="tokenizer"),
    )
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    loaded = load_model("openai/privacy-filter", modality="embeddings")

    assert loaded == ("openai/privacy-filter", "tokenizer")


def test_load_embedding_model_convenience(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlx_one.retrieval.load_retrieval_model",
        lambda model_ref, **_kwargs: SimpleNamespace(model=model_ref, tokenizer="tokenizer"),
    )
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    assert load_embedding_model("mlx-community/all-MiniLM-L6-v2-4bit") == (
        "mlx-community/all-MiniLM-L6-v2-4bit",
        "tokenizer",
    )


def test_audio_modality_requires_task() -> None:
    with pytest.raises(ModelLoadError, match="ambiguous"):
        load_model("mlx-community/audio-model", modality="audio")


def test_unknown_modality_fails_clearly() -> None:
    with pytest.raises(ModelLoadError, match="Unsupported model modality"):
        load_model("mlx-community/model", modality="video")


def test_empty_model_ref_fails() -> None:
    with pytest.raises(ModelLoadError, match="cannot be empty"):
        load_model(" ")


def test_missing_local_path_fails_clearly(tmp_path) -> None:
    missing = tmp_path / "missing-model"

    with pytest.raises(ModelLoadError, match="Local model path does not exist"):
        load_model(missing)


def test_file_path_fails_clearly(tmp_path) -> None:
    model_file = tmp_path / "model.safetensors"
    model_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="must be a directory"):
        load_model(model_file)


def test_unsupported_architecture_is_wrapped(monkeypatch) -> None:
    def fake_load(_model_ref, **_kwargs):
        raise ValueError("Model type example_arch not supported.")

    monkeypatch.setattr("mlx_one.text.load_text_model", fake_load)

    with pytest.raises(ModelLoadError, match="Unsupported llm model architecture"):
        load_model("mlx-community/example-model")


def test_missing_safetensors_is_wrapped(monkeypatch, tmp_path) -> None:
    def fake_load(_model_ref, **_kwargs):
        raise FileNotFoundError("No safetensors found")

    monkeypatch.setattr("mlx_one.text.load_text_model", fake_load)

    with pytest.raises(ModelLoadError, match="Model files were not found"):
        load_model(tmp_path)
