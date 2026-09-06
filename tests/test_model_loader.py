import sys
import types

import pytest

from mlx_one.utils import model_loader
from mlx_one.utils.model_loader import (
    ModelLoadError,
    load_audio_model,
    load_embedding_model,
    load_model,
    load_vlm_model,
)


def test_load_model_uses_mlx_lm_load(monkeypatch) -> None:
    calls = []
    fake_module = types.ModuleType("mlx_lm")

    def fake_load(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return object(), object()

    fake_module.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    model, tokenizer = load_model("mlx-community/SmolLM-135M-4bit", revision="main")

    assert model is not None
    assert tokenizer is not None
    assert calls == [("mlx-community/SmolLM-135M-4bit", {"revision": "main"})]


def test_load_model_uses_mlx_vlm_load(monkeypatch) -> None:
    calls = []
    fake_module = types.ModuleType("mlx_vlm")

    def fake_load(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return "model", "processor"

    fake_module.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    loaded = load_model("mlx-community/Qwen2-VL-2B-Instruct-4bit", modality="vlm", lazy=True)

    assert loaded == ("model", "processor")
    assert calls == [("mlx-community/Qwen2-VL-2B-Instruct-4bit", {"lazy": True})]


def test_load_vlm_model_convenience(monkeypatch) -> None:
    fake_module = types.ModuleType("mlx_vlm")
    fake_module.load = lambda model_ref, **_kwargs: (model_ref, "processor")
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    assert load_vlm_model("mlx-community/example-vlm") == (
        "mlx-community/example-vlm",
        "processor",
    )


@pytest.mark.parametrize(
    ("modality", "module_name"),
    [
        ("audio-tts", "mlx_audio.tts.utils"),
        ("audio-stt", "mlx_audio.stt.utils"),
        ("audio-sts", "mlx_audio.sts.utils"),
    ],
)
def test_load_model_uses_mlx_audio_loaders(monkeypatch, modality, module_name) -> None:
    calls = []
    fake_module = types.ModuleType(module_name)

    def fake_load_model(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return f"{modality}-model"

    fake_module.load_model = fake_load_model
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    loaded = load_model("mlx-community/audio-model", modality=modality, strict=False)

    assert loaded == f"{modality}-model"
    assert calls == [("mlx-community/audio-model", {"strict": False})]


def test_load_audio_model_convenience(monkeypatch) -> None:
    fake_module = types.ModuleType("mlx_audio.tts.utils")
    fake_module.load_model = lambda model_ref, **_kwargs: model_ref
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    assert load_audio_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16") == (
        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    )


def test_load_model_uses_mlx_embeddings_load(monkeypatch) -> None:
    fake_module = types.ModuleType("mlx_embeddings.utils")
    fake_module.load = lambda model_ref, **_kwargs: (model_ref, "tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", fake_module)
    monkeypatch.setattr(model_loader, "_safe_memory_snapshot", lambda: None)

    loaded = load_model("openai/privacy-filter", modality="embeddings")

    assert loaded == ("openai/privacy-filter", "tokenizer")


def test_load_embedding_model_convenience(monkeypatch) -> None:
    fake_module = types.ModuleType("mlx_embeddings.utils")
    fake_module.load = lambda model_ref, **_kwargs: (model_ref, "tokenizer")
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", fake_module)
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
    fake_module = types.ModuleType("mlx_lm")

    def fake_load(_model_ref, **_kwargs):
        raise ValueError("Model type example_arch not supported.")

    fake_module.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    with pytest.raises(ModelLoadError, match="Unsupported llm model architecture"):
        load_model("mlx-community/example-model")


def test_missing_safetensors_is_wrapped(monkeypatch, tmp_path) -> None:
    fake_module = types.ModuleType("mlx_lm")

    def fake_load(_model_ref, **_kwargs):
        raise FileNotFoundError("No safetensors found")

    fake_module.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    with pytest.raises(ModelLoadError, match="Model files were not found"):
        load_model(tmp_path)
