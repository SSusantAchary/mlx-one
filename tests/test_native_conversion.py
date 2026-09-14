import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mlx_one.conversion import ConversionError, convert_text_model


class _Array:
    dtype = "f32"
    nbytes = 16

    def astype(self, dtype):
        result = _Array()
        result.dtype = dtype
        return result


def _fake_mlx(monkeypatch, saved) -> None:
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.float16 = "f16"
    core.float32 = "f32"
    core.bfloat16 = "bf16"

    def save(path, tensors):
        saved.append((path, tuple(tensors)))
        Path(path).write_bytes(b"safe")

    core.save_safetensors = save
    nn = ModuleType("mlx.nn")
    nn.quantize = lambda *_args, **_kwargs: None
    utils = ModuleType("mlx.utils")
    utils.tree_flatten = lambda _value: [("model.weight", _Array())]
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", utils)


def test_native_conversion_emits_shard_index_and_provenance(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "quantization": {"bits": 8}}),
        encoding="utf-8",
    )
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    saved = []
    _fake_mlx(monkeypatch, saved)
    monkeypatch.setattr(
        "mlx_one.text.load_text_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=SimpleNamespace(parameters=lambda: {}), path=source
        ),
    )

    target = convert_text_model(source, tmp_path / "converted")

    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    index = json.loads((target / "model.safetensors.index.json").read_text())
    provenance = json.loads((target / "mlx-one-conversion.json").read_text())
    assert config["torch_dtype"] == "bfloat16"
    assert "quantization" not in config
    assert index["weight_map"] == {"model.weight": "model-00001-of-00001.safetensors"}
    assert provenance["converter"] == "mlx-one-native"
    assert saved


def test_native_conversion_rejects_existing_destination(tmp_path) -> None:
    with pytest.raises(ConversionError, match="already exists"):
        convert_text_model(tmp_path / "source", tmp_path)
