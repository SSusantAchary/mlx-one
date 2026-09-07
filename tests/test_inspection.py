import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_one import Modality, ModelSource, inspection
from mlx_one.inspection import InspectionError, inspect_model, write_inspection_result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inspect_local_safetensors_metadata_without_loading_weights(tmp_path) -> None:
    write_json(
        tmp_path / "config.json",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "tokenizer_class": "LlamaTokenizer",
            "quantization": {"bits": 4, "group_size": 64},
        },
    )
    (tmp_path / "README.md").write_text(
        "---\nlicense: mit\npipeline_tag: text-generation\n---\n# Model\n",
        encoding="utf-8",
    )
    save_file(
        {
            "layer.weight": np.zeros((2, 3), dtype=np.float32),
            "layer.bias": np.zeros((3,), dtype=np.float32),
        },
        tmp_path / "model.safetensors",
    )

    result = inspect_model(tmp_path)

    assert result.source is ModelSource.LOCAL
    assert result.model.modality is Modality.TEXT
    assert result.model.parameter_count == 9
    assert result.model.parameter_count_source == "local-safetensors-header"
    assert result.model.dtype == "F32"
    assert result.model.weight_format == "safetensors"
    assert result.model.license == "mit"
    assert result.model.quantization == {"bits": 4, "group_size": 64}
    assert result.model.revision.startswith("local:")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen2_vl", "vision_config": {}}, Modality.VISION_LANGUAGE),
        ({"architectures": ["WhisperForConditionalGeneration"]}, Modality.ASR),
        ({"architectures": ["BarkModel"]}, Modality.TTS),
        ({"architectures": ["SentenceTransformer"]}, Modality.EMBEDDING),
        ({}, Modality.UNKNOWN),
    ],
)
def test_local_modality_classification(tmp_path, config, expected) -> None:
    write_json(tmp_path / "config.json", config)

    result = inspect_model(tmp_path)

    assert result.model.modality is expected


def test_local_mixed_weights_do_not_unpickle_pytorch_file(tmp_path) -> None:
    write_json(tmp_path / "config.json", {"architectures": ["LlamaForCausalLM"]})
    save_file({"weight": np.zeros((2, 2), dtype=np.float16)}, tmp_path / "model.safetensors")
    (tmp_path / "pytorch_model.bin").write_bytes(b"not a pickle and must not be opened")

    result = inspect_model(tmp_path)

    assert result.model.weight_format == "mixed"
    assert result.model.parameter_count == 4


def test_local_pytorch_only_reports_unknown_parameter_count(tmp_path) -> None:
    write_json(tmp_path / "config.json", {"architectures": ["LlamaForCausalLM"]})
    (tmp_path / "model.bin").write_bytes(b"unsafe pickle placeholder")

    result = inspect_model(tmp_path)

    assert result.model.parameter_count is None
    assert result.model.parameter_count_source == "unknown"
    assert "Parameter count is unavailable from metadata." in result.warnings


def test_local_revision_is_deterministic_and_revision_option_warns(tmp_path) -> None:
    write_json(tmp_path / "config.json", {"model_type": "llama"})

    first = inspect_model(tmp_path, revision="ignored")
    second = inspect_model(tmp_path)

    assert first.model.revision == second.model.revision
    assert "revision option is ignored" in " ".join(first.warnings)


def test_local_remote_code_requirement_and_tokenizer_files(tmp_path) -> None:
    write_json(
        tmp_path / "config.json",
        {"model_type": "custom", "auto_map": {"AutoModel": "model.CustomModel"}},
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    result = inspect_model(tmp_path)

    assert result.model.requires_remote_code is True
    assert result.model.tokenizer_id == str(tmp_path.resolve())


def test_invalid_local_sources_fail_clearly(tmp_path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(InspectionError, match="does not exist"):
        inspect_model(missing)

    file_path = tmp_path / "model.gguf"
    file_path.write_bytes(b"gguf")
    with pytest.raises(InspectionError, match="must be a directory"):
        inspect_model(file_path)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InspectionError, match="no recognized"):
        inspect_model(empty)


def test_malformed_present_metadata_fails(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(InspectionError, match="invalid metadata file"):
        inspect_model(tmp_path)


def test_hub_inspection_resolves_sha_and_uses_api_metadata(monkeypatch) -> None:
    calls = []
    info = SimpleNamespace(
        id="organization/model",
        sha="a" * 40,
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=1024, lfs=None)],
        config={
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "tokenizer_class": "LlamaTokenizer",
        },
        card_data=SimpleNamespace(
            license="apache-2.0",
            pipeline_tag=None,
            library_name=None,
            base_model=None,
        ),
        pipeline_tag="text-generation",
        library_name="transformers",
        base_models=["organization/base"],
        tags=["text-generation"],
        safetensors=SimpleNamespace(parameters={"BF16": 8_000_000_000}, total=8_000_000_000),
        gated=False,
        private=False,
        disabled=False,
        used_storage=2048,
    )

    class FakeApi:
        def model_info(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            return info

    monkeypatch.setattr(inspection, "HfApi", FakeApi)
    monkeypatch.setattr(
        inspection,
        "hf_hub_download",
        lambda *_args, **_kwargs: pytest.fail("no supplemental metadata should be downloaded"),
    )

    result = inspect_model("organization/model", revision="main")

    assert calls == [("organization/model", {"revision": "main", "files_metadata": True})]
    assert result.source is ModelSource.HUGGING_FACE
    assert result.requested_revision == "main"
    assert result.model.revision == "a" * 40
    assert result.model.parameter_count == 8_000_000_000
    assert result.model.parameter_count_source == "hub-safetensors"
    assert result.model.dtype == "BF16"
    assert result.model.base_models == ("organization/base",)


def test_hub_downloads_only_small_known_json_metadata(monkeypatch, tmp_path) -> None:
    tokenizer = tmp_path / "tokenizer_config.json"
    write_json(tokenizer, {"auto_map": {"AutoTokenizer": "tokenizer.Custom"}})
    downloads = []
    info = SimpleNamespace(
        id="organization/model",
        sha="b" * 40,
        siblings=[
            SimpleNamespace(rfilename="tokenizer_config.json", size=128, lfs=None),
            SimpleNamespace(rfilename="model.safetensors", size=10_000_000, lfs=None),
        ],
        config={"model_type": "custom", "architectures": ["CustomForCausalLM"]},
        card_data=None,
        pipeline_tag="text-generation",
        library_name="transformers",
        base_models=None,
        tags=[],
        safetensors=None,
        gated=False,
        private=False,
        disabled=False,
        used_storage=None,
    )

    monkeypatch.setattr(
        inspection,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *_args, **_kwargs: info),
    )

    def fake_download(_repo_id, filename, **_kwargs):
        downloads.append(filename)
        assert filename.endswith(".json")
        return str(tokenizer)

    monkeypatch.setattr(inspection, "hf_hub_download", fake_download)

    result = inspect_model("organization/model")

    assert downloads == ["tokenizer_config.json"]
    assert result.model.requires_remote_code is True


def test_hub_missing_optional_metadata_succeeds_with_warnings(monkeypatch) -> None:
    info = SimpleNamespace(
        id="organization/model",
        sha=None,
        siblings=[],
        config={},
        card_data=None,
        pipeline_tag=None,
        library_name=None,
        base_models=None,
        tags=[],
        safetensors=None,
        gated=False,
        private=False,
        disabled=False,
        used_storage=None,
    )
    monkeypatch.setattr(
        inspection,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *_args, **_kwargs: info),
    )

    result = inspect_model("organization/model")

    assert result.model.modality is Modality.UNKNOWN
    assert "Parameter count is unavailable from metadata." in result.warnings
    assert "Weight format is unavailable from metadata." in result.warnings
    assert "The requested Hub revision could not be resolved" in " ".join(result.warnings)


def test_hub_not_found_is_actionable(monkeypatch) -> None:
    class FakeRepositoryNotFoundError(Exception):
        pass

    def fail(*_args, **_kwargs):
        raise FakeRepositoryNotFoundError("missing")

    monkeypatch.setattr(inspection, "HfApi", lambda: SimpleNamespace(model_info=fail))
    monkeypatch.setattr(
        inspection,
        "RepositoryNotFoundError",
        FakeRepositoryNotFoundError,
    )

    with pytest.raises(InspectionError, match="not found or is private"):
        inspect_model("organization/missing")


def test_hub_gated_error_explains_authentication(monkeypatch) -> None:
    class FakeGatedRepoError(Exception):
        pass

    def fail(*_args, **_kwargs):
        raise FakeGatedRepoError("gated")

    monkeypatch.setattr(inspection, "HfApi", lambda: SimpleNamespace(model_info=fail))
    monkeypatch.setattr(inspection, "GatedRepoError", FakeGatedRepoError)

    with pytest.raises(InspectionError, match="hf auth login"):
        inspect_model("organization/gated")


def test_offline_cache_miss_is_actionable(monkeypatch) -> None:
    class FakeLocalEntryNotFoundError(Exception):
        pass

    def fail(*_args, **_kwargs):
        raise FakeLocalEntryNotFoundError("cache miss")

    monkeypatch.setattr(inspection, "snapshot_download", fail)
    monkeypatch.setattr(
        inspection,
        "LocalEntryNotFoundError",
        FakeLocalEntryNotFoundError,
    )

    with pytest.raises(InspectionError, match="rerun without --offline"):
        inspect_model("organization/model", offline=True)


def test_offline_hub_uses_only_cached_snapshot(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "models--organization--model" / "snapshots" / ("c" * 40)
    snapshot.mkdir(parents=True)
    write_json(snapshot / "config.json", {"model_type": "llama"})
    calls = []

    def fake_snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr(inspection, "snapshot_download", fake_snapshot)

    result = inspect_model("organization/model", revision="main", offline=True)

    assert calls[0][1]["local_files_only"] is True
    assert result.source is ModelSource.HUGGING_FACE
    assert result.model.model_id == "organization/model"
    assert result.model.revision == "c" * 40


def test_write_result_is_json_and_replaces_atomically(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "llama"})
    result = inspect_model(model_dir)
    output = tmp_path / "result.json"

    saved = write_inspection_result(result, output)

    assert saved == output
    assert json.loads(output.read_text())["model"]["model_type"] == "llama"
    assert not (tmp_path / "result.json.tmp").exists()


def test_import_and_local_inspection_do_not_import_model_runtimes(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_json(model_dir / "config.json", {"model_type": "llama"})
    code = (
        "import sys; from mlx_one import inspect_model; "
        f"inspect_model({str(model_dir)!r}); "
        "assert 'mlx' not in sys.modules; "
        "assert 'mlx_lm' not in sys.modules; "
        "assert 'transformers' not in sys.modules"
    )

    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
