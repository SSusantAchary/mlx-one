"""Backend-free contracts for native GPT-2."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.language.gpt2.config import GPT2Config
from mlx_one.models.language.gpt2.weights import sanitize_weights, weight_contract
from mlx_one.text.loading import (
    TextModelLoadError,
    _checkpoint_files,
    resolve_text_model_type,
)
from mlx_one.text.schemas import GenerationResult, TextGenerationOptions
from mlx_one.text.tokenizer import GPT2Tokenizer, bytes_to_unicode
from mlx_one.training import SFTTrainer, TrainingError


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]

    def transpose(self, first: int, second: int) -> Shaped:
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return Shaped(tuple(shape))


def tiny_config(**updates: object) -> GPT2Config:
    values = {
        "model_type": "gpt2",
        "vocab_size": 257,
        "n_positions": 16,
        "n_ctx": 16,
        "n_embd": 8,
        "n_layer": 2,
        "n_head": 2,
        "bos_token_id": 256,
        "eos_token_id": 256,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }
    values.update(updates)
    return GPT2Config.from_dict(values)


@pytest.mark.parametrize(
    ("name", "layers", "width", "heads"),
    [
        ("gpt2", 12, 768, 12),
        ("gpt2-medium", 24, 1024, 16),
        ("gpt2-large", 36, 1280, 20),
        ("gpt2-xl", 48, 1600, 25),
    ],
)
def test_official_gpt2_profiles_share_one_config(
    name: str, layers: int, width: int, heads: int
) -> None:
    del name
    config = GPT2Config(n_layer=layers, n_embd=width, n_head=heads)
    assert config.inner_dim == width * 4
    assert config.head_dim == 64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "gpt_neo"}, "model_type"),
        ({"n_ctx": 8}, "n_ctx"),
        ({"n_embd": 7}, "divisible"),
        ({"activation_function": "gelu"}, "gelu_new"),
        ({"attn_pdrop": 1.0}, "attn_pdrop"),
        ({"tie_word_embeddings": False}, "tied"),
        ({"add_cross_attention": True}, "cross-attention"),
        ({"scale_attn_by_inverse_layer_idx": True}, "inverse"),
    ],
)
def test_gpt2_rejects_unsupported_config(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        tiny_config(**updates)


def test_gpt2_registry_is_lazy_and_backend_free() -> None:
    mlx_was_loaded = "mlx.core" in sys.modules
    assert "gpt2" in registered_model_types()
    registration = get_registration("gpt2")
    assert registration.modality == "text"
    assert registration.capabilities == {"forward", "cache", "generate", "sample", "stream"}
    assert registration.config_class() is GPT2Config
    assert ("mlx.core" in sys.modules) is mlx_was_loaded


def test_gpt2_weight_contract_transposes_conv1d_and_cleans_buffers() -> None:
    config = tiny_config()
    contract = weight_contract(config)
    source: dict[str, Shaped] = {}
    transposed_suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    for name, shape in contract.expected.items():
        source[name] = Shaped(
            tuple(reversed(shape)) if name.endswith(transposed_suffixes) else shape
        )
    source["lm_head.weight"] = Shaped((config.vocab_size, config.n_embd))
    source["transformer.h.0.attn.bias"] = Shaped((1, 1, 16, 16))
    source["transformer.h.0.attn.masked_bias"] = Shaped(())
    sanitized = sanitize_weights(source, config)
    contract.validate(sanitized)
    assert "lm_head.weight" not in sanitized
    invalid = dict(sanitized)
    invalid["unexpected.weight"] = Shaped((1,))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)
    missing = dict(sanitized)
    missing.pop("transformer.wte.weight")
    with pytest.raises(WeightContractError, match="missing tensors"):
        contract.validate(missing)
    mismatched = dict(sanitized)
    mismatched["transformer.wpe.weight"] = Shaped((1, 1))
    with pytest.raises(WeightContractError, match="shape mismatches"):
        contract.validate(mismatched)


def _write_tokenizer(root: Path, *, token_objects: bool = False) -> GPT2Tokenizer:
    encoder = bytes_to_unicode()
    vocabulary = {character: index for index, character in enumerate(encoder.values())}
    vocabulary["<|endoftext|>"] = 256
    (root / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    token: object = {"content": "<|endoftext|>"} if token_objects else "<|endoftext|>"
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": token, "eos_token": token, "unk_token": token}),
        encoding="utf-8",
    )
    return GPT2Tokenizer.from_directory(root)


def test_gpt2_byte_bpe_roundtrip_and_token_metadata(tmp_path: Path) -> None:
    tokenizer = _write_tokenizer(tmp_path, token_objects=True)
    text = "Hello, café!\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.encode("") == []
    assert tokenizer.bos_token_id == tokenizer.eos_token_id == 256
    merged = GPT2Tokenizer(
        {"hello": 0, "<|endoftext|>": 1},
        (("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")),
    )
    assert merged.bpe("hello") == ("hello",)


def test_generation_options_and_results_are_strict_and_serializable() -> None:
    options = TextGenerationOptions(
        max_tokens=8, temperature=0.7, top_k=5, top_p=0.9, seed=12, stop=["END"]
    )
    assert options.stop == ("END",)
    result = GenerationResult("p", "x", 1, 1, "length", "gpt2")
    assert json.loads(result.to_json())["text"] == "x"
    with pytest.raises(ValueError, match="top_p"):
        TextGenerationOptions(top_p=0)
    with pytest.raises(ValueError, match="stop"):
        TextGenerationOptions(stop=("",))


def test_loader_resolves_local_metadata_without_mlx(tmp_path: Path) -> None:
    mlx_was_loaded = "mlx.core" in sys.modules
    (tmp_path / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    assert resolve_text_model_type(tmp_path) == "gpt2"
    assert ("mlx.core" in sys.modules) is mlx_was_loaded
    with pytest.raises(TextModelLoadError, match="does not exist"):
        resolve_text_model_type(tmp_path / "missing")


def test_loader_checkpoint_policy_rejects_pickle_and_unsafe_indexes(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model.bin").touch()
    with pytest.raises(TextModelLoadError, match="pickle"):
        _checkpoint_files(tmp_path)
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    with pytest.raises(TextModelLoadError, match="require an index"):
        _checkpoint_files(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(TextModelLoadError, match="unsafe shard"):
        _checkpoint_files(tmp_path)


def test_loader_remote_resolution_is_mocked_and_honors_offline(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"gpt2"}', encoding="utf-8")
    calls = []

    def fake_download(repo: str, filename: str, **kwargs: object) -> str:
        calls.append((repo, filename, kwargs))
        return str(config)

    monkeypatch.setattr("mlx_one.text.loading.hf_hub_download", fake_download)
    assert resolve_text_model_type("openai-community/gpt2", offline=True) == "gpt2"
    assert calls[0][2]["local_files_only"] is True


def test_cli_generate_contract(monkeypatch) -> None:
    result = GenerationResult("hello", " world", 1, 1, "length", "gpt2")
    monkeypatch.setattr("mlx_one.cli.generate_text", lambda *args, **kwargs: result)
    invoked = CliRunner().invoke(
        main, ["generate", "gpt2", "hello", "--max-tokens", "1", "--json-output"]
    )
    assert invoked.exit_code == 0
    assert json.loads(invoked.output)["text"] == " world"


def test_cli_stream_generate_contract(monkeypatch) -> None:
    from mlx_one.text import GenerationChunk

    monkeypatch.setattr(
        "mlx_one.cli.stream_text",
        lambda *args, **kwargs: iter(
            (GenerationChunk(1, "hi", 1), GenerationChunk(None, "", 1, "length"))
        ),
    )
    invoked = CliRunner().invoke(main, ["generate", "gpt2", "hello", "--stream"])
    assert invoked.exit_code == 0
    assert invoked.output == "hi\n"


def test_evaluation_worker_routes_gpt2_to_native_generation(monkeypatch, tmp_path: Path) -> None:
    from mlx_one import generation_worker

    result_path = tmp_path / "result.json"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "model_id": "openai-community/gpt2",
                "revision": "a" * 40,
                "prompts": ["hello"],
                "max_tokens": 2,
                "adapter_path": None,
                "result_path": str(result_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("mlx_one.text.loading.resolve_text_model_type", lambda *a, **k: "gpt2")
    monkeypatch.setattr(
        "mlx_one.text.generate",
        lambda *a, **k: (GenerationResult("hello", " world", 1, 1, "length", "gpt2"),),
    )
    monkeypatch.setattr(sys, "argv", ["mlx_one.generation_worker", str(request_path)])
    generation_worker.main()
    assert json.loads(result_path.read_text(encoding="utf-8"))["predictions"] == [" world"]


def test_native_gpt2_training_is_explicitly_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="inference only"):
        SFTTrainer(
            model="openai-community/gpt2",
            revision="a" * 40,
            train_dataset=tmp_path / "train.jsonl",
            args=object(),  # type: ignore[arg-type]
        )
