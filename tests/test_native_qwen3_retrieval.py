"""Backend-free contracts for native Qwen3 embedding and reranking."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlx_one.cli import main
from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.embeddings.qwen3_embedding.weights import (
    sanitize_weights as sanitize_embedding_weights,
)
from mlx_one.models.embeddings.qwen3_embedding.weights import (
    weight_contract as embedding_weight_contract,
)
from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
from mlx_one.retrieval.loading import (
    RetrievalModelLoadError,
    _checkpoint_files,
    _resolve,
    detect_retrieval_task,
)
from mlx_one.retrieval.schemas import EmbeddingResult, RerankItem, RerankResult
from mlx_one.retrieval.service import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    RERANK_PREFIX,
    RERANK_SUFFIX,
)
from mlx_one.retrieval.tokenizer import Qwen3Tokenizer
from mlx_one.text.tokenizer import bytes_to_unicode


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def _tiny_text_config(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 300,
        "max_position_embeddings": 32,
        "tie_word_embeddings": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(updates)
    return values


def _write_tokenizer(root: Path) -> None:
    vocabulary = {character: index for index, character in bytes_to_unicode().items()}
    next_id = max(vocabulary.values()) + 1
    for token in ("ye", "yes", "no"):
        vocabulary[token] = next_id
        next_id += 1
    added = [
        {"id": next_id, "content": "<|endoftext|>", "special": True},
        {"id": next_id + 1, "content": "<think>", "special": False},
    ]
    payload = {
        "normalizer": {"type": "NFC"},
        "model": {
            "type": "BPE",
            "vocab": vocabulary,
            "merges": [["y", "e"], ["ye", "s"], ["n", "o"]],
        },
        "added_tokens": added,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            ],
        },
    }
    (root / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<|endoftext|>"}), encoding="utf-8"
    )


def test_official_profiles_and_dimension_validation() -> None:
    embedding = Qwen3EmbeddingConfig()
    reranker = Qwen3RerankerConfig()
    assert (
        embedding.text_config.num_hidden_layers,
        embedding.text_config.hidden_size,
        embedding.text_config.intermediate_size,
    ) == (28, 1024, 3072)
    assert embedding.text_config.num_attention_heads == 16
    assert embedding.text_config.num_key_value_heads == 8
    assert embedding.text_config.head_dim == 128
    assert embedding.max_seq_length == 32768
    assert reranker.max_seq_length == 40960
    assert (embedding.bos_token_id, embedding.eos_token_id) == (151643, 151643)
    assert (reranker.bos_token_id, reranker.eos_token_id) == (151643, 151645)
    assert embedding.validate_dimensions(32) == 32
    assert embedding.validate_dimensions(1024) == 1024
    with pytest.raises(ValueError, match="within"):
        embedding.validate_dimensions(16)
    with pytest.raises(ValueError, match="integer"):
        embedding.validate_dimensions(True)


def test_configs_reject_invalid_task_specific_settings() -> None:
    with pytest.raises(ConfigError, match="model_type"):
        Qwen3EmbeddingConfig.from_dict(
            {"model_type": "wrong", "text_config": _tiny_text_config()}
        )
    with pytest.raises(ConfigError, match="tied"):
        Qwen3RerankerConfig.from_dict(
            {
                "model_type": "qwen3_reranker",
                "text_config": _tiny_text_config(tie_word_embeddings=False),
            }
        )
    with pytest.raises(ConfigError, match="word_embedding_dimension"):
        Qwen3EmbeddingConfig.from_dict(
            {**_tiny_text_config(), "word_embedding_dimension": None}
        )
    tiny = Qwen3EmbeddingConfig.from_dict(
        {**_tiny_text_config(), "word_embedding_dimension": 8}
    )
    assert (tiny.min_dimension, tiny.max_dimension) == (8, 8)


def test_registry_is_lazy_and_declares_retrieval_capabilities() -> None:
    assert {"qwen3_embedding", "qwen3_reranker"} <= set(registered_model_types())
    embedding = get_registration("qwen3_embedding")
    reranker = get_registration("qwen3_reranker")
    assert embedding.modality == "embedding"
    assert embedding.capabilities >= {"last-token-pooling", "dimensions", "cosine"}
    assert reranker.modality == "reranking"
    assert reranker.capabilities >= {"pair-scoring", "rerank", "probability"}
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx_one; "
            "assert 'mlx_one.models.embeddings.qwen3_embedding.model' not in sys.modules; "
            "assert 'mlx_one.models.embeddings.qwen3_reranker.model' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_embedding_weight_contract_is_exact_and_cleans_tied_buffers() -> None:
    config = Qwen3EmbeddingConfig.from_dict(
        {**_tiny_text_config(), "word_embedding_dimension": 8}
    )
    contract = embedding_weight_contract(config)
    tensors = {name: Shaped(shape) for name, shape in contract.expected.items()}
    tensors["lm_head.weight"] = Shaped((300, 8))
    tensors["model.rotary_emb.inv_freq"] = Shaped((2,))
    clean = sanitize_embedding_weights(tensors, config)
    contract.validate(clean)
    assert "lm_head.weight" not in clean
    assert "model.rotary_emb.inv_freq" not in clean
    with pytest.raises(WeightContractError, match="unexpected"):
        contract.validate({**clean, "surprise.weight": Shaped((1,))})
    broken = dict(clean)
    broken["model.embed_tokens.weight"] = Shaped((1, 1))
    with pytest.raises(WeightContractError, match="shape mismatches"):
        contract.validate(broken)


def test_tokenizer_nfc_added_tokens_padding_and_yes_no(tmp_path: Path) -> None:
    _write_tokenizer(tmp_path)
    tokenizer = Qwen3Tokenizer.from_directory(tmp_path)
    assert tokenizer.decode(tokenizer.encode("e\u0301")) == "é"
    assert len(tokenizer.encode("yes")) == 1
    assert len(tokenizer.encode("no")) == 1
    ordinary = tokenizer.encode("<think>")
    assert tokenizer.decode(ordinary) == "<think>"
    special = tokenizer.encode("<|endoftext|>")
    assert tokenizer.decode(special) == ""
    assert tokenizer.decode(special, skip_special_tokens=False) == "<|endoftext|>"
    processed = tokenizer.encode("hello", add_special_tokens=True, max_length=4)
    assert len(processed) == 4
    assert processed[-1] == tokenizer.pad_token_id
    ids, masks = tokenizer.pad([[1], [1, 2]], padding_side="left")
    assert ids[0][0] == tokenizer.pad_token_id
    assert masks == [[False, True], [True, True]]


def test_metadata_task_detection_and_checkpoint_security(tmp_path: Path) -> None:
    embedding = tmp_path / "embedding"
    (embedding / "1_Pooling").mkdir(parents=True)
    (embedding / "modules.json").write_text(
        json.dumps([{"type": "sentence_transformers.models.Pooling"}]),
        encoding="utf-8",
    )
    (embedding / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    assert detect_retrieval_task(embedding) == "embedding"

    reranker = tmp_path / "reranker"
    reranker.mkdir()
    (reranker / "config_sentence_transformers.json").write_text(
        json.dumps({"model_type": "CrossEncoder"}), encoding="utf-8"
    )
    assert detect_retrieval_task(reranker) == "reranking"

    module_reranker = tmp_path / "module-reranker"
    module_reranker.mkdir()
    (module_reranker / "modules.json").write_text(
        json.dumps(
            [
                {
                    "type": (
                        "sentence_transformers.cross_encoder.modules."
                        "logit_score.LogitScore"
                    )
                }
            ]
        ),
        encoding="utf-8",
    )
    assert detect_retrieval_task(module_reranker) == "reranking"

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "pytorch_model.bin").write_bytes(b"pickle")
    with pytest.raises(RetrievalModelLoadError, match="pickle"):
        _checkpoint_files(unsafe)

    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "one.safetensors").write_bytes(b"")
    (shards / "two.safetensors").write_bytes(b"")
    with pytest.raises(RetrievalModelLoadError, match="require an index"):
        _checkpoint_files(shards)
    (shards / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}
        ),
        encoding="utf-8",
    )
    files, weight_map = _checkpoint_files(shards)
    assert {path.name for path in files} == {"one.safetensors", "two.safetensors"}
    assert weight_map == {"a": "one.safetensors", "b": "two.safetensors"}


def test_remote_resolution_is_explicit_and_offline_aware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot(model: str, **kwargs: object) -> str:
        calls.update(model=model, **kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_one.retrieval.loading.snapshot_download", fake_snapshot)
    resolved = _resolve(
        "Qwen/Qwen3-Embedding-0.6B",
        revision="abc",
        offline=True,
        cache_dir=None,
    )
    assert resolved == tmp_path
    assert calls["revision"] == "abc"
    assert calls["local_files_only"] is True
    assert "*.safetensors" in calls["allow_patterns"]


def test_prompts_and_result_serialization_are_stable() -> None:
    assert DEFAULT_RETRIEVAL_INSTRUCTION.startswith("Given a web search query")
    assert RERANK_PREFIX.startswith("<|im_start|>system\nJudge whether")
    assert RERANK_SUFFIX.endswith("</think>\n\n")
    embedded = EmbeddingResult(("a",), ((1.0, 0.0),), 2, "document", "local")
    ranked = RerankResult("q", (RerankItem(1, "b", 2.0, 0.8),), "local")
    assert json.loads(embedded.to_json())["dimensions"] == 2
    assert json.loads(ranked.to_json())["items"][0]["index"] == 1


def test_retrieval_cli_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mlx_one.cli.embed_texts",
        lambda *args, **kwargs: EmbeddingResult(
            ("hello",), ((1.0, 0.0),), 2, "document", "fake"
        ),
    )
    monkeypatch.setattr(
        "mlx_one.cli.rerank_documents",
        lambda *args, **kwargs: RerankResult(
            "query", (RerankItem(1, "second", 1.5, 0.8),), "fake"
        ),
    )
    runner = CliRunner()
    embedded = runner.invoke(main, ["embed", "fake", "hello", "--json-output"])
    ranked = runner.invoke(
        main, ["rerank", "fake", "query", "first", "second", "--json-output"]
    )
    assert embedded.exit_code == 0
    assert json.loads(embedded.output)["dimensions"] == 2
    assert ranked.exit_code == 0
    assert json.loads(ranked.output)["items"][0]["index"] == 1
