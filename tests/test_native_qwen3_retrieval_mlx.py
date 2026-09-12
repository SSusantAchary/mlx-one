"""Opt-in Metal execution tests for native Qwen3 retrieval models."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.models.embeddings.qwen3_embedding.config import Qwen3EmbeddingConfig
from mlx_one.models.embeddings.qwen3_embedding.model import (
    Qwen3ForEmbedding,
    last_token_pool,
)
from mlx_one.models.embeddings.qwen3_embedding.weights import weight_contract
from mlx_one.models.embeddings.qwen3_reranker.config import Qwen3RerankerConfig
from mlx_one.models.embeddings.qwen3_reranker.model import Qwen3ForReranking
from mlx_one.retrieval.loading import LoadedRetrievalModel, load_retrieval_model
from mlx_one.retrieval.service import embed, rerank
from mlx_one.retrieval.tokenizer import Qwen3Tokenizer
from mlx_one.text.tokenizer import bytes_to_unicode


def _text_config() -> dict[str, object]:
    return {
        "model_type": "qwen3",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "tie_word_embeddings": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _embedding_config() -> Qwen3EmbeddingConfig:
    return Qwen3EmbeddingConfig.from_dict(
        {
            "model_type": "qwen3_embedding",
            "text_config": _text_config(),
            "min_dimension": 2,
            "max_dimension": 8,
            "max_seq_length": 32,
        }
    )


def _reranker_config() -> Qwen3RerankerConfig:
    return Qwen3RerankerConfig.from_dict(
        {
            "model_type": "qwen3_reranker",
            "text_config": _text_config(),
            "max_seq_length": 32,
        }
    )


def test_last_token_pool_supports_left_and_right_padding() -> None:
    hidden = mx.arange(2 * 4 * 3).reshape(2, 4, 3)
    mask = mx.array([[0, 1, 1, 1], [1, 1, 0, 0]])
    pooled = last_token_pool(hidden, mask)
    mx.eval(pooled)
    assert pooled.tolist() == [hidden[0, 3].tolist(), hidden[1, 1].tolist()]
    with pytest.raises(ValueError, match="active token"):
        last_token_pool(hidden[:1], mx.zeros((1, 4)))


def test_embedding_mask_dimensions_normalization_and_parameter_contract() -> None:
    config = _embedding_config()
    model = Qwen3ForEmbedding(config)
    left = model(
        mx.array([[0, 0, 3, 4], [0, 5, 6, 7]]),
        attention_mask=mx.array([[0, 0, 1, 1], [0, 1, 1, 1]]),
        dimensions=4,
        output_hidden_states=True,
    )
    unpadded = model(mx.array([[3, 4]]), attention_mask=mx.array([[1, 1]]), dimensions=4)
    mx.eval(left.embeddings, unpadded.embeddings)
    norms = mx.sqrt(mx.sum(left.embeddings * left.embeddings, axis=-1))
    assert left.embeddings.shape == (2, 4)
    assert left.last_hidden_state.shape == (2, 4, 8)
    assert len(left.hidden_states) == 3
    assert mx.max(mx.abs(norms - 1)).item() < 1e-5
    assert mx.max(mx.abs(left.embeddings[0] - unpadded.embeddings[0])).item() < 1e-4
    assert set(dict(tree_flatten(model.parameters()))) == set(weight_contract(config).expected)


def test_cosine_matrix_and_reranker_selected_logits() -> None:
    embedding = Qwen3ForEmbedding(_embedding_config())
    cosine = embedding.similarity(mx.array([[1.0, 0.0]]), mx.array([[1.0, 0.0], [0.0, 1.0]]))
    assert cosine.shape == (1, 2)
    assert cosine.tolist() == [[1.0, 0.0]]

    reranker = Qwen3ForReranking(_reranker_config())
    output = reranker(
        mx.array([[1, 2, 3], [4, 5, 6]]),
        attention_mask=mx.ones((2, 3)),
        false_token_id=7,
        true_token_id=8,
    )
    mx.eval(output.scores, output.probabilities, output.logits)
    expected = output.logits[:, 1] - output.logits[:, 0]
    assert output.scores.shape == (2,)
    assert output.probabilities.shape == (2,)
    assert mx.max(mx.abs(output.scores - expected)).item() == 0
    assert mx.all(output.probabilities >= 0).item()
    assert mx.all(output.probabilities <= 1).item()


def test_padding_mask_blocks_inactive_keys() -> None:
    model = Qwen3ForEmbedding(_embedding_config())
    padded = model(
        mx.array([[10, 11, 3, 4]]),
        attention_mask=mx.array([[0, 0, 1, 1]]),
        dimensions=8,
    )
    other_padding = model(
        mx.array([[20, 21, 3, 4]]),
        attention_mask=mx.array([[0, 0, 1, 1]]),
        dimensions=8,
    )
    mx.eval(padded.embeddings, other_padding.embeddings)
    assert mx.max(mx.abs(padded.embeddings - other_padding.embeddings)).item() < 1e-5


def _service_tokenizer() -> Qwen3Tokenizer:
    vocabulary = {character: index for index, character in bytes_to_unicode().items()}
    vocabulary.update({"ye": 256, "yes": 257, "no": 258})
    return Qwen3Tokenizer(
        vocabulary,
        (("y", "e"), ("ye", "s"), ("n", "o")),
        {"<|endoftext|>": 259},
        special_tokens={"<|endoftext|>"},
        post_suffix=(259,),
    )


def test_public_embedding_and_reranking_batch_services() -> None:
    tokenizer = _service_tokenizer()
    embedding_config = Qwen3EmbeddingConfig.from_dict(
        {
            "model_type": "qwen3_embedding",
            "text_config": {**_text_config(), "vocab_size": 300},
            "min_dimension": 2,
            "max_dimension": 8,
            "max_seq_length": 32,
        }
    )
    embedding_bundle = LoadedRetrievalModel(
        Qwen3ForEmbedding(embedding_config),
        tokenizer,
        "embedding",
        Path("."),
        "tiny-embedding",
        None,
    )
    embedded = embed(
        embedding_bundle,
        ["alpha", "beta", "gamma"],
        dimensions=4,
        max_length=32,
        batch_size=2,
    )
    assert len(embedded.embeddings) == 3
    assert all(len(vector) == 4 for vector in embedded.embeddings)
    assert embedded.model == "tiny-embedding"

    reranker_config = Qwen3RerankerConfig.from_dict(
        {
            "model_type": "qwen3_reranker",
            "text_config": {
                **_text_config(),
                "vocab_size": 300,
                "num_hidden_layers": 1,
                "max_position_embeddings": 512,
            },
            "max_seq_length": 512,
        }
    )
    reranker_bundle = LoadedRetrievalModel(
        Qwen3ForReranking(reranker_config),
        tokenizer,
        "reranking",
        Path("."),
        "tiny-reranker",
        None,
    )
    ranked = rerank(
        reranker_bundle,
        "query",
        ["first", "second", "third"],
        top_k=2,
        max_length=512,
        batch_size=2,
    )
    assert len(ranked.items) == 2
    assert ranked.items[0].score >= ranked.items[1].score
    assert {item.index for item in ranked.items} <= {0, 1, 2}
    assert all(0 <= item.probability <= 1 for item in ranked.items)


def test_safe_local_embedding_bundle_loads_exact_synthetic_weights(tmp_path: Path) -> None:
    text = {**_text_config(), "vocab_size": 300, "num_hidden_layers": 1}
    config = Qwen3EmbeddingConfig.from_dict(
        {
            **text,
            "word_embedding_dimension": 8,
            "max_seq_length": 32,
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(text), encoding="utf-8")
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps(
            {"pooling_mode_lasttoken": True, "word_embedding_dimension": 8}
        ),
        encoding="utf-8",
    )
    (tmp_path / "modules.json").write_text(
        json.dumps([{"type": "sentence_transformers.models.Pooling"}]),
        encoding="utf-8",
    )
    byte_vocabulary = {
        character: index for index, character in bytes_to_unicode().items()
    }
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": byte_vocabulary, "merges": []},
                "added_tokens": [
                    {"id": 259, "content": "<|endoftext|>", "special": True}
                ],
                "post_processor": {
                    "type": "TemplateProcessing",
                    "single": [
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {
                            "SpecialToken": {
                                "id": "<|endoftext|>",
                                "type_id": 0,
                            }
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<|endoftext|>"}), encoding="utf-8"
    )
    tensors = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in weight_contract(config).expected.items()
    }
    save_file(tensors, tmp_path / "model.safetensors")
    bundle = load_retrieval_model(tmp_path)
    assert bundle.task == "embedding"
    assert bundle.model.config.max_dimension == 8
