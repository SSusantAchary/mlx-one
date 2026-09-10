"""Opt-in tiny-MLX tests for native Sentence Transformers encoders."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig
from mlx_one.models.embeddings.bert.model import BertForSentenceEmbedding
from mlx_one.models.embeddings.mpnet.config import (
    MPNetEmbeddingConfig,
    mpnet_relative_position_bucket,
)
from mlx_one.models.embeddings.mpnet.model import (
    MPNetForSentenceEmbedding,
    create_position_ids_from_input_ids,
    relative_position_bucket,
)
from mlx_one.models.shared.encoder import masked_mean_pool


def bert_config(**updates: object) -> BertEmbeddingConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "intermediate_size": 24,
        "max_position_embeddings": 16,
        "type_vocab_size": 2,
        "sentence_max_length": 8,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }
    values.update(updates)
    return BertEmbeddingConfig.from_dict(values)


def mpnet_config(**updates: object) -> MPNetEmbeddingConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 12,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "intermediate_size": 24,
        "max_position_embeddings": 18,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 16,
        "sentence_max_length": 8,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }
    values.update(updates)
    return MPNetEmbeddingConfig.from_dict(values)


def test_bert_shapes_hidden_states_pooling_normalization_and_similarity() -> None:
    model = BertForSentenceEmbedding(bert_config())
    tokens = mx.array([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = mx.array([[True, True, True, False], [True, True, False, False]])
    output = model(tokens, attention_mask=mask, output_hidden_states=True)
    similarity = model.similarity(output.embeddings, output.embeddings)
    mx.eval(output.embeddings, output.last_hidden_state, output.pooler_output, similarity)
    assert output.embeddings.shape == (2, 12)
    assert output.last_hidden_state.shape == (2, 4, 12)
    assert output.pooler_output.shape == (2, 12)
    assert len(output.hidden_states) == 3
    assert mx.max(mx.abs(mx.sum(output.embeddings**2, axis=-1) - 1)).item() < 1e-5
    assert similarity.shape == (2, 2)
    assert mx.max(mx.abs(mx.diag(similarity) - 1)).item() < 1e-5


def test_bert_masked_padding_is_invariant_and_token_types_are_active() -> None:
    model = BertForSentenceEmbedding(bert_config())
    mask = mx.array([[True, True, False, False], [True, True, False, False]])
    tokens = mx.array([[1, 2, 3, 4], [1, 2, 7, 8]])
    output = model(tokens, attention_mask=mask)
    typed = model(tokens[:1], attention_mask=mask[:1], token_type_ids=mx.ones((1, 4), mx.int32))
    mx.eval(output.embeddings, typed.embeddings)
    assert mx.max(mx.abs(output.embeddings[0] - output.embeddings[1])).item() < 1e-5
    assert mx.max(mx.abs(output.embeddings[0] - typed.embeddings[0])).item() > 1e-6


def test_mpnet_shapes_positions_relative_bias_and_similarity() -> None:
    model = MPNetForSentenceEmbedding(mpnet_config())
    tokens = mx.array([[2, 3, 4, 1], [5, 6, 1, 1]])
    output = model(tokens, output_hidden_states=True)
    positions = create_position_ids_from_input_ids(mx.array([[0, 5, 6, 1, 7]]), 1)
    scalar_positions = list(range(-20, 21))
    vector_buckets = relative_position_bucket(
        mx.array(scalar_positions), num_buckets=8, max_distance=16
    )
    expected = mx.array(
        [
            mpnet_relative_position_bucket(value, num_buckets=8, max_distance=16)
            for value in scalar_positions
        ]
    )
    similarity = model.similarity(output.embeddings, output.embeddings)
    mx.eval(output.embeddings, output.last_hidden_state, positions, vector_buckets, similarity)
    assert output.embeddings.shape == (2, 12)
    assert output.last_hidden_state.shape == (2, 4, 12)
    assert output.pooler_output.shape == (2, 12)
    assert len(output.hidden_states) == 3
    assert positions.tolist() == [[2, 3, 4, 1, 5]]
    assert mx.array_equal(vector_buckets, expected).item()
    assert model.encoder.compute_position_bias(4).shape == (1, 3, 4, 4)
    assert mx.max(mx.abs(mx.diag(similarity) - 1)).item() < 1e-5


def test_mpnet_masked_padding_is_invariant_and_all_padding_is_finite_zero() -> None:
    model = MPNetForSentenceEmbedding(mpnet_config())
    mask = mx.array([[True, True, False, False], [True, True, False, False]])
    tokens = mx.array([[2, 3, 4, 5], [2, 3, 7, 8]])
    output = model(tokens, attention_mask=mask)
    empty = model(mx.array([[1, 1, 1]]))
    mx.eval(output.embeddings, empty.embeddings)
    assert mx.max(mx.abs(output.embeddings[0] - output.embeddings[1])).item() < 1e-5
    assert mx.all(mx.isfinite(empty.embeddings)).item()
    assert mx.max(mx.abs(empty.embeddings)).item() == 0


def test_raw_masked_mean_pool_and_unnormalized_configuration() -> None:
    hidden = mx.array([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = mx.array([[True, True, False]])
    pooled = masked_mean_pool(hidden, mask)
    model = MPNetForSentenceEmbedding(
        mpnet_config(normalize_embeddings=False, add_pooling_layer=False)
    )
    output = model(mx.array([[2, 3, 1]]))
    mx.eval(pooled, output.embeddings)
    assert pooled.tolist() == [[2.0, 4.0]]
    assert output.pooler_output is None
    assert mx.abs(mx.sum(output.embeddings**2) - 1).item() > 1e-4


@pytest.mark.parametrize(
    ("model", "tokens"),
    [
        (BertForSentenceEmbedding(bert_config(sentence_max_length=3)), mx.array([[1, 2, 3, 4]])),
        (MPNetForSentenceEmbedding(mpnet_config(sentence_max_length=3)), mx.array([[2, 3, 4, 5]])),
    ],
)
def test_sentence_length_profiles_are_enforced(model: object, tokens: object) -> None:
    with pytest.raises(ValueError, match="sentence_max_length"):
        model(tokens)
