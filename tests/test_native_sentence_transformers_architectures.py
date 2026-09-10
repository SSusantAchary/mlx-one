"""Backend-free contracts for native Sentence Transformers encoders."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_one.core import ConfigError, WeightContractError
from mlx_one.core.registry import get_registration, registered_model_types
from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig
from mlx_one.models.embeddings.bert.weights import (
    sanitize_weights as sanitize_bert_weights,
)
from mlx_one.models.embeddings.bert.weights import weight_contract as bert_weight_contract
from mlx_one.models.embeddings.mpnet.config import (
    MPNetEmbeddingConfig,
    mpnet_relative_position_bucket,
)
from mlx_one.models.embeddings.mpnet.weights import (
    sanitize_weights as sanitize_mpnet_weights,
)
from mlx_one.models.embeddings.mpnet.weights import weight_contract as mpnet_weight_contract


@dataclass(frozen=True)
class Shaped:
    shape: tuple[int, ...]


def shaped(contract: object) -> dict[str, Shaped]:
    return {name: Shaped(shape) for name, shape in contract.expected.items()}


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
    }
    values.update(updates)
    return MPNetEmbeddingConfig.from_dict(values)


def test_official_sentence_transformer_profiles() -> None:
    minilm = BertEmbeddingConfig()
    assert (
        minilm.hidden_size,
        minilm.num_hidden_layers,
        minilm.num_attention_heads,
        minilm.intermediate_size,
    ) == (384, 6, 12, 1536)
    assert minilm.max_position_embeddings == 512
    assert minilm.sentence_max_length == 256

    mpnet = MPNetEmbeddingConfig()
    assert (
        mpnet.hidden_size,
        mpnet.num_hidden_layers,
        mpnet.num_attention_heads,
        mpnet.intermediate_size,
    ) == (768, 12, 12, 3072)
    assert mpnet.max_position_embeddings == 514
    assert mpnet.sentence_max_length == 384
    assert mpnet.relative_attention_num_buckets == 32


def test_sentence_encoder_registry_entries_are_lazy_and_backend_free() -> None:
    assert {"bert", "mpnet"} <= set(registered_model_types())
    bert = get_registration("bert")
    mpnet = get_registration("mpnet")
    assert bert.config_class() is BertEmbeddingConfig
    assert mpnet.config_class() is MPNetEmbeddingConfig
    assert bert.modality == mpnet.modality == "embedding"
    assert bert.capabilities == {"forward", "mean-pooling", "normalize", "cosine"}
    assert mpnet.capabilities == bert.capabilities


def test_config_aliases_and_unknown_fields_are_preserved() -> None:
    bert = BertEmbeddingConfig.from_dict({"max_seq_length": 128, "future_field": 7})
    mpnet = MPNetEmbeddingConfig.from_dict({"max_seq_length": 256, "future_field": 9})
    assert bert.sentence_max_length == 128
    assert mpnet.sentence_max_length == 256
    assert bert.extra == {"future_field": 7}
    assert mpnet.extra == {"future_field": 9}


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "roberta"}, "model_type"),
        ({"hidden_size": 10}, "divisible"),
        ({"hidden_act": "relu"}, "activation"),
        ({"position_embedding_type": "relative_key"}, "absolute"),
        ({"hidden_dropout_prob": 1.0}, r"\[0, 1\)"),
        ({"sentence_max_length": 17}, "cannot exceed"),
        ({"pad_token_id": 32}, "vocabulary"),
        ({"normalize_embeddings": 1}, "boolean"),
    ],
)
def test_bert_rejects_invalid_configs(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        bert_config(**updates)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_type": "bert"}, "model_type"),
        ({"hidden_size": 10}, "divisible"),
        ({"hidden_act": "relu"}, "activation"),
        ({"relative_attention_num_buckets": 7}, "divisible"),
        ({"relative_attention_num_buckets": 2}, "at least four"),
        ({"relative_attention_max_distance": 2}, "too small"),
        ({"attention_probs_dropout_prob": -0.1}, r"\[0, 1\)"),
        ({"sentence_max_length": 17}, "capacity"),
        ({"pad_token_id": 32}, "vocabulary"),
    ],
)
def test_mpnet_rejects_invalid_configs(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        mpnet_config(**updates)


def test_mpnet_scalar_relative_position_buckets_cover_both_directions() -> None:
    bucket = mpnet_relative_position_bucket
    assert bucket(0) == 0
    assert bucket(-1) == 1
    assert bucket(1) == 17
    assert bucket(-7) == 7
    assert bucket(-8) == 8
    assert bucket(-128) == 15
    assert bucket(128) == 31
    assert bucket(-10_000) == 15
    assert bucket(10_000) == 31


def test_bert_weight_contract_sanitizer_and_strict_rejection() -> None:
    config = bert_config()
    contract = bert_weight_contract(config)
    contract.validate(shaped(contract))
    assert contract.expected["encoder.layer.0.attention.self_attn.query.weight"] == (12, 12)
    assert contract.expected["encoder.layer.1.intermediate.dense.weight"] == (24, 12)
    cleaned = sanitize_bert_weights(
        {
            "bert.embeddings.position_ids": Shaped((1, 16)),
            "bert.embeddings.token_type_ids": Shaped((1, 16)),
            "bert.encoder.layer.0.attention.self.query.weight": Shaped((12, 12)),
        },
        config,
    )
    assert set(cleaned) == {"encoder.layer.0.attention.self_attn.query.weight"}
    invalid = shaped(contract)
    invalid["cls.predictions.decoder.weight"] = Shaped((32, 12))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)
    mismatched = shaped(contract)
    mismatched["embeddings.word_embeddings.weight"] = Shaped((31, 12))
    with pytest.raises(WeightContractError, match="shape mismatches"):
        contract.validate(mismatched)


def test_mpnet_weight_contract_sanitizer_and_strict_rejection() -> None:
    config = mpnet_config()
    contract = mpnet_weight_contract(config)
    contract.validate(shaped(contract))
    assert contract.expected["encoder.relative_attention_bias.weight"] == (8, 3)
    assert contract.expected["encoder.layer.0.attention.attn.q.weight"] == (12, 12)
    cleaned = sanitize_mpnet_weights(
        {
            "mpnet.embeddings.position_ids": Shaped((1, 18)),
            "mpnet.encoder.relative_attention_bias.weight": Shaped((8, 3)),
        },
        config,
    )
    assert set(cleaned) == {"encoder.relative_attention_bias.weight"}
    invalid = shaped(contract)
    invalid["lm_head.weight"] = Shaped((32, 12))
    with pytest.raises(WeightContractError, match="unexpected tensors"):
        contract.validate(invalid)
    mismatched = shaped(contract)
    mismatched["encoder.relative_attention_bias.weight"] = Shaped((8, 4))
    with pytest.raises(WeightContractError, match="shape mismatches"):
        contract.validate(mismatched)


def test_pooler_can_be_excluded_from_both_weight_contracts() -> None:
    bert = bert_weight_contract(bert_config(add_pooling_layer=False))
    mpnet = mpnet_weight_contract(mpnet_config(add_pooling_layer=False))
    assert not any(name.startswith("pooler.") for name in bert.expected)
    assert not any(name.startswith("pooler.") for name in mpnet.expected)
