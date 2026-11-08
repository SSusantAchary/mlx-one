"""Publisher ColBERT metadata must preserve token roles and separate masks."""

import json
from types import SimpleNamespace

import pytest

from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
from mlx_one.retrieval.colbert import prepare_token_features
from mlx_one.retrieval.loading import RetrievalModelLoadError, _published_colbert_config


def config(**updates):
    data = dict(
        model_type="lfm2_colbert",
        text_config=dict(
            model_type="lfm2",
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            block_ff_dim=32,
            block_multiple_of=8,
            layer_types=["conv", "full_attention"],
        ),
        projection_dim=4,
        query_max_length=8,
        document_max_length=16,
        query_prefix="[Q] ",
        document_prefix="[D] ",
        do_query_expansion=True,
        skiplist_words=["."],
    )
    return Lfm2ColBERTConfig.from_dict({**data, **updates})


class Tokenizer:
    pad_token_id = 0
    _tokenizer = SimpleNamespace(
        token_to_id=lambda token: {"[Q] ": 3, "[D] ": 4, ".": 5}.get(token)
    )

    def encode(self, text, *, add_special_tokens, max_length):
        return [1, 6, 5][:max_length]


def test_query_expansion_attends_only_to_real_tokens_but_scores_expansion_vectors():
    ids, attention, scoring = prepare_token_features(
        Tokenizer(), config(), ["word."], input_type="query", max_length=8
    )
    assert ids == ((1, 3, 6, 5, 0, 0, 0, 0),)
    assert attention == ((True, True, True, True, False, False, False, False),)
    assert scoring == ((True,) * 8,)


def test_document_punctuation_stays_in_encoding_but_is_excluded_from_maxsim():
    ids, attention, scoring = prepare_token_features(
        Tokenizer(), config(), ["word."], input_type="document", max_length=16
    )
    assert ids == ((1, 4, 6, 5),)
    assert attention == ((True,) * 4,)
    assert scoring == ((True, True, True, False),)


def test_alternate_query_attention_flag_is_explicit():
    _, attention, _ = prepare_token_features(
        Tokenizer(),
        config(attend_to_expansion_tokens=True),
        ["word."],
        input_type="query",
        max_length=8,
    )
    assert attention == ((True,) * 8,)


def test_publisher_projection_pipeline_is_validated_before_admission(tmp_path):
    (tmp_path / "1_Dense").mkdir()
    modules = [
        dict(type="sentence_transformers.models.Transformer", path=""),
        dict(type="pylate.models.Dense.Dense", path="1_Dense"),
    ]
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    dense = dict(
        in_features=16,
        out_features=4,
        bias=False,
        activation_function="torch.nn.modules.linear.Identity",
    )
    (tmp_path / "1_Dense/config.json").write_text(json.dumps(dense))
    recipe = dict(
        similarity_fn_name="MaxSim",
        query_prefix="[Q] ",
        document_prefix="[D] ",
        query_length=8,
        document_length=16,
        do_query_expansion=True,
        attend_to_expansion_tokens=False,
        skiplist_words=["."],
    )
    (tmp_path / "config_sentence_transformers.json").write_text(json.dumps(recipe))
    result = _published_colbert_config(tmp_path, dict(hidden_size=16))
    assert result["projection_dim"] == 4 and result["do_query_expansion"] is True
    (tmp_path / "1_Dense/config.json").write_text(json.dumps({**dense, "bias": True}))
    with pytest.raises(RetrievalModelLoadError, match="projection"):
        _published_colbert_config(tmp_path, dict(hidden_size=16))
