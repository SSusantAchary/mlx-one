"""Batch 1 contracts: true paired tokenization and publisher pooling metadata."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from mlx_one.models.embeddings.bert.config import BertEmbeddingConfig
from mlx_one.models.embeddings.bert.weights import weight_contract
from mlx_one.retrieval.loading import (
    _checkpoint_prefixes,
    _sentence_encoder_config,
    detect_retrieval_task,
)
from mlx_one.text.tokenizers import HFTokenizerAdapter


def tokenizer(tmp_path):
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "where", "is", "paris", "france", "capital"]
        )
    }
    native = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    native.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text('{"pad_token":"[PAD]"}')
    return HFTokenizerAdapter.from_directory(tmp_path)


def test_pair_encoding_keeps_segments_and_terminal_separator(tmp_path):
    tok = tokenizer(tmp_path)
    ids, types = tok.encode_pair("where is paris", "france capital", max_length=7)
    assert ids == [2, 4, 5, 3, 7, 8, 3]
    assert types == [0, 0, 0, 0, 1, 1, 1]
    assert tok.encode_pair("where", "paris", max_length=20) == ([2, 4, 3, 6, 3], [0, 0, 0, 1, 1])
    # Truncation is request-local, never left enabled on the shared tokenizer.
    assert len(tok.encode("where is paris france capital", add_special_tokens=True)) == 7


def test_single_truncation_preserves_special_tokens(tmp_path):
    tok = tokenizer(tmp_path)
    assert tok.encode("where is paris france", add_special_tokens=True, max_length=4) == [
        2,
        4,
        5,
        3,
    ]
    with pytest.raises(ValueError, match="capacity"):
        tok.encode_pair("where", "paris", max_length=4)


@pytest.mark.parametrize("pooling,normalize", [("cls", True), ("mean", False)])
def test_sentence_pipeline_honors_pooling_and_normalization(tmp_path, pooling, normalize):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 384,
                "pooling_mode_cls_token": pooling == "cls",
                "pooling_mode_mean_tokens": pooling == "mean",
            }
        )
    )
    modules = [
        dict(type="sentence_transformers.models.Transformer", path=""),
        dict(type="sentence_transformers.models.Pooling", path="1_Pooling"),
    ]
    if normalize:
        modules.append(dict(type="sentence_transformers.models.Normalize", path="2_Normalize"))
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    (tmp_path / "sentence_bert_config.json").write_text('{"max_seq_length":128}')
    config = BertEmbeddingConfig.from_dict(
        _sentence_encoder_config(tmp_path, {"model_type": "bert", "hidden_size": 384})
    )
    assert (config.pooling, config.normalize_embeddings, config.sentence_max_length) == (
        pooling,
        normalize,
        128,
    )


def test_bert_reranker_requires_actual_pooler_and_scalar_head(tmp_path):
    config = BertEmbeddingConfig(task="reranking")
    expected = weight_contract(config).expected
    assert expected["classifier.weight"] == (1, 384)
    assert "pooler.dense.weight" in expected
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "id2label": {"0": "score"},
            }
        )
    )
    assert detect_retrieval_task(tmp_path) == "reranking"
    with pytest.raises(ValueError):
        BertEmbeddingConfig(task="reranking", num_labels=2)
    with pytest.raises(ValueError):
        BertEmbeddingConfig(task="reranking", add_pooling_layer=False)


def test_e5_remote_and_explicit_local_prefixes(tmp_path):
    assert _checkpoint_prefixes(tmp_path, "intfloat/multilingual-e5-small") == (
        "query: ",
        "passage: ",
    )
    (tmp_path / "mlx_one_retrieval.json").write_text(
        json.dumps({"query_prefix": "query: ", "document_prefix": "passage: "})
    )
    assert _checkpoint_prefixes(tmp_path, "local-copy") == ("query: ", "passage: ")


def test_default_and_explicit_nomic_768_have_distinct_batch_semantics(tmp_path):
    from mlx_one.engine.task_runners.retrieval import NomicEmbeddingAdapter
    from mlx_one.models.embeddings.nomic_bert.config import NomicBertConfig

    bundle = SimpleNamespace(
        architecture="nomic_bert",
        task="embedding",
        model=SimpleNamespace(config=NomicBertConfig(), parameters=lambda: {}),
        tokenizer=tokenizer(tmp_path),
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    adapter = NomicEmbeddingAdapter(bundle)
    default = adapter.prepare("embedding", dict(texts="france"))
    matryoshka = adapter.prepare("embedding", dict(texts="france", dimensions=768))
    assert default.semantic_key != matryoshka.semantic_key


def test_kokoro_rejects_architecture_changes_without_starting_metal():
    import copy

    from mlx_one.models.audio.kokoro.config import KokoroConfig

    original = json.loads(
        (Path(__file__).parent / "fixtures/batch1/kokoro-config.json").read_text()
    )
    assert KokoroConfig.from_dict(original).sample_rate == 24000
    for mutation in (
        {"dim_in": 128},
        {"text_encoder_kernel_size": 3},
        {"model_type": "t5"},
        {"vocab": {}},
        {"istftnet": {"gen_istft_n_fft": 32}},
    ):
        changed = copy.deepcopy(original)
        changed.update(mutation)
        with pytest.raises(ValueError):
            KokoroConfig.from_dict(changed)
