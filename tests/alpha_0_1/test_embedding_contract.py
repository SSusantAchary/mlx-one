"""Reject incompatible pooling instead of silently returning a different vector."""

import json

import pytest
from test_dense_batching import native as native
from test_task_retrieval_runtime import bundle

from mlx_one.retrieval.loading import (
    RetrievalModelLoadError,
    _load_colbert,
    _load_sentence_encoder,
    _sentence_encoder_config,
)
from mlx_one.retrieval.service import embed


def assets(root, *, pooling=None, normalize=True):
    (root / "1_Pooling").mkdir(exist_ok=True)
    (root / "1_Pooling/config.json").write_text(json.dumps(pooling or {
        "pooling_mode_mean_tokens": True, "word_embedding_dimension": 384}))
    (root / "sentence_bert_config.json").write_text('{"max_seq_length": 256}')
    modules = [{"type": "sentence_transformers.models.Transformer", "path": ""},
               {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}]
    if normalize:
        modules.append({"type": "sentence_transformers.models.Normalize", "path": "2_Normalize"})
    (root / "modules.json").write_text(json.dumps(modules))


@pytest.mark.parametrize("normalize", [False, True])
def test_normalization_follows_pipeline_independently_of_mean_pooling(tmp_path, normalize):
    assets(tmp_path, normalize=normalize)
    config = _sentence_encoder_config(tmp_path, {"hidden_size": 384})
    assert config["normalize_embeddings"] is normalize
    assert config["sentence_max_length"] == 256


@pytest.mark.parametrize("updates", [
    {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True},
    {"pooling_mode_cls_token": True}, {"pooling_mode_mean_tokens": 1},
    {"word_embedding_dimension": 768}, {"pooling_mode_weightedmean_tokens": True},
])
def test_unsupported_pooling_or_width_fails_before_native_load(tmp_path, updates):
    assets(tmp_path, pooling={"pooling_mode_mean_tokens": True,
                             "word_embedding_dimension": 384, **updates})
    with pytest.raises(ValueError, match="pooling"):
        _sentence_encoder_config(tmp_path, {"hidden_size": 384})


def test_unimplemented_dense_module_is_not_silently_ignored(tmp_path):
    assets(tmp_path)
    modules = json.loads((tmp_path / "modules.json").read_text())
    modules.append({"type": "sentence_transformers.models.Dense", "path": "3_Dense"})
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    with pytest.raises(ValueError, match="module pipeline"):
        _sentence_encoder_config(tmp_path, {"hidden_size": 384})


def test_small_encoder_default_fits_native_context_and_handles_unicode(native, tmp_path):
    model = bundle("bert", tmp_path)
    result = embed(model, ["hello", "नमस्ते"])
    assert result.dimensions == 2
    assert len(result.embeddings) == 2
    with pytest.raises(ValueError, match="exceeds the model"):
        embed(model, "hello", max_length=257)


@pytest.mark.parametrize("kind", ["bert", "colbert"])
def test_bad_quantization_rejected_before_tokenizer_weights_or_model(tmp_path, kind):
    with pytest.raises(RetrievalModelLoadError, match="quantization metadata"):
        if kind == "bert":
            _load_sentence_encoder(tmp_path, "org/model", None, "bert", {"quantization": False})
        else:
            _load_colbert(tmp_path, "org/model", None, {"quantization": False})
