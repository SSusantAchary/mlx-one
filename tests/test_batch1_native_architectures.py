"""Opt-in native checks against small independently generated pinned HF fixtures."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on Apple Silicon", allow_module_level=True)

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_one.models.classification.vit import ViTForImageClassification
from mlx_one.models.classification.vit_config import ViTConfig
from mlx_one.models.classification.vit_weights import weight_contract as vit_contract
from mlx_one.models.embeddings.nomic_bert.config import NomicBertConfig
from mlx_one.models.embeddings.nomic_bert.model import NomicBertForEmbedding
from mlx_one.models.embeddings.nomic_bert.weights import weight_contract as nomic_contract
from mlx_one.models.embeddings.xlm_roberta.config import XLMRobertaConfig
from mlx_one.models.embeddings.xlm_roberta.model import XLMRobertaForRetrieval
from mlx_one.models.embeddings.xlm_roberta.weights import weight_contract as xlm_contract
from mlx_one.models.language.t5.config import T5Config
from mlx_one.models.language.t5.model import T5ForConditionalGeneration
from mlx_one.models.language.t5.weights import weight_contract as t5_contract


def weights(model):
    values = {}
    for name, tensor in tree_flatten(model.parameters()):
        phase = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "little") / 65536
        value = np.sin(np.arange(tensor.size, dtype=np.float64) * 0.13 + phase) * 0.03
        if "norm" in name.lower() and name.endswith("weight") and tensor.ndim == 1:
            value += 1
        if name == "vit.embeddings.patch_embeddings.projection.weight":
            shape = (tensor.shape[0], tensor.shape[3], tensor.shape[1], tensor.shape[2])
            value = value.reshape(shape).transpose(0, 2, 3, 1)
        else:
            value = value.reshape(tensor.shape)
        values[name] = mx.array(value.astype(np.float32))
    model.load_weights(list(values.items()), strict=True)
    model.eval()
    return values


@pytest.mark.parametrize("name", ["t5", "vit", "xlm-embedding", "xlm-reranking"])
def test_native_tensors_match_isolated_hf_reference(name):
    case = json.loads((Path(__file__).parent / "fixtures/batch1" / f"{name}.json").read_text())
    cfg = case["config"]
    if name == "t5":
        config = T5Config.from_dict(cfg)
        model, contract = T5ForConditionalGeneration(config), t5_contract(config)
    elif name == "vit":
        config = ViTConfig.from_dict(cfg)
        model, contract = ViTForImageClassification(config), vit_contract(config)
    else:
        cfg.update(
            task="reranking" if name == "xlm-reranking" else "embedding", sentence_max_length=8
        )
        config = XLMRobertaConfig.from_dict(cfg)
        model, contract = XLMRobertaForRetrieval(config), xlm_contract(config)
    contract.validate(weights(model))
    if name == "vit":
        result = model(mx.array(case["pixel_values"]))
    else:
        arguments = dict(attention_mask=mx.array(case["attention_mask"]))
        if name == "t5":
            arguments["decoder_input_ids"] = mx.array(case["decoder_input_ids"])
        result = model(mx.array(case["input_ids"]), **arguments)
    actual = result.logits if "logits" in case else result.last_hidden_state
    np.testing.assert_allclose(
        np.array(actual), case.get("logits", case.get("hidden")), atol=1e-5, rtol=1e-5
    )
    if name == "t5":
        np.testing.assert_allclose(
            np.array(result.encoder_last_hidden_state), case["encoder_hidden"], atol=1e-5, rtol=1e-5
        )
        generated = model.greedy(
            mx.array(case["input_ids"]),
            attention_mask=mx.array(case["attention_mask"]),
            max_new_tokens=4,
        )
        assert generated.tolist() == case["generation"]


def test_nomic_masking_batching_and_matryoshka_contract():
    config = NomicBertConfig(
        vocab_size=32,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_seq_length=16,
    )
    model = NomicBertForEmbedding(config)
    nomic_contract(config).validate(weights(model))
    ids = mx.array([[2, 5, 6, 3, 0], [2, 5, 6, 3, 0]])
    mask = mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    full = model(ids, attention_mask=mask)
    short = model(ids, attention_mask=mask, dimensions=64)
    hidden = full.last_hidden_state
    pooled = mx.sum(hidden * mask[..., None], axis=1) / mx.sum(mask, axis=1, keepdims=True)
    expected = (pooled - mx.mean(pooled, axis=-1, keepdims=True)) * mx.rsqrt(
        mx.var(pooled, axis=-1, keepdims=True) + 1e-5
    )
    expected = expected[:, :64]
    expected = expected / mx.sqrt(mx.sum(expected**2, axis=-1, keepdims=True))
    np.testing.assert_allclose(np.array(short.embeddings), np.array(expected), atol=1e-6)
    np.testing.assert_allclose(
        np.array(full.embeddings[0]),
        np.array(model(ids[:1], attention_mask=mask[:1]).embeddings[0]),
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        model(ids, attention_mask=mask, dimensions=63)


def test_bge_m3_auxiliary_heads_require_explicit_profile():
    config = XLMRobertaConfig.from_dict(
        dict(
            model_type="xlm-roberta",
            vocab_size=32,
            hidden_size=12,
            num_attention_heads=3,
            intermediate_size=24,
            num_hidden_layers=2,
            max_position_embeddings=18,
            sentence_max_length=8,
            multi_representation=True,
            colbert_dim=8,
        )
    )
    model = XLMRobertaForRetrieval(config)
    xlm_contract(config).validate(weights(model))
    result = model(mx.array([[0, 5, 6, 2, 1]]), representations=True)
    assert result.embeddings.shape == (1, 12)
    assert result.colbert_embeddings.shape == (1, 4, 8)
    assert result.sparse_weights.shape == (1, 5)
    assert np.all(np.array(result.sparse_weights) >= 0)
    assert np.all(np.array(result.colbert_embeddings[0, -1]) == 0)


def test_kokoro_real_fft_endpoint_phases_do_not_flip_branch():
    from mlx_one.models.audio.kokoro.vocoder import source_stft

    waveform = (-2 + np.sin(np.arange(100, dtype=np.float32) * 0.7))[None, :]
    padded = np.pad(waveform, ((0, 0), (10, 10)), mode="reflect")
    indices = np.arange(21)[:, None] * 5 + np.arange(20)[None, :]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(20) / 20)
    spectrum = np.fft.rfft(padded[:, indices] * window, axis=-1)
    actual = np.array(source_stft(mx.array(waveform)))
    np.testing.assert_allclose(actual[..., :11], np.abs(spectrum), atol=1e-5, rtol=1e-5)
    # DC is strictly negative here; a signed imaginary roundoff would yield -pi.
    np.testing.assert_allclose(actual[..., 11], np.pi, atol=1e-6)
    np.testing.assert_allclose(actual[..., [11, 21]], np.angle(spectrum)[..., [0, 10]], atol=1e-6)
