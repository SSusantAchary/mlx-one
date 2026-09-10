"""Opt-in tiny-MLX execution tests for Liquid Foundation Models."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.core.cache import make_hybrid_caches
from mlx_one.models.audio.lfm2_audio.config import Lfm2AudioConfig
from mlx_one.models.audio.lfm2_audio.model import Lfm2AudioForConditionalGeneration
from mlx_one.models.audio.lfm2_audio.processing import waveform_to_log_mel
from mlx_one.models.embeddings.lfm2_colbert.config import Lfm2ColBERTConfig
from mlx_one.models.embeddings.lfm2_colbert.model import Lfm2ColBERTModel
from mlx_one.models.language.lfm2.config import Lfm2Config
from mlx_one.models.language.lfm2.model import Lfm2ForCausalLM
from mlx_one.models.language.lfm2.weights import sanitize_weights as sanitize_dense_weights
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
from mlx_one.models.language.lfm2_moe.model import Lfm2MoeForCausalLM
from mlx_one.models.language.lfm2_moe.weights import sanitize_weights
from mlx_one.models.vision_language.lfm2_vl.config import Lfm2VLConfig
from mlx_one.models.vision_language.lfm2_vl.model import Lfm2VLForConditionalGeneration


def dense_config(**updates: object) -> Lfm2Config:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "conv_L_cache": 3,
        "intermediate_size": 24,
        "block_multiple_of": 4,
        "layer_types": ["conv", "full_attention", "conv"],
    }
    values.update(updates)
    return Lfm2Config.from_dict(values)


def test_lfm_dense_hybrid_shapes_and_cached_equivalence() -> None:
    config = dense_config()
    model = Lfm2ForCausalLM(config)
    tokens = mx.array([[1, 2, 3, 4]])
    full = model(tokens, output_hidden_states=True)
    caches = make_hybrid_caches(config.layer_types, config.conv_L_cache)
    cached = mx.concatenate(
        [model(tokens[:, i : i + 1], cache=caches).logits for i in range(4)], axis=1
    )
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 4, 64)
    assert full.last_hidden_state.shape == (1, 4, 16)
    assert len(full.hidden_states) == 4
    assert [cache.offset for cache in caches] == [4, 4, 4]
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4
    cleaned = sanitize_dense_weights(
        {"model.layers.0.conv.conv.weight": mx.zeros((16, 3, 1))}, config
    )
    assert cleaned["model.layers.0.conv.conv"].shape == (16, 3)


def test_lfm_moe_router_shapes_auxiliary_loss_and_stacking() -> None:
    config = Lfm2MoeConfig.from_dict(
        {
            **dense_config().__dict__,
            "model_type": "lfm2_moe",
            "num_dense_layers": 1,
            "moe_intermediate_size": 12,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "router_aux_loss_coef": 0.01,
            "output_router_logits": True,
        }
    )
    model = Lfm2MoeForCausalLM(config)
    output = model(mx.array([[1, 2, 3]]))
    mx.eval(output.logits, output.router_aux_loss)
    assert output.logits.shape == (1, 3, 64)
    assert len(output.router_logits) == 2
    assert output.router_logits[0].shape == (1, 3, 4)
    assert model.model.layers[1].feed_forward.top_indices.shape == (1, 3, 2)

    weights = {}
    for expert in range(4):
        weights[f"model.layers.1.feed_forward.experts.{expert}.w1.weight"] = mx.zeros((12, 16))
        weights[f"model.layers.1.feed_forward.experts.{expert}.w3.weight"] = mx.zeros((12, 16))
        weights[f"model.layers.1.feed_forward.experts.{expert}.w2.weight"] = mx.zeros((16, 12))
    cleaned = sanitize_weights(weights, config)
    assert cleaned["model.layers.1.feed_forward.experts.gate_proj"].shape == (4, 12, 16)
    assert cleaned["model.layers.1.feed_forward.experts.down_proj"].shape == (4, 16, 12)


def test_lfm_vl_vision_projection_and_token_replacement() -> None:
    config = Lfm2VLConfig.from_dict(
        {
            "model_type": "lfm2_vl",
            "image_token_id": 63,
            "text_config": dense_config().__dict__,
            "vision_config": {
                "hidden_size": 12,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_channels": 3,
                "patch_size": 2,
                "image_size": 4,
                "downsample_factor": 2,
                "projection_dim": 16,
            },
        }
    )
    model = Lfm2VLForConditionalGeneration(config)
    pixels = mx.zeros((1, 4, 12))
    output = model(mx.array([[1, 63, 2]]), pixel_values=pixels, spatial_shapes=[(2, 2)])
    mx.eval(output.logits)
    assert output.logits.shape == (1, 3, 64)
    assert output.vision_hidden_states.shape == (4, 12)


def test_lfm_colbert_normalization_and_maxsim() -> None:
    config = Lfm2ColBERTConfig.from_dict(
        {**dense_config().__dict__, "model_type": "lfm2_colbert", "projection_dim": 8}
    )
    model = Lfm2ColBERTModel(config)
    mask = mx.array([[True, True, False]])
    output = model(mx.array([[1, 2, 0]]), attention_mask=mask)
    score = model.maxsim(output.embeddings, output.embeddings, query_mask=mask, document_mask=mask)
    mx.eval(output.embeddings, score)
    assert output.embeddings.shape == (1, 3, 8)
    assert mx.max(mx.abs(output.embeddings[:, 2])).item() == 0
    assert score.shape == (1,)


def test_lfm_audio_waveform_to_logits_and_detokenized_waveform() -> None:
    config = Lfm2AudioConfig.from_dict(
        {
            "model_type": "lfm2_audio",
            "text_config": dense_config().__dict__,
            "audio_config": {
                "input_dim": 8,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "conv_kernel_size": 3,
                "subsampling_factor": 2,
            },
            "preprocessor_config": {
                "sampling_rate": 16000,
                "n_fft": 16,
                "window_length": 8,
                "hop_length": 4,
                "num_mel_bins": 8,
            },
            "depthformer_config": {
                "num_hidden_layers": 1,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "intermediate_size": 24,
            },
            "detokenizer_config": {
                "codebook_size": 17,
                "num_codebooks": 2,
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "intermediate_size": 24,
                "upsample_factor": 2,
                "n_fft": 16,
                "hop_length": 4,
                "sliding_window": 4,
            },
            "audio_vocab_size": 17,
            "audio_eos_token_id": 16,
            "num_codebooks": 2,
        }
    )
    features = waveform_to_log_mel(mx.zeros((20,)), config.preprocessor_config)
    model = Lfm2AudioForConditionalGeneration(config)
    # Four input frames become two encoder tokens under stride two.
    codes = mx.zeros((1, 2, 2), dtype=mx.int32)
    output = model(
        mx.array([[63, 63]]),
        input_features=features,
        audio_token_mask=mx.array([[True, True]]),
        audio_codes=codes,
        decode_audio=True,
    )
    mx.eval(output.text_logits, output.audio_logits, output.waveform)
    assert output.text_logits.shape == (1, 2, 64)
    assert output.audio_logits.shape == (1, 2, 2, 17)
    assert output.waveform.shape == (1, 28)
