# Author: Susant Achary
"""Short native dispatch and reference parity checks for the remaining registry types."""

import json
import os
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
from ocr.conftest import tiny_config as ocr_config
from test_audio_pipeline_native import tiny_config as native_audio_config
from test_native_lfm_mlx import dense_config
from test_native_sentence_transformers_mlx import bert_config, mpnet_config
from test_native_whisper_mlx import config as whisper_config
from test_parakeet_mlx import tiny_model as parakeet_model

from mlx_one.core.registry import get_registration
from mlx_one.engine.arena import SegmentedPageArena
from mlx_one.engine.paged import PagedKVCache
from mlx_one.engine.paged_attention import paged_attention
from mlx_one.mlxboost.gptoss import gptoss_execution
from mlx_one.models.audio.kokoro.blocks import AlbertLayer
from mlx_one.models.audio.lfm2_audio.native import NativeLFMAudio
from mlx_one.models.segmentation.sam3._runtime import Tensor, raw
from mlx_one.models.segmentation.sam3.config import Sam3VisionConfig
from mlx_one.models.shared import metal_activations, metal_reductions
from mlx_one.ocr.loading import make_config, make_model

FIXTURES = Path(__file__).parent / "fixtures" / "batch1"
_IDS = [[3, 4, 2]]


@pytest.fixture
def compare_metal(monkeypatch):
    def compare(forward, *, option="GELU", kernel="_gelu_kernel", exact=False):
        module = metal_reductions if option == "REDUCTIONS" else metal_activations
        flag = "MLX_ONE_METAL_" + option
        monkeypatch.delenv(flag, raising=False)
        expected = forward()
        mx.eval(expected)
        original = getattr(module, kernel)()
        launches = []

        def observe(**kwargs):
            launches.append(kwargs["grid"])
            return original(**kwargs)

        monkeypatch.setattr(module, kernel, lambda: observe)
        monkeypatch.setenv(flag, "1")
        actual = forward()
        assert launches, f"{kernel} was not reached from the model"
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        if exact:
            assert mx.array_equal(actual, expected).item()
        else:
            assert mx.allclose(actual, expected, atol=5e-6, rtol=1e-5).item()

    return compare


def model_for(family, **config):
    entry = get_registration(family)
    return entry.model_class()(entry.config_class().from_dict({"model_type": family, **config}))


@pytest.mark.parametrize("name,form,stock", [
    ("erf", 0, nn.gelu), ("tanh", 1, nn.gelu_approx), ("quick", 2, nn.gelu_fast_approx),
])
def test_gelu_forms_strided_range(name, form, stock, monkeypatch):
    monkeypatch.setenv("MLX_ONE_METAL_GELU", "1")
    x = mx.linspace(-12, 12, 2054).reshape(1027, 2)[:, 1]
    actual = metal_activations.try_gelu(x, form=form)
    assert mx.allclose(actual, stock(x), atol=5e-7, rtol=1e-6).item(), name
    assert metal_activations.try_gelu(x.astype(mx.bfloat16), form=form) is None


@pytest.mark.parametrize("family", ["bert", "bert_token_classification", "mpnet", "xlm-roberta"])
def test_encoder_gelu(family, compare_metal):
    if family == "bert":
        model = get_registration(family).model_class()(bert_config())
    elif family == "mpnet":
        model = get_registration(family).model_class()(mpnet_config())
    elif family == "bert_token_classification":
        model = model_for(family, vocab_size=32, hidden_size=12, num_hidden_layers=2,
                          num_attention_heads=3, intermediate_size=24, max_position_embeddings=16,
                          sentence_max_length=16, model_type="bert")
    else:
        case = json.loads((FIXTURES / "xlm-embedding.json").read_text())
        config = {**case["config"], "task": "embedding", "sentence_max_length": 8}
        model = model_for(family, **config)
    model.eval()
    ids = mx.array(_IDS)
    field = "logits" if family == "bert_token_classification" else "embeddings"
    compare_metal(lambda: getattr(model(ids, attention_mask=mx.ones_like(ids)), field))


def test_distilbert_gelu(compare_metal):
    model = model_for("distilbert", vocab_size=32, dim=16, hidden_dim=24, n_heads=4, n_layers=1)
    ids = mx.array(_IDS)
    compare_metal(lambda: model(ids, mx.ones_like(ids)))


def test_clip_quick_gelu(compare_metal):
    tower = dict(hidden_size=16, intermediate_size=24, num_attention_heads=4, num_hidden_layers=1,
                 vocab_size=32, max_position_embeddings=8, image_size=224, patch_size=32)
    model = model_for("clip", projection_dim=8, logit_scale_init_value=1.0,
                      text_config=tower, vision_config=tower)
    ids = mx.array(_IDS)
    pixels = mx.random.normal((1, 224, 224, 3), key=mx.random.key(7))
    compare_metal(lambda: model(pixels, ids, mx.ones_like(ids))["logits_per_image"])


def test_vit_gelu(compare_metal):
    case = json.loads((FIXTURES / "vit.json").read_text())
    model = model_for("vit", **{k: v for k, v in case["config"].items() if k != "model_type"})
    model.eval()
    compare_metal(lambda: model(mx.array(case["pixel_values"])).logits)


def test_t5_norm(compare_metal):
    case = json.loads((FIXTURES / "t5.json").read_text())
    model = model_for("t5", **{k: v for k, v in case["config"].items() if k != "model_type"})
    compare_metal(lambda: model(mx.array(case["input_ids"]),
                                decoder_input_ids=mx.array(case["decoder_input_ids"])).logits,
                  option="REDUCTIONS", kernel="_t5_norm_kernel")


def test_gemma_gated_gelu(compare_metal):
    model = model_for("gemma3_text", vocab_size=32, hidden_size=16, intermediate_size=24,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=4, sliding_window=8, sliding_window_pattern=2,
                      query_pre_attn_scalar=4)
    compare_metal(lambda: model(mx.array(_IDS)).logits)


def test_gpt_oss_existing_attention():
    model = model_for("gpt_oss", vocab_size=32, hidden_size=64, intermediate_size=64,
                      num_hidden_layers=1, num_local_experts=4, layer_types=["full_attention"],
                      num_attention_heads=64, num_key_value_heads=8, head_dim=64)
    cache = model.make_cache()
    mx.eval(model(mx.array(_IDS), cache=cache).logits)
    expected = model(mx.array([[7]]), cache=cache).logits
    mx.eval(expected)
    cache = model.make_cache()
    mx.eval(model(mx.array(_IDS), cache=cache).logits)
    with gptoss_execution("gqa_sinks") as state:
        actual = model(mx.array([[7]]), cache=cache).logits
        assert state.dispatch["attention candidate: gqa_sinks"] == 1
        assert mx.allclose(actual, expected, atol=5e-6, rtol=1e-5).item()


@pytest.mark.parametrize("dim,dtype,count", [(64, mx.bfloat16, 63), (128, mx.float16, 545)])
def test_paged_decode_parallel_pages_match_masked_sdpa(dim, dtype, count):
    q = mx.random.normal((1, 4, 1, dim), key=mx.random.key(61)).astype(dtype)
    k = mx.random.normal((1, 2, count, dim), key=mx.random.key(62)).astype(dtype)
    v = mx.random.normal(k.shape, key=mx.random.key(63)).astype(dtype)
    pool = SegmentedPageArena(16 * 1024**2, block_tokens=32, retain_free=False)
    cache = PagedKVCache(pool, dtype=str(dtype).split(".")[-1])
    try:
        cache.append(k, v)
        cutoff = count - 9
        scale = dim**-0.5
        mask = (mx.arange(count) <= cutoff)[None, :]
        expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        actual = paged_attention(q, (cache,), ((cutoff,),), (1,), scale=scale)
        assert actual.shape == q.shape and actual.dtype == dtype
        assert mx.allclose(actual, expected, atol=2e-3, rtol=4e-3).item()
    finally:
        cache.reset()


@pytest.mark.parametrize("family", ["trocr", "florence2"])
def test_ocr_vision_gelu(family, compare_metal):
    model = make_model(make_config(ocr_config(family), family), family)
    model.eval()
    pixels = mx.random.normal((1, 16, 16, 3), key=mx.random.key(19))
    if family == "florence2":
        compare_metal(lambda: model._encode_image(pixels.transpose(0, 3, 1, 2)))
    else:
        compare_metal(lambda: model.encode(pixels))


def test_whisper_gelu(compare_metal):
    model = get_registration("whisper").model_class()(whisper_config())
    model.eval()
    features = mx.random.normal((1, 80, 8), key=mx.random.key(15))
    compare_metal(lambda: model(features, mx.array(_IDS)).logits)


def test_kokoro_tanh_gelu(compare_metal):
    block = AlbertLayer()
    block.eval()
    x = mx.random.normal((1, 3, 768), key=mx.random.key(9))
    compare_metal(lambda: block(x, mx.zeros((1, 1, 1, 3))))


def test_silero_magnitude(compare_metal):
    model = get_registration("silero_vad").model_class()()
    pcm = mx.random.normal((2, 576), key=mx.random.key(23))
    compare_metal(lambda: model(pcm, trace=True)[2]["magnitude"],
                  option="REDUCTIONS", kernel="_magnitude_kernel")


def test_parakeet_split_glu(compare_metal):
    model = parakeet_model()
    model.eval()
    features = mx.random.normal((1, 25, 16), key=mx.random.key(17))
    compare_metal(lambda: model.encode(features, mx.array([24]))[0],
                  option="SWIGLU", kernel="_split_glu_kernel", exact=True)


@pytest.mark.parametrize("family", ["lfm2_colbert", "lfm2_vl", "lfm2_audio"])
def test_lfm_shared_swiglu(family, compare_metal):
    ids = mx.array(_IDS)
    if family == "lfm2_colbert":
        model = model_for(family, **{k: v for k, v in vars(dense_config()).items()
                                     if k != "model_type"}, projection_dim=8)
        def forward():
            return model(ids, attention_mask=mx.ones_like(ids)).embeddings
    elif family == "lfm2_vl":
        model = model_for(family, image_token_id=63, text_config=vars(dense_config()),
                          vision_config=dict(hidden_size=12, intermediate_size=24,
                                             num_hidden_layers=1, num_attention_heads=3,
                                             num_channels=3, patch_size=2, image_size=4,
                                             downsample_factor=2, projection_dim=16))
        pixels = mx.random.normal((1, 4, 12), key=mx.random.key(18))

        def forward():
            return model(
                mx.array([[1, 63, 2]]), pixel_values=pixels, spatial_shapes=[(2, 2)]
            ).logits
    else:
        model = NativeLFMAudio(native_audio_config())
        def forward():
            return model.lfm(ids)[0]
    model.eval()
    compare_metal(forward, option="SWIGLU", kernel="_swiglu_kernel", exact=True)


@pytest.mark.parametrize("family", ["sam3", "sam3_video", "sam3_tracker", "sam3_tracker_video"])
def test_sam3_components_gelu(family, compare_metal):
    vision = Sam3VisionConfig(
        backbone_config=dict(hidden_size=16, intermediate_size=24, num_hidden_layers=1,
                             num_attention_heads=4, image_size=16, patch_size=4,
                             window_size=4, pretrain_image_size=16, global_attn_indexes=[0]),
    )
    if family in {"sam3", "sam3_video"}:
        detector = get_registration("sam3").config_class()(
            vision_config=vision, text_config=dict(vocab_size=32, hidden_size=16,
                intermediate_size=24, num_hidden_layers=1, num_attention_heads=4,
                projection_dim=16, max_position_embeddings=8),
        )
        entry = get_registration(family)
        config = detector if family == "sam3" else entry.config_class()(detector_config=detector)
        model = entry.model_class()(config).eval()
        model = model.detector_model if family == "sam3_video" else model
        block = model.vision_encoder.backbone.layers[0].mlp
        shape = (1, 4, 16)
    elif family == "sam3_tracker":
        entry = get_registration(family)
        model = entry.model_class()(entry.config_class()(vision_config=vision)).eval()
        decoder = model.mask_decoder
        def block(x):
            return decoder.activation(decoder.upscale_layer_norm(decoder.upscale_conv1(x)))
        shape = (1, decoder.hidden_size, 2, 2)
    else:
        entry = get_registration(family)
        model = entry.model_class()(entry.config_class()(), remove_vision_encoder=True).eval()
        block = model.memory_encoder.mask_downsampler.layers[0]
        shape = (1, 1, 8, 8)
    x = Tensor(mx.random.normal(shape, key=mx.random.key(12)))
    compare_metal(lambda: raw(block(x)))
