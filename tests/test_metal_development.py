# Author: Susant Achary
"""Short architecture coverage checks for opt-in native Metal kernels."""

import os
from dataclasses import replace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from ocr.conftest import write_tiny_checkpoint
from PIL import Image
from test_moonshine_mlx import model as moonshine_model
from test_native_qwen3_retrieval_mlx import _embedding_config, _reranker_config
from test_native_qwen_mlx import vl_config
from test_native_qwen_multimodal_mlx import qwen2_5_vl_config
from test_qwen_audio_mlx import model as asr_model

from mlx_one.core.registry import get_registration
from mlx_one.models.embeddings.nomic_bert.config import NomicBertConfig
from mlx_one.models.language.inkling2 import controller_pack
from mlx_one.models.language.inkling2.long_attention import _supported_gpu
from mlx_one.models.language.inkling2.model import Controller
from mlx_one.models.language.inkling2.verify import same_bits
from mlx_one.models.shared import metal_activations
from mlx_one.ocr import OCRInput
from mlx_one.ocr.generation import OCRForward
from mlx_one.ocr.loading import load_ocr_model
from mlx_one.ocr.processing import prepare_ocr


@pytest.fixture
def compare_swiglu(monkeypatch):
    def compare(forward):
        monkeypatch.delenv("MLX_ONE_METAL_SWIGLU", raising=False)
        reference = forward()
        mx.eval(reference)

        launches = []
        kernel = metal_activations._swiglu_kernel()

        def launch(**kwargs):
            launches.append(kwargs["grid"])
            return kernel(**kwargs)

        monkeypatch.setattr(metal_activations, "_swiglu_kernel", lambda: launch)
        monkeypatch.setenv("MLX_ONE_METAL_SWIGLU", "1")
        actual = forward()
        assert same_bits(actual, reference).item()
        assert launches

    return compare


@pytest.mark.parametrize("rows", [5, 32, 128])
@pytest.mark.skipif(not _supported_gpu(), reason="Inkling controller packing is qualified on M4")
def test_inkling_controller_pack(rows, monkeypatch):
    width = 2304
    x = mx.random.normal((1, rows, width), key=mx.random.key(rows))
    message = mx.random.normal((1, rows, 4, width), key=mx.random.key(rows + 1))[:, :, 2, :]
    weights = {
        "controller.down_weight": mx.full((4, 2 * width), 0.01, dtype=mx.bfloat16),
        "controller.forget_weight": mx.full((width, 4), 0.1),
        "controller.gain_weight": mx.full((width, 4), 0.2),
    }
    controller = Controller(weights, "controller")
    reference = controller(x, message, 1e-6)
    controller.pack_proofs = {}
    assert same_bits(controller(x, message, 1e-6), reference).item()
    assert controller.pack_proofs[(x.shape, 1e-6)]

    launch = controller_pack._launch
    launched = []

    def observe(*args):
        launched.append(True)
        return launch(*args)

    monkeypatch.setattr(controller_pack, "_launch", observe)
    assert same_bits(controller(x, message, 1e-6), reference).item()
    assert launched
    launched.clear()
    assert same_bits(controller(x[:, :2], message[:, :2], 1e-6), reference[:, :2]).item()
    assert not launched


def test_phi3_swiglu(compare_swiglu):
    entry = get_registration("phi3")
    config = entry.config_class().from_dict(dict(
        model_type="phi3", hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
    ))
    model = entry.model_class()(config)
    compare_swiglu(lambda: model(mx.array([[1, 3, 5]])).logits)


@pytest.mark.parametrize("family,config", [
    ("qwen3_embedding", _embedding_config),
    ("qwen3_reranker", _reranker_config),
    ("nomic_bert", lambda: NomicBertConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=4, max_seq_length=16,
    )),
])
def test_retrieval_swiglu(family, config, compare_swiglu):
    model = get_registration(family).model_class()(config())
    ids = mx.array([[0, 1, 3]])
    mask = mx.array([[0, 1, 1]])
    if family == "qwen3_reranker":
        compare_swiglu(lambda: model(ids, attention_mask=mask,
                                     false_token_id=7, true_token_id=8).scores)
    else:
        compare_swiglu(lambda: model(ids, attention_mask=mask).embeddings)


@pytest.mark.parametrize("family", ["moonshine", "qwen3_asr", "qwen3_forced_aligner"])
def test_audio_swiglu(family, compare_swiglu):
    ids = mx.array([[1, 3, 5]])
    if family == "moonshine":
        model = moonshine_model()
        encoded = model.encode(mx.zeros((1, 4000)))
        compare_swiglu(lambda: model.decode(ids, encoded).logits)
        return

    model = asr_model()
    if family == "qwen3_forced_aligner":
        config = replace(model.config, model_type=family, classify_num=8,
                         text_config=replace(model.config.text_config, tie_word_embeddings=False))
        model = get_registration(family).model_class()(config)
    compare_swiglu(lambda: model(ids))


@pytest.mark.parametrize("family,config_factory", [
    ("qwen2_vl", vl_config), ("qwen2_5_vl", qwen2_5_vl_config),
])
def test_vision_language_swiglu(family, config_factory, compare_swiglu):
    config = config_factory()
    model = get_registration(family).model_class()(config)
    patch_width = (config.vision_config.in_channels * config.vision_config.temporal_patch_size
                   * config.vision_config.patch_size**2)
    patches = mx.random.normal((4, patch_width), key=mx.random.key(28))
    ids = mx.array([[5, config.image_token_id, 6]])
    compare_swiglu(lambda: model(ids, pixel_values=patches, image_grid_thw=((1, 2, 2),)).logits)


@pytest.mark.parametrize("family", ["got_ocr2", "paddleocr_vl", "lightonocr", "deepseek_ocr"])
def test_ocr_swiglu(family, tmp_path, compare_swiglu):
    root = write_tiny_checkpoint(tmp_path / family, family)
    bundle = load_ocr_model(root)
    with Image.new("RGB", (16, 12), "white") as image:
        prepared = prepare_ocr(bundle, image, OCRInput(image, max_tokens=2))

        def forward():
            runner = OCRForward(bundle, prepared)
            caches = runner.make_cache()
            try:
                logits = runner(mx.array([prepared.ids]), cache=caches).logits
                mx.eval(logits)
                return logits
            finally:
                runner.close()
                for cache in caches:
                    cache.release()

        compare_swiglu(forward)
