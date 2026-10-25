import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 for native Metal OCR tests", allow_module_level=True)

import json

import mlx.core as mx
import pytest
from conftest import tiny_config, write_tiny_checkpoint
from PIL import Image

from mlx_one.ocr import OCREngine, OCRInput
from mlx_one.ocr.generation import OCRForward
from mlx_one.ocr.loading import load_ocr_model, make_config, make_model
from mlx_one.ocr.processing import prepare_ocr

FAMILIES = ["trocr", "got_ocr2", "florence2", "paddleocr_vl", "lightonocr", "deepseek_ocr"]


def test_paddle_byte_fallback_preserves_newline_before_split_unicode(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, decoders, models

    from mlx_one.text import generation

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "</s>": 2,
                "A": 20,
                "<0x0A>": 21,
                "<0xE2>": 22,
                "<0x98>": 23,
                "<0x90>": 24,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    assert tokenizer.decode([20, 21]) == "A\n"
    assert not tokenizer.decode([20, 21, 22]).startswith("A\n")
    assert tokenizer.decode([20, 21, 22, 23, 24]) == "A\n☐"
    root = write_tiny_checkpoint(tmp_path / "paddle", "paddleocr_vl")
    bundle = load_ocr_model(root)
    monkeypatch.setattr(bundle.tokenizer, "decode", tokenizer.decode)
    choices = iter([20, 21, 22, 23, 24, bundle.eos_token_ids[0]])
    monkeypatch.setattr(generation, "select_token", lambda *a, **kw: next(choices))
    with OCREngine(bundle) as engine, Image.new("RGB", (16, 12), "white") as image:
        result = engine.recognize(image, max_tokens=8)
    assert result.text == "A\n☐"
    assert result.pages[0].metadata["output_tokens"] == 5


def test_lighton_structured_checkpoint_template_uses_native_ocr_message(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "lighton", "lightonocr")
    (root / "chat_template.jinja").write_text(
        "{% for message in messages %}<|im_start|>{{message['role']}}\n"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{% if add_vision_id %}Picture 1:{% endif %}<|image_pad|>"
        "{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    bundle = load_ocr_model(root)
    with Image.new("RGB", (16, 12), "white") as image:
        prepared = prepare_ocr(bundle, image, OCRInput(image, max_tokens=2))
        assert (
            prepared.ids.count(bundle.config.image_token_index)
            == prepared.metadata["visual_tokens"]
        )
        with OCREngine(bundle) as engine:
            result = engine.recognize(image, max_tokens=2)
        assert result.pages[0].metadata["output_tokens"] <= 2


@pytest.mark.parametrize("family", FAMILIES)
def test_native_full_checkpoint_path(tmp_path, family):
    root = write_tiny_checkpoint(tmp_path / family, family)
    bundle = load_ocr_model(root)
    with Image.new("RGB", (16, 12), "white") as image:
        prepared = prepare_ocr(bundle, image, OCRInput(image, max_tokens=2))
        forward = OCRForward(bundle, prepared)
        caches = forward.make_cache()
        output = forward(mx.array([prepared.ids]), cache=caches)
        mx.eval(output.logits)
        assert output.logits.shape == (1, 1, 64)
        assert bool(mx.all(mx.isfinite(output.logits)))
        second = forward(mx.array([[7]]), cache=caches)
        mx.eval(second.logits)
        assert caches[0].offset == len(prepared.ids) + 1
        forward.close()
        for cache in caches:
            cache.release()
        with OCREngine(bundle) as engine:
            a = engine.recognize(image, max_tokens=2)
            b = engine.recognize(image, max_tokens=2)
            assert a.text == b.text
            assert a.pages[0].metadata["visual_tokens"] > 0
            assert a.pages[0].metadata["output_tokens"] <= 2
            assert a.pages[0].elements == []
            json.dumps(a.to_dict())


@pytest.mark.parametrize("family", ["got_ocr2", "paddleocr_vl", "lightonocr", "deepseek_ocr"])
def test_affine_q4_native_loader(tmp_path, family):
    root = write_tiny_checkpoint(tmp_path / family, family, quantized=True)
    bundle = load_ocr_model(root)
    assert bundle.quantization["bits"] == 4
    with OCREngine(bundle) as engine, Image.new("RGB", (16, 12), "white") as image:
        result = engine.recognize(image, max_tokens=2)
        assert result.pages[0].metadata["visual_tokens"] > 0


def test_trocr_cross_attention_cache_and_image_use():
    cfg = make_config(tiny_config("trocr"), "trocr")
    model = make_model(cfg, "trocr")
    source = model.encode(mx.ones((1, 16, 16, 3)))
    changed = model.encode(mx.zeros((1, 16, 16, 3)))
    assert not bool(mx.allclose(source, changed))
    ids = mx.array([[2, 7, 8]])
    expected = model.decode(ids, source, None).logits
    caches = model.make_cache()
    pieces = [model.decode(ids[:, i : i + 1], source, caches).logits for i in range(3)]
    actual = mx.concatenate(pieces, axis=1)
    assert bool(mx.allclose(expected, actual, atol=1e-4))
    assert caches[0].cross_keys.shape[2] == source.shape[1]


def test_deepseek_crops_and_blank_image(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "deepseek_ocr", "deepseek_ocr")
    bundle = load_ocr_model(root)
    with Image.new("RGB", (128, 90), "black") as image:
        prepared = prepare_ocr(bundle, image, OCRInput(image, max_tokens=2))
        assert prepared.kwargs["images_spatial_crop"].prod().item() > 1
        assert prepared.kwargs["pixel_values"][0].shape[0] <= 9
        forward = OCRForward(bundle, prepared)
        caches = forward.make_cache()
        output = forward(mx.array([prepared.ids]), cache=caches)
        mx.eval(output.logits)
        assert bool(mx.all(mx.isfinite(output.logits)))
        forward.close()
        for cache in caches:
            cache.release()


def test_loading_missing_weights_fails(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "trocr", "trocr")
    weights = mx.load(str(root / "model.safetensors"))
    mx.eval(weights)
    weights.pop(next(iter(weights)))
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    with pytest.raises(ValueError):
        load_ocr_model(root)


def test_antialiased_position_resize_and_decoder_image_dependence():
    from mlx_one.models.ocr._layers import bicubic_interpolate

    constant = mx.ones((1, 3, 8, 8))
    for size in [(5, 5), (12, 12)]:
        result = bicubic_interpolate(constant, size, antialias=True)
        assert result.shape == (1, 3, *size)
        assert bool(mx.allclose(result, mx.ones_like(result), atol=1e-5))
    # Half-pixel symmetry and preservation of the average on a symmetric ramp.
    ramp = mx.arange(64, dtype=mx.float32).reshape(1, 1, 8, 8)
    result = bicubic_interpolate(ramp, (5, 5), antialias=True)
    assert abs(result.mean().item() - 31.5) < 1e-4


def test_direct_benchmark_page_execution(tmp_path):
    from benchmarks.ocr.run import measure

    root = write_tiny_checkpoint(tmp_path / "trocr", "trocr")
    bundle = load_ocr_model(root)
    with Image.new("RGB", (16, 12), "white") as image:
        result = measure(bundle, OCRInput(image, max_tokens=2))
    assert result.metadata["measurement_type"] == "direct-model"
    assert len(result.pages) == 1
    assert result.pages[0].metadata["output_tokens"] <= 2


def test_checkpoint_specific_resize_and_plain_prompt(tmp_path):
    lighton = load_ocr_model(write_tiny_checkpoint(tmp_path / "lightonocr", "lightonocr"))
    with Image.new("RGB", (37, 42), "white") as image:
        prepared = prepare_ocr(lighton, image, OCRInput(image, max_tokens=2))
        assert prepared.metadata["processed_size"] == (28, 32)
    deepseek = load_ocr_model(write_tiny_checkpoint(tmp_path / "deepseek_ocr", "deepseek_ocr"))
    with Image.new("RGB", (16, 12), "white") as image:
        prepared = prepare_ocr(deepseek, image, OCRInput(image, max_tokens=2))
        assert prepared.ids[0] == deepseek.tokenizer.bos_token_id
        assert prepared.kwargs["images_seq_mask"][0, :2].tolist() == [False, True]


def test_eos_stops_and_truncation_is_honest(tmp_path):
    bundle = load_ocr_model(write_tiny_checkpoint(tmp_path / "trocr", "trocr"))
    norm = bundle.model.decoder.model.decoder.layers[-1].final_layer_norm
    norm.weight = mx.zeros_like(norm.weight)
    norm.bias = mx.ones_like(norm.bias)
    output = bundle.model.decoder.output_projection
    output.weight = mx.zeros_like(output.weight)
    output.weight[2] = mx.ones_like(output.weight[2])
    mx.eval(bundle.model.parameters())
    with OCREngine(bundle) as engine, Image.new("RGB", (16, 12), "white") as image:
        result = engine.recognize(image, max_tokens=8)
        assert result.pages[0].metadata["finish_reason"] == "stop"
        assert not result.pages[0].metadata["truncated"]
        assert result.pages[0].metadata["output_tokens"] == 0


@pytest.mark.parametrize(
    "family,dtype",
    [
        ("florence2", mx.float16),
        ("got_ocr2", mx.bfloat16),
        ("paddleocr_vl", mx.bfloat16),
        ("lightonocr", mx.bfloat16),
        ("deepseek_ocr", mx.bfloat16),
    ],
)
def test_checkpoint_floating_dtypes(tmp_path, family, dtype):
    root = write_tiny_checkpoint(tmp_path / family, family, dtype=dtype)
    with OCREngine(root) as engine, Image.new("RGB", (16, 12), "white") as image:
        result = engine.recognize(image, max_tokens=2)
        assert result.pages[0].metadata["visual_tokens"] > 0
