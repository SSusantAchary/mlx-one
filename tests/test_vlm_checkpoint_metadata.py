"""Published checkpoint header contracts, without tensor payloads or model downloads."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_one.core.registry import get_registration
from mlx_one.text.chat import ChatTemplate
from mlx_one.vision.loading import make_processor, processor_metadata
from mlx_one.vision.messages import prepare_messages

ROOT = Path(__file__).parent / "fixtures" / "vlm_checkpoints"
CHECKPOINTS = sorted(p for p in ROOT.iterdir() if p.is_dir())


class Shape:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def reshape(self, *shape):
        return Shape(
            [s if s != -1 else __import__("math").prod(self.shape) // shape[0] for s in shape]
        )

    def transpose(self, *axes):
        return Shape([self.shape[i] for i in axes])

    def squeeze(self, axis):
        return Shape([n for i, n in enumerate(self.shape) if i != axis % self.ndim])

    def __sub__(self, _):
        return self


@pytest.mark.parametrize("root", CHECKPOINTS, ids=lambda p: p.name)
def test_published_packed_tensor_names_and_logical_shapes(root):
    payload = json.loads((root / "config.json").read_text())
    family = payload["model_type"].replace("-", "_")
    registration = get_registration(family)
    config = registration.config_class().from_dict(payload)
    tensors = registration.sanitizer()(
        {
            k: Shape(v["shape"])
            for k, v in json.loads((root / "tensor-metadata.json").read_text()).items()
        },
        config,
    )
    contract = registration.weight_contract()(config)
    required = set(contract.expected) - set(contract.optional)
    actual = {
        name: value for name, value in tensors.items() if not name.endswith((".scales", ".biases"))
    }
    assert required <= actual.keys()
    assert not actual.keys() - contract.expected.keys()
    for name, tensor in actual.items():
        scale = tensors.get(name.removesuffix(".weight") + ".scales")
        logical = (*tensor.shape[:-1], scale.shape[-1] * 64) if scale else tensor.shape
        assert logical == contract.expected[name], name
    assert sum(n.endswith(".scales") for n in tensors) > 90


@pytest.mark.parametrize("root", CHECKPOINTS, ids=lambda p: p.name)
def test_pinned_assets_and_processor_selection(root):
    source = json.loads((root / "source.json").read_text())
    assert len(source["revision"]) == 40
    assert source["weights_downloaded"] is False
    for name, digest in source["files"].items():
        path = root / name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    payload = json.loads((root / "config.json").read_text())
    family = payload["model_type"].replace("-", "_")
    config = get_registration(family).config_class().from_dict(payload)
    metadata = processor_metadata(root)
    processor = make_processor(None, config, family, metadata)
    expected = {
        "qwen2_vl": "QwenImageProcessor",
        "qwen2_5_vl": "Qwen25ImageProcessor",
        "qwen3_5": "Qwen35ImageProcessor",
        "lfm2_vl": "LFMImageProcessor",
    }
    assert type(processor).__name__ == expected[family]
    if "LFM2.5" in root.name:
        assert processor.splitting is False
        assert config.vision_config.use_projection_layernorm is False


@pytest.mark.parametrize("root", CHECKPOINTS, ids=lambda p: p.name)
def test_exact_chat_templates_preserve_interleaved_multiturn_images(root):
    metadata = json.loads((root / "tokenizer-metadata.json").read_text())
    tokenizer = SimpleNamespace(
        bos_token=metadata["bos_token"], eos_token=metadata["eos_token"], special_tokens={}
    )
    template = metadata.get("chat_template") or (root / "chat_template.jinja").read_text()
    bundle = SimpleNamespace(chat_template=ChatTemplate("vision", template, tokenizer))
    prompt, images = prepare_messages(
        bundle,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image", "image": b"a"},
                    {"type": "text", "text": "after"},
                ],
            },
            {"role": "assistant", "content": "first answer"},
            {
                "role": "user",
                "content": [{"type": "image", "image": b"b"}, {"type": "text", "text": "compare"}],
            },
        ],
    )
    assert images == (b"a", b"b")
    assert prompt.count("<image>") == 2
    assert (
        prompt.index("before")
        < prompt.index("<image>")
        < prompt.index("after")
        < prompt.index("first answer")
        < prompt.index("compare")
    )


def test_local_decoder_rejects_unsafe_inputs_and_handles_alpha_exif():
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    from mlx_one.vision.images import decode_image

    with pytest.raises(UnidentifiedImageError):
        decode_image(b"not an image")
    image = Image.new("RGBA", (4, 3), (20, 40, 60, 0))
    result = decode_image(image)
    assert result.mode == "RGB" and result.getpixel((0, 0)) == (255, 255, 255)
    result.close()
    image = Image.new("RGB", (4, 3))
    exif = image.getexif()
    exif[274] = 6
    data = BytesIO()
    image.save(data, format="JPEG", exif=exif)
    result = decode_image(data.getvalue())
    assert result.size == (3, 4)
    result.close()
    with pytest.raises(ValueError, match="PNG, JPEG"):
        data = BytesIO()
        image.save(data, format="GIF")
        decode_image(data.getvalue())


def test_publisher_liquid_model_type_alias():
    assert get_registration("lfm2-vl") is get_registration("lfm2_vl")


def test_remote_fetch_cannot_be_enabled_without_a_safe_transport():
    from mlx_one.server.media import ImageResolver

    with pytest.raises(ValueError, match="DNS-pinned"):
        ImageResolver(allow_remote=True)


def test_model_resolution_fetches_processor_assets_without_changing_metadata_call(
    monkeypatch, tmp_path
):
    import mlx_one.text.loading as loading

    (tmp_path / "config.json").write_text('{"model_type":"qwen2_vl"}')
    calls = []

    def snapshot(model, **options):
        calls.append(options)
        return str(tmp_path)

    monkeypatch.setattr(loading, "snapshot_download", snapshot)
    assert (
        loading._resolve("test/not-downloaded", revision="pinned", offline=True, cache_dir=None)
        == tmp_path
    )
    assert {"preprocessor_config.json", "processor_config.json"} <= set(calls[0]["allow_patterns"])
    assert calls[0]["local_files_only"] is True
