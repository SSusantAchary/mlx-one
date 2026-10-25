"""Full native tiny-image pipeline; never a checkpoint performance measurement."""

import base64
import json
import os
import threading
from io import BytesIO

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip(
        "requires Metal; all checkpoints here are tiny synthetic fixtures", allow_module_level=True
    )

import mlx.core as mx
import numpy as np
from click.testing import CliRunner
from fastapi.testclient import TestClient
from PIL import Image
from vlm_fixtures import create_checkpoint

from mlx_one.text.schemas import TextGenerationOptions
from mlx_one.vision import VisionLanguageEngine, load_vlm_model

CASES = [
    ("qwen2_vl", 4, False),
    ("qwen2_5_vl", 4, False),
    ("qwen3_5", 4, False),
    ("lfm2_vl", 8, False),
    ("lfm2_vl", 4, True),
]


@pytest.mark.parametrize("family,bits,variant", CASES)
def test_quantized_load_processor_real_forward_managed_generation_and_cleanup(
    tmp_path, family, bits, variant
):
    root = create_checkpoint(tmp_path, family, bits, variant)
    bundle = load_vlm_model(root, offline=True)
    assert bundle.eos_token_ids == (2, 59)
    images = [Image.new("RGB", (8, 4), "red"), Image.new("RGBA", (4, 8), "blue")]
    prepared = bundle.processor.prepare("w3<image>w4<image>w5", images)
    assert len(prepared.image_metadata) == 2
    assert sum(row["visual_tokens"] for row in prepared.image_metadata) == int(
        mx.sum(prepared.input_ids == 60).item()
    )
    from mlx_one.vision.generation import _media_kwargs

    output = bundle.model(prepared.input_ids, **_media_kwargs(prepared))
    mx.eval(output.logits)
    assert mx.all(mx.isfinite(output.logits)).item()
    assert output.logits.shape[:2] == prepared.input_ids.shape
    options = TextGenerationOptions(max_tokens=3)
    with VisionLanguageEngine(bundle) as engine:
        for _ in range(3):
            result = engine.generate("w3<image>w4", [images[0]], options=options)
            assert result.prompt_tokens > 0
            assert result.generation_tokens == len(result.token_ids)
            assert result.finish_reason in {"stop", "length"}
            assert result.image_metadata[0]["source_size"] == [8, 4]
        stream = engine.stream("w3<image>w4", [images[0]], options=options)
        next(stream)
        stream.close()
        cancel = threading.Event()
        cancel.set()
        assert list(engine.stream("w3<image>w4", [images[0]], options=options, cancel=cancel)) == []
        result = engine.generate(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[0]},
                        {"type": "text", "text": "w3"},
                    ],
                },
                {"role": "assistant", "content": "w4"},
                {"role": "user", "content": "w5"},
            ],
            options=options,
        )
        assert result.image_metadata
    assert all(im.getpixel((0, 0)) for im in images)  # Caller images remain usable.


@pytest.mark.parametrize("family,bits,variant", CASES)
def test_native_cli_and_authenticated_http_image_chat_sse_alias(tmp_path, family, bits, variant):
    from mlx_one.cli import main
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.model_manager import ModelManager

    root = create_checkpoint(tmp_path / "model", family, bits, variant)
    image = Image.new("RGB", (4, 4), "red")
    path = tmp_path / "image.png"
    image.save(path)
    result = CliRunner().invoke(
        main,
        [
            "vlm",
            "--model",
            str(root),
            "--offline",
            "--image",
            str(path),
            "--prompt",
            "w3",
            "--max-tokens",
            "2",
            "--json-output",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["prompt_tokens"] > 0
    manager = ModelManager(alias="tiny-vlm")
    manager.load(root, offline=True)
    assert manager.model_info().modalities == ("text", "image")
    app = create_app(
        manager, config=ServerConfig(api_keys=("secret",), warmup=False, cache_prompt=False)
    )
    data = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    payload = {
        "model": "tiny-vlm",
        "max_tokens": 2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w3"},
                    {"type": "image_url", "image_url": {"url": data}},
                    {"type": "text", "text": "w4"},
                ],
            }
        ],
    }
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "tiny-vlm"
        response = client.post("/v1/chat/completions", json=payload, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["prompt_tokens"] > 0
        response = client.post(
            "/v1/chat/completions", json={**payload, "stream": True}, headers=headers
        )
        assert response.status_code == 200, response.text
        assert "data: [DONE]" in response.text
        assert (
            client.post(
                "/v1/chat/completions", json={**payload, "model": "wrong"}, headers=headers
            ).status_code
            == 404
        )
        bad = json.loads(json.dumps(payload))
        bad["messages"][0]["content"][1]["image_url"]["url"] = "https://127.0.0.1/internal"
        response = client.post("/v1/chat/completions", json=bad, headers=headers)
        assert response.status_code == 400 and "disabled" in response.text
        plain = {**payload, "messages": [{"role": "user", "content": "w3"}]}
        assert client.post("/v1/chat/completions", json=plain, headers=headers).status_code == 200


def test_qwen_patch_order_and_normalization_are_independently_computed(tmp_path):
    root = create_checkpoint(tmp_path)
    bundle = load_vlm_model(root, offline=True)
    source = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    prepared = bundle.processor.prepare("<image>", [Image.fromarray(source)])
    expected = (
        ((source.astype(np.float32) / 255 - 0.5) / 0.5)
        .reshape(2, 2, 2, 2, 3)
        .transpose(0, 2, 4, 1, 3)
        .reshape(4, 12)
    )
    np.testing.assert_allclose(np.array(prepared.pixel_values), expected, atol=2e-7)


def test_siglip2_dynamic_position_interpolation_and_patch_packing(tmp_path):
    from mlx_one.models.vision_language.lfm2_vl.model import naflex_position_embeddings

    table = mx.array([[0.0], [2.0], [4.0], [6.0]])
    actual = naflex_position_embeddings(table, 4, 4)
    # Half-pixel, align_corners=False bilinear interpolation, independently evaluated.
    expected = np.array([[0, 0.5, 1.5, 2], [1, 1.5, 2.5, 3], [3, 3.5, 4.5, 5], [4, 4.5, 5.5, 6]])
    np.testing.assert_allclose(np.array(actual).reshape(4, 4), expected, atol=1e-6)
    root = create_checkpoint(tmp_path, "lfm2_vl", 8)
    bundle = load_vlm_model(root, offline=True)
    source = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    prepared = bundle.processor.prepare("<image>", [Image.fromarray(source)])
    expected = (
        ((source.astype(np.float32) / 255 - 0.5) / 0.5)
        .reshape(2, 2, 2, 2, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(4, 12)
    )
    np.testing.assert_allclose(np.array(prepared.pixel_values[0]), expected, atol=2e-7)


@pytest.mark.parametrize("family,bits,variant", CASES)
def test_positional_serve_command_warmup_and_dispatch(tmp_path, monkeypatch, family, bits, variant):
    import uvicorn

    from mlx_one.cli import main

    root = create_checkpoint(tmp_path / "model", family, bits, variant)
    raw = BytesIO()
    Image.new("RGB", (4, 4), "green").save(raw, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()
    called = []

    def run(app, host, port):
        assert (host, port) == ("127.0.0.1", 8086)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer local-llama"},
                json={
                    "model": "qwen2-vl-2b",
                    "max_tokens": 2,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data}},
                                {"type": "text", "text": "w3"},
                            ],
                        }
                    ],
                },
            )
            assert response.status_code == 200, response.text
            assert (
                client.get(
                    "/v1/runtime", headers={"Authorization": "Bearer local-llama"}
                ).status_code
                == 200
            )
        called.append(True)

    monkeypatch.setattr(uvicorn, "run", run)
    result = CliRunner().invoke(
        main,
        [
            "serve",
            str(root),
            "--offline",
            "--host",
            "127.0.0.1",
            "--port",
            "8086",
            "--alias",
            "qwen2-vl-2b",
            "--api-key",
            "local-llama",
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    assert called == [True]


def test_eos_error_cancellation_release_and_bounded_repeated_memory(tmp_path, monkeypatch):
    import mlx_one.text.generation as generation
    from mlx_one.vision.generation import _stream_vlm

    bundle = load_vlm_model(create_checkpoint(tmp_path), offline=True)
    image = Image.new("RGB", (4, 4), "red")
    monkeypatch.setattr(generation, "select_token", lambda *args, **kwargs: 2)
    events = []
    chunks = list(
        _stream_vlm(
            bundle,
            "<image>w3",
            [image],
            TextGenerationOptions(max_tokens=4),
            lambda: False,
            observer=events.append,
        )
    )
    assert chunks[-1].terminal_token_id == 2 and chunks[-1].finish_reason == "stop"
    assert chunks[-1].generated_tokens == 0
    assert events[-1] == {"kind": "cleanup", "closed": True, "cache_bytes": 0}
    monkeypatch.setattr(generation, "select_token", lambda *args, **kwargs: 3)
    baseline = None
    for index in range(12):
        iterator = _stream_vlm(
            bundle, "<image>w3", [image], TextGenerationOptions(max_tokens=3), lambda: False
        )
        next(iterator)
        iterator.close()
        mx.synchronize()
        mx.clear_cache()
        if index == 2:
            baseline = mx.get_active_memory()
    assert mx.get_active_memory() - baseline < 2 * 1024**2
    events = []

    def fail(*args, **kwargs):
        raise RuntimeError("sample failure")

    monkeypatch.setattr(generation, "select_token", fail)
    with pytest.raises(RuntimeError, match="sample failure"):
        list(
            _stream_vlm(
                bundle,
                "<image>w3",
                [image],
                TextGenerationOptions(max_tokens=3),
                lambda: False,
                observer=events.append,
            )
        )
    assert events[-1]["kind"] == "cleanup" and events[-1]["cache_bytes"] == 0


@pytest.mark.parametrize("family,bits,variant", CASES)
def test_bfloat_vision_dtype_and_request_wrapper_collects_without_gc(
    tmp_path, monkeypatch, family, bits, variant
):
    import gc
    import weakref

    import mlx_one.vision.generation as vision_generation

    bundle = load_vlm_model(create_checkpoint(tmp_path, family, bits, variant), offline=True)
    tower = bundle.model.vision_model if family == "lfm2_vl" else bundle.model.visual
    tower.set_dtype(mx.bfloat16)
    image = Image.new("L", (4, 4), 70)
    prepared = bundle.processor.prepare("<image>w3", [image])
    output = bundle.model(prepared.input_ids, **vision_generation._media_kwargs(prepared))
    mx.eval(output.logits)
    assert output.vision_hidden_states.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(output.logits)).item()
    original = vision_generation.VisionForward
    refs = []

    class WatchedForward(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))

    monkeypatch.setattr(vision_generation, "VisionForward", WatchedForward)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(4):
            iterator = vision_generation._stream_vlm(
                bundle, "<image>w3", [image], TextGenerationOptions(max_tokens=2), lambda: False
            )
            list(iterator)
            iterator.close()
        assert all(ref() is None for ref in refs), "request cycle retains image arrays"
    finally:
        if enabled:
            gc.enable()


def test_direct_benchmark_uses_observed_counts_and_cleanup(tmp_path, monkeypatch):
    import mlx_one.text.generation as generation
    from benchmarks.vlm.run import measure
    from benchmarks.vlm.workloads import workload

    bundle = load_vlm_model(create_checkpoint(tmp_path), offline=True)
    monkeypatch.setattr(generation, "select_token", lambda *args, **kwargs: 3)
    sample = measure(bundle, workload("VLM-S1"))
    assert sample["output_tokens"] == len(sample["token_ids"]) == 128
    assert sample["cleanup"] is True
    assert sample["decode_tps"] == 127 * 1000 / sample["decode_ms"]
    monkeypatch.setattr(generation, "select_token", lambda *args, **kwargs: 2)
    sample = measure(bundle, workload("VLM-S1"))
    assert sample["output_tokens"] == 0
    assert sample["terminal_token_id"] == 2
    assert sample["decode_tps"] is None and sample["ttft_ms"] is None


@pytest.mark.parametrize(
    "checkpoint,expected_size,expected_views,visual_tokens",
    [
        ("mlx-community--Qwen2-VL-2B-Instruct-4bit", [1036, 1036], 1, 1369),
        ("mlx-community--Qwen2.5-VL-3B-Instruct-4bit", [1036, 1036], 1, 1369),
        ("mlx-community--Qwen3.5-0.8B-4bit", [1024, 1024], 1, 1024),
        ("mlx-community--LFM2-VL-450M-8bit", [512, 512], 4, 1024),
        ("LiquidAI--LFM2.5-VL-3B-MLX-4bit", [512, 512], 1, 256),
    ],
)
def test_exact_publisher_processor_dynamic_resolution_and_liquid_tile_order(
    checkpoint, expected_size, expected_views, visual_tokens
):
    from pathlib import Path

    from mlx_one.core.registry import get_registration
    from mlx_one.vision.loading import make_processor, processor_metadata

    root = Path(__file__).parent / "fixtures/vlm_checkpoints" / checkpoint
    payload = json.loads((root / "config.json").read_text())
    family = payload["model_type"].replace("-", "_")
    config = get_registration(family).config_class().from_dict(payload)
    # Tokenization is a stand-in here; real vision math and quantized loading are tested above.
    from types import SimpleNamespace

    tokenizer = SimpleNamespace(encode=lambda text: [1] if text else [])
    processor = make_processor(tokenizer, config, family, processor_metadata(root))
    image = Image.new("RGB", (1024, 1024))
    image.paste((255, 0, 0), (0, 0, 512, 512))
    image.paste((0, 255, 0), (512, 0, 1024, 512))
    image.paste((0, 0, 255), (0, 512, 512, 1024))
    prepared = processor.prepare("<image>w3", [image])
    mx.eval(prepared.pixel_values)
    metadata = prepared.image_metadata[0]
    assert metadata["processed_sizes"][0] == expected_size
    assert len(metadata["processed_sizes"]) == expected_views
    assert metadata["visual_tokens"] == visual_tokens
    assert int(mx.sum(prepared.input_ids == config.image_token_id).item()) == visual_tokens
    if expected_views == 4:
        np.testing.assert_allclose(
            np.array(prepared.pixel_values[:3, 0, :3]),
            [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
            atol=1e-6,
        )
        assert metadata["tile_grid"] == [2, 2]
