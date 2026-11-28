"""Targeted backend-free image contracts and opt-in native Metal pipelines."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import sys
import threading
from concurrent.futures import CancelledError, Future
from dataclasses import replace

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from mlx_one import ImageEngine, ImageResult
from mlx_one.cli import main
from mlx_one.engine.execution_context import QueueFullError
from mlx_one.image import IMAGE_PROFILES, ImageMemoryError
from mlx_one.image.checkpoint import header_digest, tensor_header
from mlx_one.image.profiles import artifact_lock, image_profile
from mlx_one.image.request import ImageRequest
from mlx_one.image.schedule import flow_schedule
from mlx_one.server import ServerConfig, create_app


def test_catalog_and_request_limits():
    for item in IMAGE_PROFILES:
        assert item.task.value == "image-generation"
        assert len(item.revision) == 40
        assert artifact_lock(item)["weights"].keys() == {"text_encoder", "transformer", "vae"}
        request = ImageRequest.create(item, "grass", 768, 512, None, 5)
        assert (request.steps, request.seed) == (item.default_steps, 5)
        for args in [
            (" ", 512, 512, None, 1),
            ("a" * 8193, 512, 512, None, 1),
            ("grass", 65, 512, None, 1),
            ("grass", 1536, 1024, None, 1),
            ("grass", 512, 512, True, 1),
            ("grass", 512, 512, None, -1),
        ]:
            with pytest.raises(ValueError):
                ImageRequest.create(item, *args)
    with pytest.raises(ValueError, match="unknown"):
        image_profile("other")
    result = CliRunner().invoke(main, ["image", "list"])
    assert result.exit_code == 0
    assert [item["model"] for item in json.loads(result.output)] == [
        item.model for item in IMAGE_PROFILES
    ]


def test_schedule_nodes():
    flux = flow_schedule("flux2-klein", 4, 1024)
    np.testing.assert_allclose(flux, [1, 0.9578087, 0.8832782, 0.7161136, 0], rtol=2e-3)
    assert not np.array_equal(flux, flow_schedule("flux2-klein", 4, 1536))
    zimage = flow_schedule("z-image-turbo", 9, 1024)
    np.testing.assert_allclose(
        zimage, [1, 21 / 22, 0.9, 5 / 6, 0.75, 9 / 14, 0.5, 0.3, 0], rtol=1e-7
    )
    assert flux.dtype == zimage.dtype == np.float32
    assert len(zimage) - 1 == 8


def test_safetensors_header_rejects_duplicates_and_truncation(tmp_path):
    path = tmp_path / "weights.safetensors"
    spec = '{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
    payload = ('{"weight":' + spec + "}").encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(4))
    assert tensor_header(path) == {"weight": ["F32", [1]]}
    assert header_digest(tensor_header(path)) != header_digest({"weight": ["BF16", [1]]})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        tensor_header(path)
    payload = ('{"weight":' + spec + ',"weight":' + spec + "}").encode()
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + bytes(4))
    with pytest.raises(ValueError, match="duplicate"):
        tensor_header(path)


class _StubEngine:
    profile = replace(IMAGE_PROFILES[0], qualified_precisions=(4,))
    quantization = 4

    def submit(self, prompt, **kwargs):
        self.call = (prompt, kwargs)
        future = Future()
        if prompt == "busy":
            raise QueueFullError()
        if prompt == "memory":
            raise ImageMemoryError("insufficient unified memory")
        if prompt != "wait":
            future.set_result(ImageResult(b"image-content", {}))
        else:
            self.future = future
        return future

    def close(self):
        pass


def test_openai_http_contract():
    stub = _StubEngine()
    model = stub.profile.model
    app = create_app(image_engines={model: stub}, config=ServerConfig(timeout=0.03))
    with TestClient(app) as client:
        assert client.get("/v1/models").json()["data"][0]["mlx"]["tasks"] == ["image-generation"]
        endpoint = "/v1/images/generations"
        body = {"model": model, "prompt": "flower", "size": "768x512", "n": 1, "seed": 42}
        response = client.post(endpoint, json=body)
        assert response.status_code == 200
        assert response.json()["created"] > 0
        assert base64.b64decode(response.json()["data"][0]["b64_json"]) == b"image-content"
        assert stub.call[1]["width"] == 768 and stub.call[1]["height"] == 512
        assert stub.call[1]["seed"] == 42
        for option in ({"n": 2}, {"quality": "hd"}, {"response_format": "url"}, {"size": "auto"}):
            assert client.post(endpoint, json=body | option).status_code == 422
        assert client.post(endpoint, json=body | {"model": "unknown"}).status_code == 404
        assert client.post(endpoint, json=body | {"prompt": "busy"}).status_code == 429
        response = client.post(endpoint, json=body | {"prompt": "memory"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "insufficient_memory"
        assert client.post(endpoint, json=body | {"prompt": "wait"}).status_code == 408
        assert stub.call[1]["cancel"].is_set() and stub.future.cancelled()
        assert client.post("/v1/images/edits").status_code == 400
        assert "/v1/images/edits" not in client.get("/openapi.json").json()["paths"]
        assert client.post(endpoint, content=b" " * (1024**2 + 1)).status_code == 413


def test_bounded_executor_and_cancellation(monkeypatch):
    started, release = threading.Event(), threading.Event()
    model = IMAGE_PROFILES[0].model

    def fake_generate(self, request, cancel):
        started.set()
        assert release.wait(3)
        if cancel.is_set():
            raise CancelledError()
        return ImageResult(b"ok", {"seed": request.seed})

    monkeypatch.setattr(ImageEngine, "_generate", fake_generate)
    with ImageEngine(model, queue_size=1) as engine:
        cancel = threading.Event()
        running = engine.submit("active", cancel=cancel)
        assert started.wait(3)
        queued = engine.submit("queued", seed=9)
        try:
            with pytest.raises(QueueFullError):
                engine.submit("overflow")
        finally:
            cancel.set()
            release.set()
        with pytest.raises(CancelledError):
            running.result(timeout=3)
        assert queued.result(timeout=3).metadata["seed"] == 9


def test_lazy_import():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mlx_one import ImageEngine; "
            "from mlx_one.image import IMAGE_PROFILES; "
            "assert 'mlx.core' not in sys.modules and 'PIL' not in sys.modules; "
            "assert all(x not in sys.modules for x in ('mflux', 'torch', 'diffusers', 'mlx_lm'))",
        ],
        check=True,
    )


def _tiny_checkpoint(tmp_path, profile, monkeypatch, *, published=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from mlx_one.image import checkpoint as checkpoint_module
    from mlx_one.models.image_generation.conditioning import ImageConditioner
    from mlx_one.models.image_generation.flux2 import Flux2Denoiser
    from mlx_one.models.image_generation.vae import ImageDecoder
    from mlx_one.models.image_generation.zimage import ZImageDenoiser

    hidden = 64 if published else 16
    qwen = dict(
        model_type="qwen3",
        vocab_size=2,
        hidden_size=hidden,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    flux = profile.family == "flux2-klein"
    if flux:
        qwen["num_hidden_layers"] = 27
        transformer = dict(
            num_attention_heads=4,
            attention_head_dim=16,
            mlp_ratio=2,
            in_channels=16,
            eps=1e-6,
            axes_dims_rope=[4, 4, 4, 4],
            rope_theta=2000,
            timestep_guidance_channels=32,
            joint_attention_dim=3 * hidden,
            num_layers=1,
            num_single_layers=1,
            out_channels=None,
        )
    else:
        transformer = dict(
            all_patch_size=[2],
            all_f_patch_size=[1],
            in_channels=4,
            dim=64,
            n_heads=4,
            axes_dims=[4, 6, 6],
            rope_theta=256,
            t_scale=1000,
            norm_eps=1e-5,
            cap_feat_dim=hidden,
            n_refiner_layers=1,
            n_layers=1,
        )
    vae = dict(
        block_out_channels=[8, 8, 32, 64] if published else [4, 8, 8, 8],
        norm_num_groups=2,
        latent_channels=4,
        layers_per_block=1,
        out_channels=3,
        use_post_quant_conv=flux,
        batch_norm_eps=0.0001,
        scaling_factor=0.36,
        shift_factor=0.1159,
    )

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    Tokenizer(WordLevel({"<pad>": 0, "test": 1}, unk_token="<pad>")).save(
        str(tokenizer / "tokenizer.json")
    )
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "<pad>", "chat_template": "{{ messages[0].content }}"})
    )
    factories = {
        "text_encoder": (qwen, lambda cfg: ImageConditioner(cfg, tokenizer, profile.family, 8)),
        "transformer": (transformer, Flux2Denoiser if flux else ZImageDenoiser),
        "vae": (vae, ImageDecoder),
    }
    lock = {"files": {}, "weights": {}}
    if published:
        lock["configs"] = {name: config for name, (config, _) in factories.items()}
        lock["configs"]["scheduler"] = {"shift": 3}
    for component, (config, factory) in factories.items():
        root = tmp_path / component
        root.mkdir()
        (root / "config.json").write_text(json.dumps(config))
        weights = {}
        for key, weight in tree_flatten(factory(config).parameters()):
            weight = weight.astype(mx.bfloat16)
            if published:
                key = _published_fixture_name(key, component, flux)
                if key.endswith(".weight") and weight.ndim == 2 and weight.shape[-1] % 64 == 0:
                    packed = mx.quantize(weight, bits=4, group_size=64)
                    for suffix, value in zip(("weight", "scales", "biases"), packed, strict=True):
                        weights[key.removesuffix(".weight") + "." + suffix] = value
                    continue
            else:
                if component == "text_encoder":
                    key = f"model.{key}"
                if component == "vae" and weight.ndim == 4:
                    weight = weight.transpose(0, 3, 1, 2)
            weights[key] = weight
        path = root / "weights.safetensors"
        mx.save_safetensors(path, weights)
        layout = header_digest(tensor_header(path))
        identity = (
            {
                "layout": layout,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
            if published
            else layout
        )
        lock["weights"][component] = {f"{component}/{path.name}": identity}
    scheduler = tmp_path / "scheduler"
    scheduler.mkdir()
    (scheduler / "scheduler_config.json").write_text('{"shift": 3}')
    for path in tmp_path.rglob("*.json"):
        lock["files"][path.relative_to(tmp_path).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if published:
        monkeypatch.setattr(checkpoint_module, "published_q4_lock", lambda _: lock)
        monkeypatch.setattr("mlx_one.image.engine.published_q4_lock", lambda _: lock)
    else:
        monkeypatch.setattr(checkpoint_module, "artifact_lock", lambda _: lock)
    monkeypatch.setattr("mlx_one.image.engine.image_profile", lambda _: profile)


def _published_fixture_name(key, component, flux):
    if flux and component == "transformer":
        if key.startswith("transformer_blocks."):
            return key.replace(".attn.to_out.0.", ".attn.to_out.")
        return key.replace("time_guidance_embed.timestep_embedder.", "time_guidance_embed.")
    if flux and component == "vae":
        return key.replace(
            "decoder.mid_block.attentions.0.to_out.0.", "decoder.mid_block.attentions.0.to_out."
        )
    if component == "transformer":
        return (
            key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
            .replace("t_embedder.mlp.2.", "t_embedder.linear2.")
            .replace(
                "all_final_layer.2-1.adaLN_modulation.1.", "all_final_layer.2-1.adaLN_modulation.0."
            )
        )
    if component == "vae":
        return (
            key.replace("decoder.conv_in.", "decoder.conv_in.conv.")
            .replace("decoder.conv_norm_out.", "decoder.conv_norm_out.norm.")
            .replace("decoder.conv_out.", "decoder.conv_out.conv.")
        )
    return key


@pytest.mark.parametrize("profile", IMAGE_PROFILES, ids=lambda p: p.family)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.skipif(os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1", reason="requires Metal")
def test_tiny_native_generation_conversion_and_reload(tmp_path, monkeypatch, profile, bits):
    import mlx.core as mx
    from PIL import Image

    from mlx_one.image.checkpoint import convert_checkpoint

    profile = replace(profile, max_prompt_tokens=8)
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    _tiny_checkpoint(checkpoint, profile, monkeypatch)
    arguments = {"width": 64, "height": 64, "steps": 2}
    with ImageEngine(profile.model, checkpoint=checkpoint) as engine:
        dense = engine.generate("test", seed=42, **arguments)
        assert dense.metadata["denoiser_calls"] == (2 if profile.default_steps == 4 else 1)

    packed = convert_checkpoint(profile, checkpoint, tmp_path / "packed", bits=bits)
    with ImageEngine(profile.model, checkpoint=checkpoint, quantization=bits) as engine:
        dynamic = engine.generate("test", seed=42, **arguments)
    with ImageEngine(profile.model, checkpoint=packed, quantization=bits, idle_ttl=0) as engine:
        first = engine.generate("test", seed=42, **arguments)
        engine.unload()
        mx.random.seed(13)
        unrelated = mx.random.normal((12,))
        mx.eval(unrelated)
        second = engine.generate("test", seed=42, **arguments)
        other = engine.generate("test", seed=43, **arguments)
    assert first.png == second.png == dynamic.png
    assert first.png != other.png
    with Image.open(io.BytesIO(first.png)) as image:
        assert image.size == (64, 64) and image.mode == "RGB"


@pytest.mark.parametrize("profile", IMAGE_PROFILES, ids=lambda p: p.family)
@pytest.mark.skipif(
    os.environ.get("MLX_ONE_RUN_IMAGE_SDK_TESTS") != "1",
    reason="requires Metal and the OpenAI Python SDK",
)
def test_native_image_python_cli_and_sdk(tmp_path, monkeypatch, profile):
    from openai import OpenAI

    profile = replace(profile, max_prompt_tokens=8)
    checkpoint = tmp_path / "source"
    checkpoint.mkdir()
    _tiny_checkpoint(checkpoint, profile, monkeypatch)
    output = tmp_path / "cli.png"
    cli = CliRunner().invoke(
        main,
        [
            "image",
            "generate",
            profile.model,
            "--checkpoint",
            str(checkpoint),
            "--offline",
            "--quantization",
            "4",
            "--prompt",
            "test",
            "--width",
            "64",
            "--height",
            "64",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["denoiser_calls"] == (4 if profile.default_steps == 4 else 8)

    with ImageEngine(profile.model, quantization=4, checkpoint=checkpoint) as engine:
        expected = engine.generate("test", width=64, height=64, seed=42)
        app = create_app(image_engines={profile.model: engine})
        with TestClient(app) as transport:
            assert transport.get("/v1/models").json()["data"] == []
            with OpenAI(
                base_url="http://testserver/v1",
                api_key="local",
                http_client=transport,
                max_retries=0,
            ) as client:
                response = client.images.generate(
                    model=profile.model,
                    prompt="test",
                    size="64x64",
                    n=1,
                    response_format="b64_json",
                    extra_body={"seed": 42},
                )
    assert base64.b64decode(response.data[0].b64_json) == expected.png == output.read_bytes()


@pytest.mark.skipif(os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1", reason="requires Metal")
def test_native_denoising_cancellation_and_ttl(tmp_path, monkeypatch):
    from mlx_one.models.image_generation.flux2 import Flux2Denoiser

    profile = replace(IMAGE_PROFILES[0], max_prompt_tokens=8)
    _tiny_checkpoint(tmp_path, profile, monkeypatch)
    stop, expired = threading.Event(), threading.Event()
    forward = Flux2Denoiser.__call__
    unload = ImageEngine._release_weights

    def cancel_after_forward(self, *args):
        output = forward(self, *args)
        stop.set()
        return output

    def observe_unload(self):
        had_weights = bool(self._components)
        unload(self)
        if had_weights:
            expired.set()

    monkeypatch.setattr(Flux2Denoiser, "__call__", cancel_after_forward)
    monkeypatch.setattr(ImageEngine, "_release_weights", observe_unload)
    with ImageEngine(profile.model, quantization=4, checkpoint=tmp_path, idle_ttl=0) as engine:
        work = engine.submit("test", width=64, height=64, steps=2, seed=42, cancel=stop)
        with pytest.raises(CancelledError):
            work.result(timeout=10)
        assert not engine._components
        monkeypatch.setattr(Flux2Denoiser, "__call__", forward)
        expired.clear()
        result = engine.generate("test", width=64, height=64, steps=2, seed=42)
        assert result.metadata["denoiser_calls"] == 2
        assert expired.wait(3)
        assert not engine._components


@pytest.mark.parametrize("profile", IMAGE_PROFILES, ids=lambda p: p.family)
def test_published_q4_uses_exact_repo_revision(profile, monkeypatch, tmp_path):
    from mlx_one.image import checkpoint as checkpoint_module

    assert image_profile(profile.q4_model) == profile
    lock = {"files": {}, "weights": {"transformer": {"transformer/0.safetensors": {}}}}
    monkeypatch.setattr(checkpoint_module, "published_q4_lock", lambda _: lock)
    shard = tmp_path / "transformer" / "0.safetensors"
    shard.parent.mkdir()
    shard.touch()

    def snapshot_download(model, *, revision, allow_patterns, local_files_only):
        assert model == profile.q4_model
        assert revision == profile.q4_revision
        assert allow_patterns == ["transformer/0.safetensors"]
        assert local_files_only is True
        return tmp_path

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    bundle = checkpoint_module.resolve_checkpoint(profile, None, 4, offline=True, published=True)
    assert bundle.published and bundle.packed_bits == 4
    with pytest.raises(ValueError, match="already uses Q4"):
        ImageEngine(profile.q4_model, quantization=8)


@pytest.mark.parametrize("profile", IMAGE_PROFILES, ids=lambda p: p.family)
@pytest.mark.skipif(os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1", reason="requires Metal")
def test_published_q4_native_cli_http_and_integrity(tmp_path, monkeypatch, profile):
    import mlx.nn as nn

    profile = replace(profile, max_prompt_tokens=8)
    _tiny_checkpoint(tmp_path, profile, monkeypatch, published=True)
    out = tmp_path / "out.png"
    command = [
        "image",
        "generate",
        profile.q4_model,
        "--checkpoint",
        str(tmp_path),
        "--offline",
        "--prompt",
        "test",
        "--width",
        "64",
        "--height",
        "64",
        "--seed",
        "42",
        "--output",
        str(out),
    ]
    cli = CliRunner().invoke(main, command)
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["revision"] == profile.q4_revision
    with ImageEngine(profile.q4_model, checkpoint=tmp_path, offline=True) as engine:
        png = engine.generate("test", width=64, height=64, seed=42)
        assert png.metadata["model"] == profile.q4_model
        for name in ("text_encoder", "vae", "denoiser"):
            assert png.metadata["precision"][name]["bits"] == 4
        encoder = engine._components["text_encoder"]
        assert isinstance(encoder.embed_tokens, nn.QuantizedEmbedding)
        decoder = engine._components["vae"]
        assert isinstance(decoder.decoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
        assert out.read_bytes() == png.png
        with TestClient(create_app(image_engines={profile.q4_model: engine})) as client:
            body = {"model": profile.q4_model, "prompt": "test", "size": "64x64", "seed": 42}
            response = client.post("/v1/images/generations", json=body)
            assert response.status_code == 200, response.text
            assert base64.b64decode(response.json()["data"][0]["b64_json"]) == png.png

    shard = tmp_path / "transformer" / "weights.safetensors"
    contents = shard.read_bytes()
    shard.write_bytes(contents[:-1] + bytes([contents[-1] ^ 1]))
    with ImageEngine(profile.q4_model, checkpoint=tmp_path, offline=True) as engine:
        with pytest.raises(ValueError, match="published Q4 shard differs"):
            engine.generate("test", width=64, height=64)
