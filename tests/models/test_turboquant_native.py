"""Bounded packed arithmetic and native GPT integration; no checkpoint bodies."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_large_moe_contracts import tiny_config
from test_large_moe_native import write_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/turboquant"))
from oracle import fixture, reconstruct, rotate_power2  # noqa: E402


@pytest.fixture(scope="module")
def mx():
    try:
        import mlx.core as core
    except ImportError as exc:
        pytest.skip(str(exc))
    if not core.metal.is_available():
        pytest.skip("Metal required")
    return core


@pytest.mark.parametrize("inputs,outputs", [(64, 7), (128, 67), (4096, 5)])
@pytest.mark.parametrize("rows", [1, 2, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_packed_dense(mx, inputs, outputs, rows, dtype):
    from mlx_one.quantization.turboquant.dispatch import collect_dispatch
    from mlx_one.quantization.turboquant.linear import TQ3Linear

    data = fixture(outputs, inputs)
    layer = TQ3Linear(inputs, outputs, {k: mx.array(v) for k, v in data.items()})
    rng = np.random.default_rng(9)
    x = rng.normal(size=(rows, inputs)).astype(dtype)
    weight = reconstruct(data["weight"], data["scales"], data["codebook"], inputs)
    rotated = rotate_power2(x, data["signs"]).astype(dtype)
    # MLX Hadamard rounds to input dtype; compare fused arithmetic on that input.
    from mlx_one.quantization.turboquant.linear import rotate_input

    xr = rotate_input(mx.array(x), layer.signs)
    np.testing.assert_allclose(
        np.array(xr), rotated, atol=0.004 if dtype == "float16" else 2e-6, rtol=0.002
    )
    expected = np.array(xr).astype(np.float64) @ weight.T
    dense = mx.array(np.array(xr)) @ mx.array(weight.astype(np.float32)).T
    with collect_dispatch() as stats:
        actual = layer(mx.array(x))
        mx.eval(actual)
    assert stats["qmv" if rows == 1 else "qmv_rows" if rows == 2 else "qmm"] == 1
    atol = 0.002 if dtype == "float16" else 2e-5
    np.testing.assert_allclose(
        np.array(actual), expected, rtol=0.001 if dtype == "float16" else 2e-5, atol=atol
    )
    np.testing.assert_allclose(np.array(actual), np.array(dense), rtol=0.001, atol=atol)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 128, 256])
@pytest.mark.parametrize("shared", [True, False])
def test_sparse_selected_gather(mx, tokens, shared):
    from mlx_one.quantization.turboquant.experts import TQ3ExpertLinear
    from mlx_one.quantization.turboquant.linear import rotate_input

    data = fixture(67, 128, experts=8)
    layer = TQ3ExpertLinear(128, 67, {k: mx.array(v) for k, v in data.items()})
    rng = np.random.default_rng(12)
    indices = rng.integers(0, 8, (1, tokens, 4), dtype=np.uint32)
    shape = (1, tokens, 1, 1, 128) if shared else (1, tokens, 4, 1, 128)
    x = mx.array(rng.normal(size=shape).astype(np.float16))
    xrot = np.array(rotate_input(x, layer.signs)).reshape(-1, 128)
    xrows = np.repeat(xrot, 4, axis=0) if shared else xrot
    bank = reconstruct(data["weight"], data["scales"], data["codebook"], 128)
    expected = np.stack(
        [bank[e] @ row.astype(np.float64) for e, row in zip(indices.ravel(), xrows, strict=False)]
    )
    actual = layer(x, mx.array(indices))
    assert actual.shape == (1, tokens, 4, 1, 67)
    np.testing.assert_allclose(np.array(actual).reshape(-1, 67), expected, rtol=0.001, atol=0.0005)
    # Sorted and unsorted call forms must preserve the routing order.
    order = np.argsort(indices.ravel())
    # xrows already rotated; feed original rows to avoid an extra rotation.
    original = np.array(x).reshape(-1, 128)
    original = np.repeat(original, 4, axis=0) if shared else original
    sorted_y = layer(
        mx.array(original[order]).reshape(-1, 1, 128),
        mx.array(indices.ravel()[order]),
        sorted_indices=True,
    )
    np.testing.assert_allclose(
        np.array(sorted_y).reshape(-1, 67), expected[order], rtol=0.001, atol=0.0005
    )


def write_tq3_checkpoint(mx, root):
    from mlx.utils import tree_flatten

    from mlx_one.core.registry import get_registration
    from mlx_one.quantization.turboquant.config import TQ3Config
    from mlx_one.quantization.turboquant.loading import prepare_tq3_model

    # Reuse tokenizer/template setup, then replace every parameter with our own fixture.
    write_checkpoint(mx, root, "gpt_oss")
    config = tiny_config("gpt_oss", num_experts_per_tok=4)
    model = get_registration("gpt_oss").model_class()(config)
    model.set_dtype(mx.bfloat16)
    weights = dict(tree_flatten(model.parameters()))
    for path, module in list(model.named_modules()):
        if path == "lm_head" or ".self_attn." in path or ".experts." in path:
            if "weight" not in module or module.weight.ndim not in (2, 3):
                continue
            shape = module.weight.shape
            data = fixture(shape[-2], shape[-1], experts=shape[0] if len(shape) == 3 else None)
            for name, value in data.items():
                weights[path + "." + name] = mx.array(value)
            if path + ".bias" in weights:
                weights[path + ".bias"] = weights[path + ".bias"].astype(mx.float16)
    q = TQ3Config().to_dict()
    prepare_tq3_model(model, weights, q)
    model.load_weights(list(weights.items()), strict=True)
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(
        json.dumps({**asdict(config), "quantization": q, "eos_token_id": 2})
    )
    return model


def test_native_tiny_checkpoint_cache_and_owner(mx, tmp_path):
    from mlx_one import CacheConfig, Engine
    from mlx_one.text.loading import load_text_model

    write_tq3_checkpoint(mx, tmp_path)
    for _ in range(2):
        bundle = load_text_model(str(tmp_path), offline=True)
        # First execution on Engine's owning thread verifies lazy-load ownership.
        engine = Engine(bundle, cache_config=CacheConfig(prefix_cache_enabled=False))
        result = engine.generate("hello", max_tokens=2)
        assert result.generation_tokens <= 2
        engine.close()
        model = bundle.model
        cache = model.make_cache()
        prefill = model(mx.array([[1, 3, 4]]), cache=cache).logits
        continuation = model(mx.array([[5]]), cache=cache).logits
        full = model(mx.array([[1, 3, 4, 5]])).logits
        mx.eval(prefill, continuation, full)
        assert prefill.shape == (1, 3, 128) and mx.all(mx.isfinite(full)).item()
        np.testing.assert_allclose(
            np.array(continuation.astype(mx.float32)),
            np.array(full[:, -1:].astype(mx.float32)),
            atol=0.02,
            rtol=0.03,
        )
        for c in cache:
            c.reset()
        assert all(c.offset == 0 for c in cache)
        replay = model(mx.array([[1, 3, 4]]), cache=cache).logits
        np.testing.assert_array_equal(
            np.array(prefill.astype(mx.float32)), np.array(replay.astype(mx.float32))
        )
        del bundle, model, engine


@pytest.mark.parametrize(
    "corruption", ["missing", "shape", "dtype", "signs", "codebook", "scales", "extra"]
)
def test_reject_corrupt_storage(mx, tmp_path, corruption):
    from mlx_one.text.loading import TextModelLoadError, load_text_model

    write_tq3_checkpoint(mx, tmp_path)
    file = str(tmp_path / "model.safetensors")
    weights = mx.load(file)
    prefix = "model.layers.0.mlp.experts.gate_proj."
    if corruption == "missing":
        del weights[prefix + "signs"]
    elif corruption == "shape":
        weights[prefix + "weight"] = weights[prefix + "weight"][..., :-1]
    elif corruption == "dtype":
        weights[prefix + "weight"] = weights[prefix + "weight"].astype(mx.int32)
    elif corruption == "extra":
        weights[prefix + "garbage"] = mx.zeros((1,))
    elif corruption == "signs":
        weights[prefix + "signs"] = mx.zeros_like(weights[prefix + "signs"])
    elif corruption == "scales":
        weights[prefix + "scales"] = -mx.ones_like(weights[prefix + "scales"])
    else:
        weights[prefix + "codebook"] = mx.zeros((8,), mx.float16)
    mx.eval(weights)
    mx.save_safetensors(file, weights)
    with pytest.raises(TextModelLoadError, match="TQ3|quantized"):
        load_text_model(str(tmp_path), offline=True)


def test_tq3_python_engine_cli_http_sse_cancellation(mx, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one import CacheConfig, Engine
    from mlx_one.cli import main
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text import TextGenerationOptions, generate, stream_chat
    from mlx_one.text.loading import load_text_model

    family = "gpt_oss"
    write_tq3_checkpoint(mx, tmp_path)
    monkeypatch.setattr(
        "mlx_one.text.loading.snapshot_download", lambda *a, **k: pytest.fail("network")
    )
    bundle = load_text_model(tmp_path, offline=True)
    assert bundle.eos_token_ids == (2, 15, 16)
    assert bundle.quantization["mode"] == "turboquant"
    assert bundle.model.lm_head.format == "turboquant_tq3"

    # Sampling is deterministic; every forward/cache transition still executes
    # the real native model on Metal. Analysis must not reach Python/CLI/HTTP.
    script = [9, 6, 10, 5, 11, 8, 14, 9, 7, 10, 4, 15] if family == "gpt_oss" else [4, 15]

    def reset_sampler():
        sequence = iter(script)
        monkeypatch.setattr(
            "mlx_one.text.generation.select_token",
            lambda *a, sequence=sequence, **kw: next(sequence),
        )

    reset_sampler()
    prompt = "<|start|>assistant" if family == "gpt_oss" else "hello"
    result = generate(bundle, prompt, options=TextGenerationOptions(max_tokens=32))
    assert result.text.strip() == "Answer" and result.finish_reason == "stop"
    assert result.generation_tokens == len(script) - 1
    for eos in bundle.eos_token_ids:
        sequence = iter([*script[:-1], eos])
        monkeypatch.setattr(
            "mlx_one.text.generation.select_token",
            lambda *a, sequence=sequence, **kw: next(sequence),
        )
        result = generate(bundle, prompt, options=TextGenerationOptions(max_tokens=32))
        assert result.text.strip() == "Answer" and result.finish_reason == "stop"
        assert result.generation_tokens == len(script) - 1
    reset_sampler()
    chunks = list(
        stream_chat(
            bundle,
            [{"role": "user", "content": "hello"}],
            options=TextGenerationOptions(max_tokens=32),
        )
    )
    assert "".join(c.text for c in chunks).strip() == "Answer"
    reset_sampler()
    with Engine(
        bundle,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        assert engine.generate(prompt, max_tokens=32).text.strip() == "Answer"
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
        import threading

        reset_sampler()
        cancelled = threading.Event()
        stream = engine.stream(prompt, max_tokens=32, cancel=cancelled)
        next(stream)
        cancelled.set()
        list(stream)
        stream.close()
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
        reset_sampler()
        assert engine.generate(prompt, max_tokens=32).text.strip() == "Answer"
    reset_sampler()
    cli = CliRunner().invoke(
        main,
        ["generate", str(tmp_path), prompt, "--offline", "--max-tokens", "32", "--json-output"],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["text"].strip() == "Answer"
    manager = ModelManager()
    manager.load(tmp_path, offline=True)
    app = create_app(
        manager,
        config=ServerConfig(
            warmup=False,
            cache_prompt=False,
            api_keys=("test-key",),
            inference_mode="reference_dense",
        ),
    )
    try:
        with TestClient(app) as client:
            assert client.get("/v1/models").status_code == 401
            client.headers["Authorization"] = "Bearer test-key"
            model_id = client.get("/v1/models").json()["data"][0]["id"]
            for route in ("completions", "chat/completions"):
                payload = {"model": model_id, "max_tokens": 32, "temperature": 0}
                payload.update(
                    {"prompt": prompt}
                    if route == "completions"
                    else {"messages": [{"role": "user", "content": "hello"}]}
                )
                reset_sampler()
                response = client.post("/v1/" + route, json=payload)
                assert response.status_code == 200, response.text
                assert "Answer" in response.text and "scratchpad-secret" not in response.text
                reset_sampler()
                streamed = client.post("/v1/" + route, json={**payload, "stream": True})
                assert streamed.status_code == 200 and "[DONE]" in streamed.text
                assert "Answer" in streamed.text and "scratchpad-secret" not in streamed.text
            runtime = client.get("/v1/runtime").json()
            assert "test-key" not in json.dumps(runtime)
    finally:
        manager.unload()
    assert "mlx_lm" not in sys.modules and "transformers" not in sys.modules


@pytest.mark.parametrize("inputs", [64, 2880, 4096])
def test_reference_generated_rotation_fixture(mx, inputs):
    from mlx_one.quantization.turboquant.linear import TQ3Linear

    root = Path(__file__).resolve().parents[1] / "fixtures/turboquant_metadata"
    data = np.load(root / f"reference-generated-{inputs}.npz")
    layer = TQ3Linear(
        inputs,
        3,
        {name: mx.array(data[name]) for name in ["weight", "scales", "codebook", "signs"]},
    )
    actual = layer(mx.array(data["x"]))
    np.testing.assert_allclose(np.array(actual), data["expected"], atol=0.008, rtol=0.002)


def test_tiny_gpt_against_independent_dense_reconstruction(mx, tmp_path):
    from mlx_one.models.language.gpt_oss.model import GptOssForCausalLM
    from mlx_one.text.loading import load_text_model

    write_tq3_checkpoint(mx, tmp_path)
    bundle = load_text_model(tmp_path, offline=True)
    packed = mx.load(str(tmp_path / "model.safetensors"))
    dense = {
        key: value
        for key, value in packed.items()
        if not key.endswith((".scales", ".signs", ".codebook"))
    }
    for key, value in packed.items():
        if key.endswith(".codebook"):
            prefix = key[:-9]
            signs = np.array(packed[prefix + ".signs"])
            rotated = reconstruct(
                np.array(packed[prefix + ".weight"]),
                np.array(packed[prefix + ".scales"]),
                np.array(value),
                len(signs),
            )
            # Inverse rotation: normalized H then column signs.
            dense[prefix + ".weight"] = mx.array(
                (rotate_power2(rotated, np.ones_like(signs)) * signs).astype(np.float32)
            )
    reference = GptOssForCausalLM(bundle.model.config)
    reference.load_weights(list(dense.items()), strict=True)
    tokens = mx.array([[1, 3, 4, 5]])
    expected = reference(tokens).logits
    actual = bundle.model(tokens).logits
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.0001,
        rtol=0.001,
    )
    assert all(layer.mlp.top_k == 4 for layer in bundle.model.model.layers)
    caches = bundle.model.make_cache()
    bundle.model(tokens, cache=caches)
    assert all(
        (cache.state()[0] if hasattr(cache, "state") else cache._keys).dtype == mx.float16
        for cache in caches
    )


def test_mixed_tq3_and_explicit_affine_override(mx, tmp_path, monkeypatch):
    from mlx_one.text.loading import load_text_model

    write_tq3_checkpoint(mx, tmp_path)
    file = str(tmp_path / "model.safetensors")
    weights = mx.load(file)
    path = "model.layers.0.mlp.router"
    weights[path + ".weight"], weights[path + ".scales"], weights[path + ".biases"] = mx.quantize(
        weights[path + ".weight"], bits=8, group_size=64
    )
    mx.eval(weights)
    mx.save_safetensors(file, weights)
    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization"][path] = {"bits": 8, "group_size": 64, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    # The TQ3 loader must never send its metadata to the ordinary model quantizer.
    monkeypatch.setattr(
        "mlx.nn.quantize", lambda *a, **k: pytest.fail("ordinary model quantization")
    )
    bundle = load_text_model(tmp_path, offline=True)
    assert bundle.model.model.layers[0].mlp.router.bits == 8
    assert bundle.model.model.layers[0].mlp.experts.gate_proj.format == "turboquant_tq3"
    logits = bundle.model(mx.array([[1, 3]])).logits
    assert mx.all(mx.isfinite(logits)).item()
