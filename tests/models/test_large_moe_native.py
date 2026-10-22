"""Bounded, automatic Metal tests using fabricated safetensors, never full weights."""

import json
import sys
from dataclasses import asdict

import numpy as np
import pytest
from test_large_moe_contracts import METADATA, tiny_config


@pytest.fixture(scope="module")
def mx():
    try:
        import mlx.core as core
    except ImportError as exc:
        pytest.skip(f"native Metal runtime unavailable: {exc}")
    if not core.metal.is_available():
        pytest.skip("a Metal device is required for bounded native operations")
    return core


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_forward_real_continuation_reset_and_batch(mx, family, dtype):
    from mlx_one.core.registry import get_registration

    config = tiny_config(family)
    model = get_registration(family).model_class()(config)
    model.set_dtype(getattr(mx, dtype))
    cache = model.make_cache()
    first = model(mx.array([[1, 2]]), cache=cache)
    mx.eval(first.logits)
    assert first.logits.shape == (1, 2, config.vocab_size)
    assert first.logits.dtype == getattr(mx, dtype)
    assert mx.all(mx.isfinite(first.logits)).item()
    assert all(c.offset == 2 for c in cache)
    initial = np.array(cache[1].state()[0].astype(mx.float32))
    for token in (3, 4, 5, 6):
        logits = model(mx.array([[token]]), cache=cache).logits
        mx.eval(logits)
        assert mx.all(mx.isfinite(logits)).item()
    assert all(c.offset == 6 for c in cache)
    if family == "gpt_oss":
        assert cache[0].keys.shape[2] == 4
        assert cache[1].state()[0].shape[2] == 6
    else:
        assert cache[0].values.shape == (1, 2, config.hidden_size)
        assert cache[1].state()[0].shape[2] == 6
        assert not hasattr(model, "lm_head")
        assert model.model.layers[0].feed_forward.w1.weight.shape == (64, 64)
    np.testing.assert_array_equal(
        np.array(cache[1].state()[0].astype(mx.float32))[:, :, :2], initial
    )
    for c in cache:
        c.reset()
    assert all(c.offset == 0 for c in cache)
    replay = model(mx.array([[1, 2]]), cache=cache).logits
    np.testing.assert_array_equal(
        np.array(first.logits.astype(mx.float32)), np.array(replay.astype(mx.float32))
    )
    fresh = model.make_cache()
    batch = model(mx.array([[1, 2], [3, 4]]), cache=fresh).logits
    assert batch.shape == (2, 2, config.vocab_size)
    assert mx.all(mx.isfinite(batch)).item()
    # A fresh request's result does not depend on the previous request's carries.
    np.testing.assert_array_equal(
        np.array(model(mx.array([[1, 2]])).logits.astype(mx.float32)),
        np.array(replay.astype(mx.float32)),
    )


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
def test_invalid_cache_fails_before_any_layer_advances(mx, family):
    from mlx_one.core.registry import get_registration

    model = get_registration(family).model_class()(tiny_config(family))
    tokens = mx.array([[1, 2]])
    for kind in ("count", "type", "offset", "missing", "shape", "position"):
        cache = model.make_cache()
        kwargs = {}
        if kind == "count":
            cache = cache[:-1]
        elif kind == "type":
            cache = (*cache[:-1], object())
        elif kind in {"offset", "missing"}:
            cache[-1]._offset = -1 if kind == "offset" else 4
        elif kind == "shape":
            model(tokens, cache=cache)
            cache[-1].values = mx.zeros((2, 2, 64))
        else:
            kwargs["position_ids"] = mx.array([[-1, 0]])
        before = tuple(getattr(c, "offset", None) for c in cache)
        with pytest.raises(ValueError, match="cache|position|state"):
            model(tokens, cache=cache, **kwargs)
        assert tuple(getattr(c, "offset", None) for c in cache) == before


def test_lfm_sigmoid_bias_selection_unbiased_weights_and_scaling(mx):
    from mlx_one.models.language.lfm2_moe.model import LfmSparseMoe

    for normalized in (False, True):
        config = tiny_config("lfm2_moe", norm_topk_prob=normalized, routed_scaling_factor=1.7)
        block = LfmSparseMoe(config)
        block.gate.weight = mx.zeros((4, 64))
        block.gate.weight[:, 0] = mx.array([-2.0, -0.5, 0.8, 2.0])
        block.expert_bias = mx.array([2.0, 1.0, 0.0, -1.0])
        x = mx.ones((1, 2, 64))
        experts = block.switch_mlp
        for e in range(4):
            experts.gate_proj.weight[e] = mx.eye(64) * (e + 1) / 4
            experts.up_proj.weight[e] = mx.eye(64) * (e + 2) / 4
            experts.down_proj.weight[e] = mx.eye(64) * (e + 3) / 4
        actual = block(x)
        indices = np.array(block.top_indices)
        assert set(indices.ravel()) == {0, 1}
        prob = 1 / (1 + np.exp(-np.array([-2.0, -0.5])))
        scores = prob / (prob.sum() + 1e-6) if normalized else prob
        scores *= 1.7
        outputs = [
            ((e + 1) / 4 / (1 + np.exp(-(e + 1) / 4))) * ((e + 2) / 4) * ((e + 3) / 4)
            for e in (0, 1)
        ]
        np.testing.assert_allclose(np.array(actual), np.dot(scores, outputs), rtol=2e-6, atol=2e-6)


def test_gpt_selected_experts_and_capped_activation_match_explicit_math(mx):
    from mlx_one.models.language.gpt_oss.model import SparseMLP

    block = SparseMLP(tiny_config("gpt_oss"))
    block.router.weight = mx.zeros_like(block.router.weight)
    block.router.bias = mx.array([-3.0, 2.0, -1.0, 1.0])
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(block.experts, name)
        layer.weight = mx.zeros_like(layer.weight)
        layer.bias = mx.stack([mx.full((64,), value) for value in [-12, 8, -5, 1]])
    # down projection biases make the selected-expert weighted sum nonuniform.
    block.experts.down_proj.weight = mx.stack([mx.eye(64) * value for value in [1, 2, 3, 4]])
    actual = block(mx.ones((1, 2, 64)))
    weights = np.exp([2.0, 1.0])
    weights /= weights.sum()
    gate = np.minimum([8.0, 1.0], 7)
    up = np.clip([8.0, 1.0], -7, 7)
    outputs = (up + 1) * gate / (1 + np.exp(-1.702 * gate)) * [2, 4] + [8, 1]
    np.testing.assert_allclose(np.array(actual), weights @ outputs, rtol=1e-6)


def test_attention_sinks_window_boundary_and_yarn(mx):
    from mlx_one.models.language.gpt_oss.model import YarnRoPE
    from mlx_one.models.shared.decoder import attention_mask_for

    q = mx.zeros((1, 1, 1, 32))
    v = mx.ones_like(q)
    actual = mx.fast.scaled_dot_product_attention(q, q, v, scale=1, sinks=mx.array([0.0]))
    np.testing.assert_allclose(np.array(actual), 0.5)
    mask = np.isfinite(np.array(attention_mask_for(2, 5, 4, window=4)))
    np.testing.assert_array_equal(
        mask, [[False, True, True, True, True, False], [False, False, True, True, True, True]]
    )
    config = tiny_config("gpt_oss")
    rope = YarnRoPE(config)
    dims, base, original, factor = 32, 150000, 16, 32
    import math

    low = max(dims * math.log(original / (32 * 2 * math.pi)) / (2 * math.log(base)), 0)
    high = min(dims * math.log(original / (2 * math.pi)) / (2 * math.log(base)), dims - 1)
    ramp = np.clip((np.arange(dims // 2) - low) / (high - low), 0, 1)
    expected = base ** (np.arange(0, dims, 2) / dims) / (1 - ramp + ramp / factor)
    np.testing.assert_allclose(np.array(rope._freqs), expected, rtol=1e-6)
    x = mx.random.normal((1, 2, 2, 32))
    np.testing.assert_allclose(
        np.array(rope(x, offset=12)),
        np.array(rope(x, position_ids=mx.array([[12, 13]]))),
        atol=1e-6,
    )


def write_checkpoint(mx, root, family, dtype="float32"):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from mlx_one.core.quantization import module_quantization
    from mlx_one.core.registry import get_registration

    config = tiny_config(family)
    model = get_registration(family).model_class()(config)
    model.set_dtype(getattr(mx, dtype))
    if family == "lfm2_moe":
        for layer in model.model.layers[config.num_dense_layers :]:
            layer.feed_forward.gate.weight = layer.feed_forward.gate.weight.astype(mx.float32)
            layer.feed_forward.expert_bias = layer.feed_forward.expert_bias.astype(mx.float32)
    q = {"group_size": 32, "bits": 4, "mode": "affine" if family == "lfm2_moe" else "mxfp4"}
    q["model.embed_tokens"] = {
        "bits": 6 if family == "lfm2_moe" else 4,
        "group_size": 32,
        "mode": "affine",
    }
    if family == "gpt_oss":
        for index in range(config.num_hidden_layers):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                q[f"model.layers.{index}.self_attn.{name}"] = {
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                }
            q[f"model.layers.{index}.mlp.router"] = {"bits": 8, "group_size": 32, "mode": "affine"}
        q["lm_head"] = {"bits": 4, "group_size": 32, "mode": "affine"}
    nn.quantize(
        model,
        class_predicate=lambda path, module: (
            module_quantization(q, path)
            if hasattr(module, "to_quantized") and not path.endswith(".gate")
            else False
        ),
    )
    weights = dict(tree_flatten(model.parameters()))
    if family == "lfm2_moe":
        assert weights["model.layers.2.feed_forward.gate.weight"].dtype == mx.float32
        weights = {
            name.replace("feed_forward.w1.", "feed_forward.gate_proj.")
            .replace("feed_forward.w2.", "feed_forward.down_proj.")
            .replace("feed_forward.w3.", "feed_forward.up_proj."): value
            for name, value in weights.items()
        }
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(
        json.dumps({**asdict(config), "quantization": q, "eos_token_id": 2})
    )
    (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 15, 16]}))
    words = [
        "[UNK]",
        "<s>",
        "</s>",
        "hello",
        "Answer",
        "scratchpad-secret",
        "analysis",
        "final",
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "assistant",
        "EOS2",
        "EOS3",
    ]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(len(words)), strict=True)), "[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens([word for word in words if word.startswith("<")])
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>"})
    )
    if family == "gpt_oss":
        template = json.loads((METADATA / family / "tokenizer_config.json").read_text())[
            "chat_template"
        ]
    else:
        template = (METADATA / family / "chat_template.jinja").read_text()
    (root / "chat_template.jinja").write_text(template)
    return model


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
def test_safetensors_loader_python_engine_cli_http_and_sse(mx, family, tmp_path, monkeypatch):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one import CacheConfig, Engine
    from mlx_one.cli import main
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text import TextGenerationOptions, generate, stream_chat
    from mlx_one.text.loading import load_text_model

    write_checkpoint(mx, tmp_path, family)
    monkeypatch.setattr(
        "mlx_one.text.loading.snapshot_download", lambda *a, **k: pytest.fail("network")
    )
    bundle = load_text_model(tmp_path, offline=True)
    assert bundle.eos_token_ids == (2, 15, 16)
    assert bundle.quantization["model.embed_tokens"]["bits"] == (6 if family == "lfm2_moe" else 4)
    for name, module in bundle.model.named_modules():
        if family == "gpt_oss" and name.endswith("mlp.router"):
            assert module.bits == 8 and module.mode == "affine"
        if family == "gpt_oss" and name.endswith("experts.gate_proj"):
            assert module.mode == "mxfp4" and module.scales.dtype == mx.uint8

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


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
def test_malformed_packing_and_missing_weights_are_rejected(mx, family, tmp_path):
    from mlx_one.text.loading import TextModelLoadError, load_text_model

    write_checkpoint(mx, tmp_path, family)
    path = tmp_path / "model.safetensors"
    tensors = mx.load(str(path))
    mx.eval(tensors)
    bad = dict(tensors)
    key = (
        "model.layers.2.feed_forward.switch_mlp.gate_proj.weight"
        if family == "lfm2_moe"
        else "model.layers.2.mlp.experts.gate_proj.weight"
    )
    bad[key] = bad[key][..., :-1]
    mx.save_safetensors(str(path), bad)
    with pytest.raises(TextModelLoadError, match="shape_mismatch"):
        load_text_model(tmp_path, offline=True)
    bad = dict(tensors)
    bad[key] = bad[key].astype(mx.float32)
    mx.save_safetensors(str(path), bad)
    with pytest.raises(TextModelLoadError, match="uint32"):
        load_text_model(tmp_path, offline=True)
    path.unlink()
    with pytest.raises(TextModelLoadError, match="missing"):
        load_text_model(tmp_path, offline=True)


def test_unsupported_mxfp4_core_is_explicit(mx, monkeypatch):
    from mlx_one.core.quantization import UnsupportedQuantization
    from mlx_one.text.loading import _check_mxfp4_core

    monkeypatch.setattr(mx, "gather_qmm", lambda *a, **kw: (_ for _ in ()).throw(TypeError("mode")))
    with pytest.raises(
        UnsupportedQuantization, match="UNSUPPORTED_QUANTIZATION.*INCOMPATIBLE_MLX_CORE"
    ):
        _check_mxfp4_core()


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mixed_quantized_compute_dtypes(mx, family, dtype, tmp_path, monkeypatch):
    from mlx_one import CacheConfig, Engine
    from mlx_one.text.loading import load_text_model

    write_checkpoint(mx, tmp_path, family, dtype=dtype)
    bundle = load_text_model(tmp_path, offline=True)
    # No caller-side eval/forward before the bundle crosses the thread boundary.
    monkeypatch.setattr("mlx_one.text.generation.select_token", lambda *a, **kw: 4)
    prompt = "hello" if family == "lfm2_moe" else "<|start|>assistant<|channel|>final<|message|>"
    with Engine(
        bundle,
        inference_mode="reference_dense",
        cache_config=CacheConfig(prefix_cache_enabled=False),
    ) as engine:
        result = engine.generate(prompt, max_tokens=2)
        assert result.text.strip() == "Answer Answer" and result.generation_tokens == 2
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
    cache = bundle.model.make_cache()
    for tokens in ([[1, 2]], [[3]]):
        logits = bundle.model(mx.array(tokens), cache=cache).logits
        mx.eval(logits)
        assert logits.dtype == getattr(mx, dtype)
        assert mx.all(mx.isfinite(logits)).item()
    assert all(state.offset == 3 for state in cache)


def test_gpt_original_mxfp4_blocks_interleaved_experts(mx):
    from mlx.utils import tree_flatten

    from mlx_one.models.language.gpt_oss.model import GptOssForCausalLM
    from mlx_one.models.language.gpt_oss.weights import sanitize_weights
    from mlx_one.text.loading import _prepare_quantized_model, _validate_quantized_tensors

    config = tiny_config("gpt_oss")
    model = GptOssForCausalLM(config)
    weights = dict(tree_flatten(model.parameters()))
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}.mlp.experts."
        for suffix in ("weight", "bias"):
            gate = weights.pop(prefix + "gate_proj." + suffix)
            up = weights.pop(prefix + "up_proj." + suffix)
            value = mx.stack((gate, up), axis=2).reshape(
                config.num_local_experts, 2 * config.intermediate_size, *gate.shape[2:]
            )
            if suffix == "bias":
                weights[prefix + "gate_up_proj_bias"] = value
                continue
            packed, scales = mx.quantize(value, mode="mxfp4", group_size=32, bits=4)
            weights[prefix + "gate_up_proj_blocks"] = packed.view(mx.uint8).reshape(
                *packed.shape[:-1], config.hidden_size // 32, 16
            )
            weights[prefix + "gate_up_proj_scales"] = scales
        value = weights.pop(prefix + "down_proj.weight")
        packed, scales = mx.quantize(value, mode="mxfp4", group_size=32, bits=4)
        weights[prefix + "down_proj_blocks"] = packed.view(mx.uint8).reshape(
            *packed.shape[:-1], config.intermediate_size // 32, 16
        )
        weights[prefix + "down_proj_scales"] = scales
        weights[prefix + "down_proj_bias"] = weights.pop(prefix + "down_proj.bias")
    cleaned = sanitize_weights(weights, config)
    quant = {"mode": "mxfp4", "bits": 4, "group_size": 32}
    _prepare_quantized_model(model, cleaned, quant)
    _validate_quantized_tensors(model, cleaned)
    model.load_weights(list(cleaned.items()), strict=True)
    logits = model(mx.array([[1, 2]])).logits
    assert mx.all(mx.isfinite(logits)).item()
