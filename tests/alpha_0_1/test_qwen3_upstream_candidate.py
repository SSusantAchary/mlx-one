"""CPU-only candidate integration tests; these do not qualify native MLX numerics."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.core.cache import KVCache, QuantizedKVCache
from mlx_one.engine.modes import ModelCapabilityUnsupported
from mlx_one.models.language.qwen3.config import Qwen3Config
from mlx_one.models.language.qwen3.upstream_adapter import (
    DenseCacheView,
    cache_views,
    validate_candidate_runtime,
)


def config(**changes):
    return Qwen3Config.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 12,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 20,
            "tie_word_embeddings": True,
            **changes,
        }
    )


@pytest.fixture
def candidate(monkeypatch):
    """Use NumPy for control/data-flow tests without importing the Metal backend."""
    rng = np.random.default_rng(48)
    calls = []
    mx, nn, package = ModuleType("mlx.core"), ModuleType("mlx.nn"), ModuleType("mlx")

    class Module:
        pass

    class Linear(Module):
        def __init__(self, inputs, outputs, bias=False):
            assert bias is False
            self.weight = (rng.normal(size=(outputs, inputs)) * 0.1).astype(np.float32)

        def __call__(self, x):
            calls.append(("linear", self, x.shape))
            return x @ self.weight.T

    class Embedding(Module):
        def __init__(self, count, dim):
            self.weight = rng.normal(size=(count, dim)).astype(np.float32) * 0.1

        def __call__(self, ids):
            return self.weight[ids]

        def as_linear(self, x):
            calls.append(("tied_head", self, x.shape))
            return x @ self.weight.T

    class RMSNorm(Module):
        def __init__(self, dim, eps):
            self.weight, self.eps = np.ones(dim, np.float32), eps

        def __call__(self, x):
            return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)

    class RoPE(Module):
        def __init__(self, dims, **kwargs):
            self.dims, self.settings = dims, kwargs

        def __call__(self, x, offset=0):
            calls.append(("rope", offset, x.shape))
            positions = np.arange(offset, offset + x.shape[2]) * self.settings["scale"]
            freq = self.settings["base"] ** (-np.arange(self.dims // 2) * 2 / self.dims)
            angle = positions[:, None] * freq
            angle = np.concatenate((angle, angle), axis=-1)[None, None]
            rotated = np.concatenate((-x[..., self.dims // 2 :], x[..., : self.dims // 2]), -1)
            return (x * np.cos(angle) + rotated * np.sin(angle)).astype(x.dtype)

    def sdpa(q, k, v, **kwargs):
        calls.append(("sdpa", q.shape, k.shape, v.shape, kwargs))
        repeats = q.shape[1] // k.shape[1]
        scores = q @ np.repeat(k, repeats, axis=1).swapaxes(-1, -2) * kwargs["scale"]
        if kwargs["mask"] == "causal":
            valid = (
                np.arange(k.shape[2])[None, :]
                <= np.arange(k.shape[2] - q.shape[2], k.shape[2])[:, None]
            )
            scores = np.where(valid, scores, -np.inf)
        scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return (scores / scores.sum(axis=-1, keepdims=True)) @ np.repeat(v, repeats, axis=1)

    mx.compile = lambda f, **kwargs: f
    mx.fast = SimpleNamespace(scaled_dot_product_attention=sdpa)
    mx.arange, mx.all, mx.zeros = np.arange, np.all, np.zeros
    nn.Module, nn.Linear, nn.Embedding, nn.RMSNorm, nn.RoPE = (
        Module,
        Linear,
        Embedding,
        RMSNorm,
        RoPE,
    )
    nn.silu = lambda x: x / (1 + np.exp(-x))
    package.core, package.nn = mx, nn
    for name, module in [("mlx", package), ("mlx.core", mx), ("mlx.nn", nn)]:
        monkeypatch.setitem(sys.modules, name, module)
    for suffix in ["upstream_math", "upstream_model"]:
        name = "mlx_one.models.language.qwen3." + suffix
        parent = importlib.import_module("mlx_one.models.language.qwen3")
        monkeypatch.setattr(parent, suffix, None, raising=False)
        monkeypatch.delitem(sys.modules, name, raising=False)
        # Register the new import with monkeypatch so fake-backed modules never leak.
        module = importlib.import_module(name)
        del sys.modules[name]
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(model=module.Qwen3UpstreamForCausalLM, calls=calls)


@pytest.mark.parametrize("tied", [True, False])
def test_prefill_incremental_cache_reset_and_output_contract(candidate, tied):
    model = candidate.model(config(tie_word_embeddings=tied))
    ids = np.array([[1, 3, 4, 2]])
    full = model(ids, output_hidden_states=True)
    assert full.logits.shape == (1, 4, 20)
    assert full.hidden_states[-1] is full.last_hidden_state
    assert len(full.hidden_states) == 3
    cache = model.make_cache()
    first = model(ids[:, :2], cache=cache)
    assert first.cache is cache and [c.offset for c in cache] == [2, 2]
    rest = model(ids[:, 2:], cache=cache)
    np.testing.assert_allclose(rest.logits, full.logits[:, 2:], atol=1e-4, rtol=1e-4)
    assert [c.offset for c in cache] == [4, 4]
    assert all(c.state()[0].shape == (1, 2, 4, 4) for c in cache)
    for c in cache:
        c.reset()
    np.testing.assert_array_equal(model(ids, cache=cache).logits, full.logits)
    np.testing.assert_allclose(
        model(ids, logits_mode="last_only").logits, full.logits[:, -1:], atol=1e-4, rtol=1e-4
    )
    assert model(ids, logits_mode="no_logits").logits is None
    assert any(c[0] == "tied_head" for c in candidate.calls) is tied


def test_direct_projection_dispatch_rope_grouping_and_sdpa(candidate):
    model = candidate.model(config(rope_scaling={"type": "linear", "factor": 2}))
    q = model.layers[0].self_attn
    q.q_proj.bits, q.q_proj.mode = 8, "affine"
    assert q.rope.settings == {"traditional": False, "base": 1e6, "scale": 0.5}
    model(np.array([[1, 2, 3]]), cache=model.make_cache())
    attention = [c for c in candidate.calls if c[0] == "sdpa"]
    assert len(attention) == 2
    assert attention[0][1:4] == ((1, 4, 3, 4), (1, 2, 3, 4), (1, 2, 3, 4))
    assert attention[0][4] == {"scale": 0.5, "mask": "causal", "sinks": None}
    assert any(c[0] == "linear" and c[1] is q.q_proj for c in candidate.calls)


def test_adapter_rejects_unsupported_inputs_before_mutating_cache(candidate):
    model = candidate.model(config())
    cache = model.make_cache()
    for kwargs in [
        {"input_ids": np.array([[1], [2]])},
        {"input_ids": np.array([[1, 2]]), "attention_mask": np.array([[0, 1]])},
        {"input_ids": np.array([[1, 2]]), "position_ids": np.array([[2, 1]])},
    ]:
        with pytest.raises(ModelCapabilityUnsupported):
            model(cache=cache, **kwargs)
        assert [c.offset for c in cache] == [0, 0]
    ids = np.array([[1, 2]])
    np.testing.assert_array_equal(
        model(ids).logits,
        model(ids, attention_mask=np.ones_like(ids), position_ids=np.array([[0, 1]])).logits,
    )


def test_cache_bridge_uses_owner_and_rejects_incompatible_state():
    owner = KVCache()
    view = DenseCacheView(owner)
    assert view.owner is owner and view.offset == 0
    assert view.make_mask(1) is None and view.make_mask(3) == "causal"
    assert DenseCacheView(QuantizedKVCache(None, None)).offset == 0
    with pytest.raises(ModelCapabilityUnsupported, match="dense KV"):
        DenseCacheView(QuantizedKVCache(8, 8))
    with pytest.raises(ModelCapabilityUnsupported):
        DenseCacheView(SimpleNamespace(offset=0, attend=lambda *a: None))
    with pytest.raises(ValueError, match="distinct"):
        cache_views((owner, owner), 2)
    other = KVCache()
    other._offset = 1
    with pytest.raises(ValueError, match="positions"):
        cache_views((owner, other), 2)
    with pytest.raises(ValueError, match="count"):
        cache_views((owner,), 2)


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "paged"},
        {"mode": "auto"},
        {"mode": "native_optimized"},
        {"backend": "hybrid"},
        {"key_type": "q8_0"},
        {"value_type": "bf16"},
        {"max_sequences": 2},
        {"spec_type": "draft-model"},
    ],
)
def test_candidate_runtime_fails_closed(settings):
    bundle = SimpleNamespace(model=SimpleNamespace(_mlx_one_qwen3_candidate=True))
    with pytest.raises(ModelCapabilityUnsupported, match="reference dense B1"):
        validate_candidate_runtime(bundle, **settings)
    validate_candidate_runtime(SimpleNamespace(model=object()), **settings)


def test_upstream_notice_packaging_and_default_registry_are_preserved():
    from setuptools.config.pyprojecttoml import read_configuration

    from mlx_one.core.registry import get_registration

    root = Path(__file__).resolve().parents[2]
    assert get_registration("qwen3").model_path.endswith("qwen3.model:Qwen3ForCausalLM")
    package = root / "src/mlx_one/models/language/qwen3"
    notice = (package / "UPSTREAM_LICENSE").read_text()
    assert "MIT License" in notice and "Copyright © 2023 Apple Inc." in notice
    metadata = read_configuration(root / "pyproject.toml", expand=False)
    assert (
        "models/language/qwen3/UPSTREAM_LICENSE"
        in (metadata["tool"]["setuptools"]["package-data"]["mlx_one"])
    )
    assert not any(dep.startswith("mlx-lm") for dep in metadata["project"]["dependencies"])


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("use_candidate", [False, True])
def test_loader_keeps_strict_identity_weight_and_quantization_pipeline(
    candidate, monkeypatch, tmp_path, quantized, use_candidate
):
    import mlx_one.conversion as conversion
    import mlx_one.text.loading as loading

    settings = config().__dict__.copy()
    settings.pop("extra")
    if quantized:
        settings["quantization"] = {"bits": 8, "group_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(settings))
    events = []

    class Loaded:
        def __init__(self, config):
            self.config = config
            events.append(("candidate" if use_candidate else "current", config))

        def load_weights(self, weights, *, strict):
            assert strict is True and weights == [("fixture", "packed-payload")]
            events.append(("load", strict))

        def eval(self):
            events.append(("eval", True))

    registration = SimpleNamespace(
        loader_path="unchanged-native-loader",
        supports=lambda task: task == "text-generation",
        config_class=lambda: Qwen3Config,
        model_class=lambda: Loaded,
        sanitizer=lambda: lambda weights, config: weights,
        weight_contract=lambda: (
            lambda config: SimpleNamespace(
                validate=lambda weights: events.append(("contract", weights))
            )
        ),
    )
    monkeypatch.setattr(loading, "get_registration", lambda kind: registration)
    monkeypatch.setattr(conversion, "verify_conversion_manifest", lambda path: None)
    monkeypatch.setattr(loading, "_load_tokenizer", lambda *args: SimpleNamespace())
    monkeypatch.setattr(loading, "_check_checkpoint_memory", lambda root: None)
    monkeypatch.setattr(loading, "_materialize_loaded_model", lambda model: None)
    monkeypatch.setattr(loading.ChatTemplate, "from_directory", lambda *args: None)
    monkeypatch.setattr(loading, "_read_safetensors", lambda root: {"fixture": "packed-payload"})
    monkeypatch.setattr(
        loading, "_prepare_quantized_model", lambda *args: events.append(("prepare", args[2]))
    )
    monkeypatch.setattr(
        loading, "_validate_quantized_tensors", lambda *args: events.append(("validate", True))
    )
    module = sys.modules["mlx_one.models.language.qwen3.upstream_model"]
    monkeypatch.setattr(module, "Qwen3UpstreamForCausalLM", Loaded)
    bundle = loading.load_text_model(tmp_path, offline=True, _qwen3_candidate=use_candidate)
    assert isinstance(bundle.model.config, Qwen3Config)
    assert bundle.model_id == str(tmp_path.resolve()) and bundle.architecture == "qwen3"
    assert [name for name, _ in events] == [
        "candidate" if use_candidate else "current",
        *(["prepare", "validate"] if quantized else ["contract"]),
        "load",
        "eval",
    ]
    assert bundle.quantization == (
        {"bits": 8, "group_size": 64, "mode": "affine"} if quantized else {}
    )


def test_candidate_loader_rejects_other_architectures_before_tensor_read(monkeypatch, tmp_path):
    import mlx_one.text.loading as loading

    (tmp_path / "config.json").write_text('{"model_type":"gpt2"}')
    monkeypatch.setattr(loading, "_read_safetensors", lambda path: pytest.fail("tensor read"))
    with pytest.raises(loading.TextModelLoadError, match="requires model_type=qwen3"):
        loading.load_text_model(tmp_path, _qwen3_candidate=True)


@pytest.mark.parametrize(
    "options",
    [
        {"inference_mode": "paged"},
        {"spec_type": "draft-model"},
    ],
)
def test_engine_rejects_candidate_experiments_before_dtype_or_runtime_creation(options):
    from mlx_one.engine import Engine
    from mlx_one.text.loading import LoadedTextModel

    bundle = LoadedTextModel(
        SimpleNamespace(_mlx_one_qwen3_candidate=True),
        SimpleNamespace(),
        Path("."),
        "candidate-test",
        "immutable",
        architecture="qwen3",
    )
    with pytest.raises(ModelCapabilityUnsupported, match="reference dense B1"):
        Engine(bundle, **options)


@pytest.mark.parametrize(
    "kind,implementation",
    [
        ("gpt2", "upstream-candidate"),
        ("qwen3", "unrecognized"),
    ],
)
def test_task_profile_cannot_silently_select_candidate(kind, implementation):
    from mlx_one.task_qualification import TaskProfile

    with pytest.raises(ValueError, match="Qwen3 implementation"):
        TaskProfile(
            model_type=kind,
            task="text-generation",
            model_id="test-only",
            revision="1" * 40,
            checkpoint_path="unused",
            inputs_manifest="unused",
            inputs_sha256="2" * 64,
            settings={"qwen3_implementation": implementation},
        )
