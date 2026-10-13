"""Backend-free P1 contracts and differential runtime regressions."""

import pytest
from test_dense_batching import native as native
from test_paged_hybrid import advanced as advanced

from mlx_one.p1_profiles import CHECKPOINTS, REFERENCE_PINS


def test_p1_pins_are_immutable_and_converted_models_are_explicit():
    assert len(CHECKPOINTS) == 6 and len(REFERENCE_PINS) == 3
    assert all(len(c.revision) == 40 for c in CHECKPOINTS.values())
    assert [name for name, c in CHECKPOINTS.items() if c.conversion_required] == [
        "smol-small", "smol-large"]
    assert CHECKPOINTS["smol-small"].context_tokens == 8192


def test_int8_metadata_and_module_layout_validation():
    from types import SimpleNamespace

    import numpy as np
    import pytest

    from mlx_one.engine.adapters import affine_tensor_spec, native_adapter
    from mlx_one.text.loading import _quantization_config

    assert _quantization_config({"quantization": {"bits": 8, "group_size": 64}})["bits"] == 8
    for invalid in ({"bits": True}, {"bits": 8, "mode": "mxfp4"}):
        with pytest.raises(ValueError):
            _quantization_config({"quantization": invalid})
    weight = np.zeros((4, 16), dtype=np.uint32)
    scales = np.ones((4, 1), dtype=np.float16)
    assert affine_tensor_spec("linear", weight, scales, scales, bits=8,
                              group_size=64, mode="affine").bits == 8
    with pytest.raises(ValueError, match="packing mismatch"):
        affine_tensor_spec("linear", weight, scales[:, :0], scales,
                           bits=8, group_size=64, mode="affine")
    config = SimpleNamespace(num_attention_heads=16, num_key_value_heads=8,
                             head_dim=128, hidden_size=1024)
    bundle = SimpleNamespace(model=SimpleNamespace(config=config,
        parameters=lambda: {"weight": np.ones(1, dtype=np.float16)}), quantization={})
    plan = SimpleNamespace(architecture="qwen3", block_compatible=True,
                           topology_digest="topology", state_specs=[])
    assert native_adapter(bundle, plan).spec.head_dim == 128


def test_explicit_float_conversion_preserves_packed_weights(monkeypatch):
    from types import ModuleType, SimpleNamespace

    import numpy as np

    from mlx_one.engine.modes import configure_execution_dtype

    packed = np.arange(32, dtype=np.uint32).reshape(2, 16)
    module = SimpleNamespace(weight=packed, scales=np.ones((2, 1), dtype=np.float32),
                             biases=np.zeros((2, 1), dtype=np.float32), bits=8,
                             group_size=64, mode="affine")
    class Model:
        def parameters(self):
            return {"linear": {name: getattr(module, name)
                               for name in ("weight", "scales", "biases")}}

        def named_modules(self):
            return [("linear", module)]

        def set_dtype(self, dtype, predicate):
            for name in ("weight", "scales", "biases"):
                value = getattr(module, name)
                if predicate(value.dtype):
                    setattr(module, name, value.astype(dtype))
    import sys

    core = ModuleType("mlx.core")
    core.float16, core.eval = np.float16, lambda *args: None
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    bundle = SimpleNamespace(model=Model(), quantization={"bits": 8, "group_size": 64})
    configure_execution_dtype(bundle, "float16")
    assert module.weight is packed
    assert module.scales.dtype == np.float16
    assert bundle.model._mlx_one_execution_policy["quantized_payload_preserved"]


@pytest.fixture
def int8_native(native):
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np

    module = SimpleNamespace(weight=np.zeros((2, 16), dtype=np.uint32),
                             scales=np.ones((2, 1), dtype=np.float16),
                             biases=np.zeros((2, 1), dtype=np.float16),
                             bits=8, group_size=64, mode="affine")
    original = native.model.parameters
    native.model.named_modules = lambda: [("linear", module)]
    native.model.parameters = lambda: {**original(), "linear": {
        name: getattr(module, name) for name in ("weight", "scales", "biases")}}
    native.model.config.model_type = "qwen3"
    return replace(native, architecture="qwen3", quantization={"bits": 8, "group_size": 64,
                                                              "mode": "affine"})


@pytest.mark.parametrize("size", [1, 2, 4])
def test_qwen3_int8_cpu_batch_forwards_and_independent_output(int8_native, size):
    from mlx_one import BatchingConfig, Engine

    prompts = ["x" * (i + 1) for i in range(size * 2)]
    with Engine(int8_native, inference_mode="experimental_dense_batch", batching_config=
                BatchingConfig(max_sequences=size, queue_size=size * 2, prefill_chunk_size=2,
                               max_prefill_tokens=8, max_scheduled_tokens=16)) as engine:
        tasks = [engine.submit(prompt, max_tokens=8) for prompt in prompts]
        actual = [task.result(timeout=5).text for task in tasks]
        traces = engine.runtime_stats()["scheduler"]["forward_traces"]
        assert all(t["batch_size"] <= size for t in traces)
        if size > 1:
            assert any(t["batch_size"] > 1 and t["phase"] == "decode" for t in traces)
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
    with Engine(int8_native) as oracle:
        expected = [oracle.generate(prompt, max_tokens=8).text for prompt in prompts]
    assert actual == expected


@pytest.mark.parametrize("chunk", [128, 256, 512, 1024])
def test_p1_profiles_preserve_default_budgets_and_precision(chunk):
    from mlx_one.p1_profiles import profile

    value = profile("qwen3-small", mode="experimental_dense_batch", slots=4,
                    chunk_tokens=chunk)
    assert value.repeats == 5 and value.max_scheduled_tokens == 2048
    assert value.max_prefill_tokens == 1024 and value.cache_type == "f16"
    assert value.execution_dtype == "float16"


def test_8k_prefill_yields_to_decoders_with_bounded_progress():
    from types import SimpleNamespace

    from mlx_one.engine.batching import BatchingConfig, BatchScheduler

    config = BatchingConfig(max_sequences=4, max_scheduled_tokens=1024,
                            max_prefill_tokens=512, prefill_chunk_size=512)
    scheduler = BatchScheduler(config)
    decoders = [SimpleNamespace(phase="decode", token_ids=(1,), request_id=str(i))
                for i in range(3)]
    long = SimpleNamespace(phase="prefill", token_ids=(1,) * 512, request_id="long")
    committed = {r.request_id: 0 for r in [*decoders, long]}
    for _ in range(16):
        groups = scheduler.select([*decoders, long])
        assert groups[0][0].phase == "decode"
        assert sum(len(r.token_ids) for g in groups for r in g) <= 1024
        for group in groups:
            for request in group:
                committed[request.request_id] += len(request.token_ids)
    assert committed == {"0": 16, "1": 16, "2": 16, "long": 8192}


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("page_width", [16, 32])
def test_qwen3_int8_paged_adapter_ragged_parity(advanced, size, page_width):
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np
    from test_paged_hybrid import _requests

    from mlx_one import CacheConfig
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.paged import PagedRunner

    bundle = advanced()
    module = SimpleNamespace(weight=np.zeros((2, 16), dtype=np.uint32),
        scales=np.ones((2, 1), dtype=np.float16), biases=np.zeros((2, 1), dtype=np.float16),
        bits=8, group_size=64, mode="affine")
    original = bundle.model.parameters
    bundle.model.named_modules = lambda: [("linear", module)]
    bundle.model.parameters = lambda: {**original(), "packed": module.weight}
    bundle = replace(bundle, architecture="qwen3", quantization={"bits": 8, "group_size": 64})
    runtime = CacheManager(bundle, CacheConfig(backend="paged", block_size_tokens=page_width))
    requests = _requests(bundle, runtime, [range(1, page_width + i + 2) for i in range(size)])
    independent = [bundle.model.make_cache() for _ in requests]
    runner = PagedRunner()
    try:
        for step in range(3):
            result = runner.execute(tuple(requests))
            assert bundle.model.calls[-1][0] == size
            for i, request in enumerate(requests):
                expected = bundle.model(np.array([request.token_ids]), cache=independent[i])
                np.testing.assert_allclose(result.logits[i], expected.logits[:, -1:],
                                           atol=.03, rtol=.01)
                request.phase, request.token_ids = "decode", (step + 9,)
            assert runner.last_trace["history_materializations"] == 0
    finally:
        runtime.close()
    assert runtime.stats().physical_bytes == 0


def test_int8_prefix_identity_includes_observed_modules_and_float_policy(int8_native):
    from mlx_one import CacheConfig
    from mlx_one.engine.cache_manager import CacheManager

    manager = CacheManager(int8_native, CacheConfig())
    try:
        first = manager.fingerprint(namespace="same")
        int8_native.model._mlx_one_execution_policy = {"version": "changed", "dtype": "float16"}
        assert manager.fingerprint(namespace="same") != first
        assert manager.fingerprint(namespace="other") != first
    finally:
        manager.close()


def test_int8_dense_prefix_reuse_skips_committed_inputs(int8_native):
    from mlx_one import BatchingConfig, Engine

    with Engine(int8_native, inference_mode="experimental_dense_batch", cache_namespace="same",
                batching_config=BatchingConfig(max_sequences=2, prefill_chunk_size=2)) as engine:
        first = engine.generate("shared prompt", max_tokens=3)
        before = len(int8_native.model.calls)
        second = engine.generate("shared prompt", max_tokens=3)
        calls = int8_native.model.calls[before:]
        assert first.text == second.text
        assert engine.runtime_stats()["cache_stats"]["prefix_hits"] > 0
        assert sum(shape[0][1] for shape in calls) < len("shared prompt") + 2
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0


def test_smol_uses_shared_adapter_and_enforces_total_context(int8_native):
    from dataclasses import replace

    from mlx_one import BatchingConfig, Engine
    from mlx_one.inference_baseline import BaselineError
    from mlx_one.p1_profiles import profile

    bundle = replace(int8_native, architecture="llama", context_length=8192)
    with Engine(bundle, inference_mode="experimental_dense_batch",
                batching_config=BatchingConfig(max_sequences=2)) as engine:
        tasks = [engine.submit(s, max_tokens=4) for s in ("short", "longer input")]
        assert all(t.result(timeout=5).text for t in tasks)
    with pytest.raises(BaselineError, match="prompt plus output"):
        profile("smol-small", mode="paged", prompt_tokens=8192, output_tokens=32)
    with pytest.raises(BaselineError, match="verified pinned"):
        profile("smol-small", mode="paged", prompt_tokens=8160, output_tokens=32)


def test_converted_artifact_identity_is_content_derived(tmp_path):
    from mlx_one.conversion import (
        ConversionError,
        verify_conversion_manifest,
        write_conversion_manifest,
    )
    from mlx_one.p1_profiles import profile

    asset = CHECKPOINTS["smol-small"]
    (tmp_path / "model.safetensors").write_bytes(b"test payload")
    manifest = write_conversion_manifest(tmp_path, {"schema_version": 2,
        "source": asset.model_id, "revision": asset.revision, "precision": "int8-affine",
        "group_size": 64, "floating_dtype": "float16", "converter": "mlx-one-native-v2"})
    assert verify_conversion_manifest(tmp_path) == manifest
    result = profile("smol-small", mode="paged", prompt_tokens=8160, output_tokens=32,
                     checkpoint_path=str(tmp_path))
    assert result.revision == manifest["artifact_digest"]
    (tmp_path / "model.safetensors").write_bytes(b"changed payload")
    with pytest.raises(ConversionError, match="checksum"):
        verify_conversion_manifest(tmp_path)


@pytest.fixture
def int8_hybrid(advanced):
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np

    bundle = advanced(True)
    module = SimpleNamespace(weight=np.zeros((2, 16), dtype=np.uint32),
        scales=np.ones((2, 1), dtype=np.float16), biases=np.zeros((2, 1), dtype=np.float16),
        bits=8, group_size=64, mode="affine")
    parameters = bundle.model.parameters
    bundle.model.named_modules = lambda: [("linear", module)]
    bundle.model.parameters = lambda: {**parameters(), "packed": module.weight}
    return replace(bundle, quantization={"bits": 8, "group_size": 64, "mode": "affine"})


@pytest.mark.parametrize("mode", ["experimental_hybrid_batch", "paged"])
@pytest.mark.parametrize("size", [1, 2, 4])
def test_int8_lfm_full_state_ragged_parity_and_cancellation(int8_hybrid, mode, size):
    import threading

    import numpy as np
    from test_paged_hybrid import _requests

    from mlx_one import CacheConfig
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.hybrid import HybridRunner

    runtime = CacheManager(int8_hybrid, CacheConfig(
        backend="paged" if mode == "paged" else "hybrid", block_size_tokens=16))
    requests = _requests(int8_hybrid, runtime, [range(i + 1, 20) for i in range(size)])
    independent = [int8_hybrid.model.make_cache() for _ in requests]
    runner = HybridRunner()
    try:
        for step in range(3):
            result = runner.execute(tuple(requests))
            assert int8_hybrid.model.calls[-1][0] == size
            for i, request in enumerate(requests):
                expected = int8_hybrid.model(np.array([request.token_ids]), cache=independent[i])
                np.testing.assert_allclose(result.logits[i], expected.logits[:, -1:],
                                           atol=.03, rtol=.01)
                np.testing.assert_array_equal(request.cache[0].values, independent[i][0].values)
                assert len({c.offset for c in request.cache}) == 1
                request.phase, request.token_ids = "decode", (step + 9,)
        if size > 1:
            offset = requests[0].cache[0].offset
            requests[0].cancel = threading.Event()
            int8_hybrid.model.pause = requests[0].cancel.set
            runner.execute(tuple(requests))
            assert requests[0].cache[0].offset == offset
            assert requests[1].cache[0].offset == independent[1][0].offset + 1
        if mode == "paged":
            assert runner.last_trace["direct_attention_calls"] == 1
            assert runner.last_trace["history_materializations"] == 0
    finally:
        runtime.close()
    assert runtime.stats().physical_bytes == 0


def test_hybrid_paged_actor_restore_shares_attention_and_copies_carries(int8_hybrid):
    from mlx_one import BatchingConfig, Engine
    from mlx_one.engine.pressure import MemoryPressureConfig

    with Engine(int8_hybrid, inference_mode="paged", cache_namespace="principal:test",
                memory_pressure_config=MemoryPressureConfig(enabled=False),
                batching_config=BatchingConfig(max_sequences=2, prefill_chunk_size=16)) as engine:
        prompt = "A repeated prompt with a sealed page. " * 2
        donor = engine.generate(prompt, max_tokens=5)
        repeated = engine.generate(prompt, max_tokens=5)
        assert donor.text == repeated.text
        stats = engine.runtime_stats()["cache_stats"]
        assert stats["prefix_hits"] > 0 and stats["active_handles"] == 0
        engine._cache_manager.clear_prefixes()
        assert engine.runtime_stats()["cache_stats"]["reserved_bytes"] == 0


def test_paged_hybrid_snapshot_restores_full_state(int8_hybrid, tmp_path):
    import numpy as np
    from test_paged_hybrid import _requests

    from mlx_one import CacheConfig
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.components import load_snapshot, save_snapshot
    from mlx_one.engine.hybrid import HybridRunner

    runtime = CacheManager(int8_hybrid, CacheConfig(backend="paged", block_size_tokens=16))
    requests = _requests(int8_hybrid, runtime, [range(16)])
    restored = None
    try:
        HybridRunner().execute(tuple(requests))
        identity = {"topology": runtime.plan.topology_digest, "revision": int8_hybrid.revision}
        manifest = save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
        assert manifest["schema_version"] == "hybrid-state-v3"
        restored = load_snapshot(tmp_path, identity=identity,
            expected_specs=runtime.plan.state_specs, page_pool=runtime.block_pool)
        assert restored.offset == 16 and restored.entries[1].capabilities().can_page
        np.testing.assert_array_equal(restored.entries[0].values, requests[0].cache[0].values)
        with pytest.raises(ValueError, match="matching page arena"):
            load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
    finally:
        if restored:
            restored.release()
        runtime.close()
