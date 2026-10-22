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
    from mlx_one.engine.cache_specs import CacheKind, CachePlan, LayerCacheSpec

    plan = CachePlan("qwen3", None, "qwen3",
        (LayerCacheSpec(0, CacheKind.ATTENTION_KV, 8, 128, 128,
                        block_compatible=True, num_query_heads=16),), 128)
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
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
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


def test_p1_gate_requires_targets_and_complete_native_observations():
    from copy import deepcopy
    from dataclasses import asdict

    from mlx_one.p1_decode import DECODE_VERSION
    from mlx_one.p1_profiles import profile
    from mlx_one.p1_qualification import evaluate_p1

    item = profile("qwen3-small", mode="paged", slots=4, ttft_budget_ms=1000,
                   decode_gap_budget_ms=100, max_queue_wait_ms=2000, max_service_gap_ms=100)
    identity = {key: "unit-test" for key in ("hardware", "source", "weight_precision",
        "cache_precision", "mlx_version", "quantization_digest", "floating_policy")}
    summary = {"identity": identity, "aggregate_decode_tokens_per_second_by_group": [125] * 5,
        "controlled_decode": {"schema_version": DECODE_VERSION, "input_identity": "tokens",
                              "rates": [125] * 5},
        "comparison_inputs": {"traffic_identity": "traffic", "rendered_tokens_sha256": "tokens",
                              "generated_tokens_sha256": "output"},
        "failed": 0, "cleanup_passed": True, "starved_requests": 0, "queue_wait_p95_ms": 10,
        "max_admission_wait_ms": 10,
        "max_runnable_service_gap_ms": 10, "warm_client_ttft_p95_ms": 100,
        "model_token_gap_p95_ms": 15}
    serial = {"native_execution": True,
        "profile": {**asdict(item), "slots": 1, "inference_mode": "reference_dense"},
        "identity": identity, "summary": {**summary,
        "controlled_decode": {**summary["controlled_decode"], "rates": [100] * 5},
        "aggregate_decode_tokens_per_second_by_group": [100] * 5}}
    b1 = {**serial, "profile": {**serial["profile"], "inference_mode": "paged"},
          "summary": {**serial["summary"], "model_token_gap_p95_ms": 10}}
    traces = [{"phase": "decode", "batch_size": len(ids), "request_ids": ids,
               "direct_attention_calls": 1, "history_materializations": 0}
              for ids in (["a", "b", "c", "d"], ["b", "c", "d", "e"])]
    parity = {"status": "passed"}
    assert evaluate_p1(item, summary, traces, parity, serial=serial, b1=b1)["status"] == "passed"
    bad = deepcopy(summary)
    bad["controlled_decode"]["rates"][-1] = 124.9
    result = evaluate_p1(item, bad, traces, parity, serial=serial, b1=b1)
    assert result["status"] == "failed" and not result["checks"]["decode_target"]
    assert evaluate_p1(item, summary, traces, parity)["status"] == "pending"
    bad = deepcopy(b1)
    bad["summary"]["controlled_decode"]["rates"][-1] = 94.9
    assert not evaluate_p1(item, summary, traces, parity, serial=serial, b1=bad)[
        "checks"]["b1_regression"]


def test_p1_profile_cli_is_backend_free_and_keeps_native_pending():
    import json
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-c", """
import sys
from mlx_one.p1_benchmark import main
assert main(['profiles', '--prompt-tokens', '128', '--output-tokens', '32',
             '--slots', '1', '2', '4']) == 0
assert not any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules)
"""], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    assert len(value["profiles"]) == 3 and value["native_qualification"] == "pending"
    assert all(p["repeats"] == 5 and p["qualification_track"] == "p1" for p in value["profiles"])


def test_p1_decode_aggregation_excludes_prefill_and_first_tokens():
    from mlx_one.inference_baseline import summarize

    rows = [{"status": "completed", "temperature_phase": "warm", "group_id": "0",
             "started": 0, "ended": 100, "generated_tokens": 3,
             "metrics": {"model_generation_started_at": 0,
                         "model_token_times_ms": [10000, 11000, 12000]}}]
    value = summarize(rows)
    assert value["aggregate_output_tokens_per_second_by_group"] == [.03]
    assert value["aggregate_decode_tokens_per_second_by_group"] == [1]


def test_paged_hybrid_lifecycle_resume_fork_cancel_reload(int8_hybrid, tmp_path):
    from mlx_one.engine.lifecycle import hybrid_lifecycle
    from mlx_one.inference_baseline import BaselineProfile

    item = BaselineProfile(int8_hybrid.model_id, int8_hybrid.revision, "paged",
                           workload="lifecycle", cache_type="f16", page_tokens=16)
    result = hybrid_lifecycle(int8_hybrid, item, tmp_path, reload_bundle=lambda: int8_hybrid)
    assert result["status"] == "passed", result
    assert all(c["active_handles"] == c["reserved_bytes"] == 0 for c in result["cleanup"])


def test_int8_performance_cannot_promote_without_p1_gate(int8_native, tmp_path):
    import json

    from native_evidence import attach_provenance

    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.reporting import native_report
    from mlx_one.registry import (
        InferenceFeatureRegistry,
        InferenceFeatureStatus,
        InferenceScope,
        RegistryError,
    )

    runtime = CacheManager(int8_native)
    try:
        report = native_report(int8_native, runtime, mode="experimental_dense_batch")
        scope = {**report["scope"], "cache_precision": "float16"}
        proof = {"schema_version": "native-inference-qualification-v4", "backend": "mlx-one-native",
            "feature": "continuous_batching", "scope": scope, "status": "passed",
            "correctness_passed": True, "performance_qualified": True,
            "execution_contract": report["execution_contract"]}
        path = tmp_path / "proof.json"
        path.write_text(json.dumps(attach_provenance(proof, tmp_path)))
        registry = InferenceFeatureRegistry()
        with pytest.raises(RegistryError, match="measured P1 gates"):
            registry.promote("continuous_batching", InferenceScope(**scope), path,
                             InferenceFeatureStatus.PERFORMANCE_QUALIFIED)
        proof["schema_version"] = "native-inference-qualification-v3"
        path.write_text(json.dumps(proof))
        from mlx_one.engine.selection import load_evidence

        legacy = load_evidence([path])
        assert legacy.records()[0].status.value == "experimental"
    finally:
        runtime.close()


def test_qwen3_int8_dense_one_b4_forward_with_unequal_histories(int8_native):
    import numpy as np
    from test_paged_hybrid import _requests

    from mlx_one.engine.batching import BatchedDenseRunner
    from mlx_one.engine.cache_manager import CacheManager

    runtime = CacheManager(int8_native)
    requests = _requests(int8_native, runtime, [(1,), (2, 3), (4, 5, 6), (7, 8, 9, 10)])
    references = [int8_native.model.make_cache() for _ in requests]
    runner = BatchedDenseRunner()
    try:
        for request, cache in zip(requests, references, strict=True):
            runner.execute((request,))
            int8_native.model(np.array([request.token_ids]), cache=cache)
            request.phase, request.token_ids = "decode", (11,)
        result = runner.execute(tuple(requests))
        assert int8_native.model.calls[-1][0][0] == 4
        assert runner.last_trace["history_lengths"] == [1, 2, 3, 4]
        for logits, cache in zip(result.logits, references, strict=True):
            expected = int8_native.model(np.array([[11]]), cache=cache).logits[:, -1:]
            np.testing.assert_array_equal(logits, expected)
    finally:
        runtime.close()
    assert runtime.stats().reserved_bytes == runtime.stats().active_handles == 0


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
def test_int8_http_streaming_batching_and_principal_isolation(int8_native, architecture):
    from dataclasses import replace

    from test_dense_batching import test_http_batched_requests_principal_isolation_and_sse

    test_http_batched_requests_principal_isolation_and_sse(
        replace(int8_native, architecture=architecture))


def test_existing_affine_int4_hybrid_omitted_policy_is_preserved(int8_hybrid):
    from dataclasses import replace

    import numpy as np

    from mlx_one import Engine

    module = int8_hybrid.model.named_modules()[0][1]
    module.bits = 4
    module.weight = np.zeros((2, 8), dtype=np.uint32)
    bundle = replace(int8_hybrid, quantization={"bits": 4, "group_size": 64, "mode": "affine"})
    with Engine(bundle, inference_mode="experimental_hybrid_batch") as engine:
        actual = engine.generate("Legacy hybrid path", max_tokens=4)
    with Engine(bundle) as reference:
        assert actual.text == reference.generate("Legacy hybrid path", max_tokens=4).text
    assert not hasattr(bundle.model, "_mlx_one_execution_policy")


def test_p1_harness_rejects_precision_substitution_and_records_pressure():
    from test_inference_baseline import identity

    from mlx_one.inference_baseline import BaselineError, validate_identity
    from mlx_one.p1_profiles import profile

    item = profile("qwen3-small", mode="experimental_dense_batch")
    runtime = identity(item)
    runtime["inference"]["capabilities"]["continuous_batching"] = True
    runtime["inference"]["scope"]["cache_precision"] = "float16"
    with pytest.raises(BaselineError, match="observed affine INT8"):
        validate_identity(item, runtime)
    runtime["inference"]["scope"]["weight_precision"] = "float16,uint32"
    runtime["inference"]["execution_contract"] = {
        "quantization": {"bits": 8, "group_size": 64, "mode": "affine"},
        "modules": [{"bits": 8, "group_size": 64, "mode": "affine", "scale_dtype": "float16"}],
        "floating_policy": {"dtype": "float16", "quantized_payload_preserved": True}}
    validate_identity(item, runtime)
    assert item.pressure_config["enabled"] is True


def test_optimized_activation_policy_does_not_persist_on_reference(int8_native, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from mlx_one.engine.adapters import native_adapter

    attention = SimpleNamespace(q_proj=object())
    modules = int8_native.model.named_modules()
    int8_native.model.named_modules = lambda: [*modules, ("attention", attention)]
    model_type = type(int8_native.model)
    original = model_type.__call__
    def forward(model, *args, **kwargs):
        assert attention._mlx_one_activation_dtype == "float16"
        return original(model, *args, **kwargs)
    monkeypatch.setattr(model_type, "__call__", forward)
    adapter = native_adapter(int8_native)
    adapter.forward(np.array([[1]]), cache=int8_native.model.make_cache())
    assert not hasattr(attention, "_mlx_one_activation_dtype")
