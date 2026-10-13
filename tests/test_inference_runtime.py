"""Contract, scope, and execution-path regression tests without Metal."""

import asyncio
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.core.cache import KVCache
from mlx_one.core.outputs import ModelOutput
from mlx_one.engine import (
    BatchPlan,
    CacheKind,
    CausalLMRunner,
    EngineRequest,
    LayerCacheSpec,
    LayerStateSpec,
    Phase,
    RequestKind,
    RunnerOutput,
    ScheduledSequence,
    SequenceState,
)
from mlx_one.engine.modes import ModelCapabilityUnsupported, resolve_backend, validate_mode
from mlx_one.engine.reporting import describe_cache
from mlx_one.registry import (
    InferenceFeatureRegistry,
    InferenceFeatureStatus,
    InferenceScope,
    RegistryError,
)
from mlx_one.server.config import ServerConfig
from mlx_one.server.scheduler import GenerationScheduler


@pytest.fixture
def fake_mx(monkeypatch):
    core = ModuleType("mlx.core")
    core.array = np.array
    core.zeros = np.zeros
    core.concatenate = np.concatenate
    core.eval = lambda *args: None
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return core


def make_runner(*, fail=False):
    class Model:
        calls = []

        def make_cache(self):
            return (KVCache(),)

        def __call__(self, ids, *, cache):
            self.calls.append(ids.shape)
            values = np.zeros((ids.shape[0], 1, ids.shape[1], 4), dtype=np.float16)
            cache[0].update(values, values)
            if fail:
                raise RuntimeError("injected execution failure")
            return ModelOutput(np.zeros((ids.shape[0], ids.shape[1], 8)), values, cache)

    return CausalLMRunner(SimpleNamespace(
        model=Model(), tokenizer=SimpleNamespace(encode=lambda _: [1, 2], bos_token_id=1),
    ))


def test_committed_boundary_sampling_and_idempotent_terminal():
    state = SequenceState("request", (1, 2), 2)
    state.transition(Phase.PREFILL)
    state.generated_ids.append(3)
    assert state.computed_len == 0
    state.commit(2)
    state.transition(Phase.DECODE)
    state.commit(1)
    with pytest.raises(ValueError, match="exceeds"):
        state.commit(1)
    state.transition(Phase.CANCELLED)
    state.transition(Phase.CANCELLED)
    with pytest.raises(ValueError, match="transition"):
        state.transition(Phase.DECODE)
    with pytest.raises(ValueError, match="active"):
        state.commit(1)


def test_late_out_of_order_and_old_model_events_do_not_change_state():
    state = SequenceState("r", (1,), 1, model_epoch=2)
    assert state.accept_event(request_id="r", model_epoch=2, event_no=4)
    for request_id, epoch, event_no in (("r", 1, 5), ("r", 2, 3), ("other", 2, 6)):
        assert not state.accept_event(request_id=request_id, model_epoch=epoch, event_no=event_no)
    assert state.computed_len == 0 and state.last_event_no == 4
    state.transition(Phase.CANCELLED)
    assert not state.accept_event(request_id="r", model_epoch=2, event_no=5)


def test_batch_and_output_mapping_validation():
    slot = ScheduledSequence("r", "prefill", 0, 2)
    with pytest.raises(ValueError, match="unique"):
        BatchPlan((slot, slot), 4)
    with pytest.raises(ValueError, match="counts"):
        BatchPlan((slot,), 1)
    with pytest.raises(ValueError, match="exactly one"):
        ScheduledSequence("r", "decode", 0, 2)
    with pytest.raises(ValueError, match="align"):
        RunnerOutput(("r",), (), (1,))


def test_runner_only_commits_evaluated_inputs_and_rejects_stale_plans(fake_mx):
    runner = make_runner()
    state = runner.prepare(EngineRequest(RequestKind.TEXT, "model", {"prompt": "x"}))
    state.transition(Phase.PREFILL)
    plan = BatchPlan((ScheduledSequence(state.request_id, "prefill", 0, 2),), 2)
    result = runner.prefill(plan)
    assert result.committed_token_counts == (2,)
    assert state.computed_len == state.cache.offset == 2
    with pytest.raises(ValueError, match="start position"):
        runner.prefill(plan)
    state.generated_ids.append(3)
    assert state.computed_len == 2
    state.transition(Phase.DECODE)
    runner.decode(BatchPlan((ScheduledSequence(state.request_id, "decode", 2, 1),), 1))
    assert state.computed_len == state.cache.offset == 3
    runner.finalize(state)
    runner.finalize(state)
    assert state.cache.closed


def test_failed_execution_releases_dirty_cache_without_commit(fake_mx):
    runner = make_runner(fail=True)
    state = runner.prepare(EngineRequest(RequestKind.TEXT, "model", {"prompt": "x"}))
    state.transition(Phase.PREFILL)
    with pytest.raises(RuntimeError, match="injected"):
        runner.prefill(BatchPlan((ScheduledSequence(state.request_id, "prefill", 0, 2),), 2))
    assert state.computed_len == 0
    assert state.phase is Phase.FAILED and state.cache.closed
    runner.release()


def test_unknown_layer_capabilities_fail_closed():
    unknown = LayerStateSpec.from_cache_spec(LayerCacheSpec(0, CacheKind.MODEL_DEFINED))
    assert unknown.kind == "unknown"
    assert not any((unknown.can_batch, unknown.can_page, unknown.can_snapshot,
                    unknown.can_quantize, unknown.can_share_prefix))
    dense = LayerStateSpec.from_cache_spec(LayerCacheSpec(0, CacheKind.ATTENTION_KV,
                                                        block_compatible=True))
    assert not dense.can_page
    assert dense.dtype is None
    assert dense.to_cache_spec().kind is CacheKind.ATTENTION_KV


@pytest.mark.parametrize("mode,backend,expected", [
    (None, None, "dense"), (None, "block", "block"),
    ("reference_dense", None, "dense"), ("experimental_block", None, "block"),
])
def test_explicit_mode_resolution(mode, backend, expected):
    assert resolve_backend(mode, backend) == expected
    assert ServerConfig(inference_mode=mode, cache_backend=backend).cache_backend == expected


def test_conflicting_and_unsupported_modes():
    with pytest.raises(ValueError, match="conflicts"):
        ServerConfig(inference_mode="experimental_block", cache_backend="dense")
    with pytest.raises(ModelCapabilityUnsupported):
        resolve_backend("unsupported", None)
    assert resolve_backend("paged", None) == "paged"
    with pytest.raises(ModelCapabilityUnsupported, match="f16"):
        ServerConfig(inference_mode="experimental_block", cache_type_k="bf16")
    plan = SimpleNamespace(architecture="lfm2", block_compatible=False)
    with pytest.raises(ModelCapabilityUnsupported):
        validate_mode(SimpleNamespace(quantization={}), plan, "experimental_block")
    with pytest.raises(ModelCapabilityUnsupported):
        validate_mode(SimpleNamespace(quantization={"bits": 4}),
                      SimpleNamespace(architecture="qwen2", block_compatible=True),
                      "experimental_block")


def scope():
    return InferenceScope("Qwen/test", "a" * 40, "reference_dense", "topology", "bfloat16",
                          "bfloat16", "0.32.2", "M4:32GB:macOS", "source-hash")


def test_native_evidence_is_exact_scoped_checksummed_and_revalidated(tmp_path):
    registry = InferenceFeatureRegistry()
    identity = scope()
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "schema_version": "native-inference-qualification-v3", "backend": "mlx-one-native",
        "scope": asdict(identity), "feature": "dense_execution", "status": "passed",
        "correctness_passed": True,
    }))
    from native_evidence import attach_provenance

    path.write_text(json.dumps(attach_provenance(json.loads(path.read_text()), tmp_path)))
    registry.promote("dense_execution", identity, path, InferenceFeatureStatus.CORRECTNESS_PASSED)
    assert registry.report("dense_execution", identity)["status"] == "correctness_passed"
    assert registry.report("dense_execution", replace(identity, mlx_version="new"))[
        "status"] == "not_implemented"
    with pytest.raises(RegistryError, match="performance"):
        registry.promote("dense_execution", identity, path,
                         InferenceFeatureStatus.PERFORMANCE_QUALIFIED)
    with pytest.raises(RegistryError, match="match"):
        registry.promote("paged_attention", identity, path,
                         InferenceFeatureStatus.CORRECTNESS_PASSED)
    path.write_text("changed")
    assert registry.report("dense_execution", identity)["stale_evidence"]
    assert registry.report("dense_execution", identity)["status"] == "experimental"


def test_mlx_lm_evidence_and_missing_identity_cannot_promote(tmp_path):
    registry = InferenceFeatureRegistry()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"backend": "mlx-lm", "status": "passed"}))
    with pytest.raises(RegistryError):
        registry.promote("dense_execution", scope(), path,
                         InferenceFeatureStatus.CORRECTNESS_PASSED)
    with pytest.raises(RegistryError, match="complete"):
        registry.promote("dense_execution", replace(scope(), revision=None), path,
                         InferenceFeatureStatus.CORRECTNESS_PASSED)


def test_dtype_reporting_does_not_materialize_block_cache():
    class BlockCache:
        _blocks = [SimpleNamespace(block=SimpleNamespace(
            state=(np.zeros((1,), dtype=np.float16),))) ]
        nbytes = 2
        allocated_nbytes = 2
        offset = 1

        def state(self):
            raise AssertionError("reporting must not materialize")

    plan = SimpleNamespace(state_specs=(LayerStateSpec(0, "full_kv"),))
    assert describe_cache((BlockCache(),), plan)[0]["actual_dtypes"] == ("float16",)


def test_server_slots_prove_separate_b1_model_calls(fake_mx):
    runner = make_runner()
    scheduler = GenerationScheduler(parallel=2)
    states = []

    def factory(index):
        def run(cancel):
            state = runner.prepare(EngineRequest(RequestKind.TEXT, "model", {"prompt": str(index)}))
            states.append(state)
            state.transition(Phase.PREFILL)
            for position in range(2):
                yield runner.prefill(BatchPlan((ScheduledSequence(
                    state.request_id, "prefill", position, 1
                ),), 1))
            state.transition(Phase.FINISHED)
            runner.finalize(state)
        return run

    async def collect():
        async def read(stream):
            return [item async for item in stream]
        return await asyncio.gather(*(read(scheduler.schedule(factory(i))) for i in range(2)))

    try:
        assert len(asyncio.run(collect())) == 2
        assert runner.bundle.model.calls == [(1, 1)] * 4
        assert all(state.cache.closed for state in states)
    finally:
        scheduler.close()


def test_forced_token_parity_hybrid_carries_and_seeded_replay_with_fake_backend(fake_mx):
    from mlx_one.core.cache import ConvCache
    from mlx_one.engine.parity import forced_token_parity
    from mlx_one.text import LoadedTextModel

    for name in ("abs", "max", "mean", "all", "argmax", "float32"):
        setattr(fake_mx, name, getattr(np, name))
    fake_mx.allclose = lambda *args, **kwargs: np.bool_(np.allclose(*args, **kwargs))
    fake_mx.softmax = lambda values, axis: np.exp(values - np.max(values)) / np.sum(
        np.exp(values - np.max(values)), axis=axis)

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 99

        def encode(self, value):
            return [ord(c) % 7 + 1 for c in value]

        def decode(self, ids):
            return "".join(chr(65 + i) for i in ids)

    class Hybrid:
        config = SimpleNamespace(num_key_value_heads=1, num_attention_heads=1, hidden_size=1)

        def make_cache(self):
            return (KVCache(), ConvCache(3))

        def __call__(self, ids, *, cache):
            values = ids.astype(np.float32)[:, None, :, None]
            keys, _ = cache[0].update(values, values)
            carry = cache[1].update(ids[:, :, None].astype(np.float32))
            sums = np.cumsum(keys[:, 0, :, 0], axis=1)[:, -ids.shape[1]:]
            windows = np.stack([carry[:, i:i + 3, 0].sum(axis=1)
                                for i in range(ids.shape[1])], axis=1)
            logits = np.zeros((*ids.shape, 8), dtype=np.float32)
            for i in range(ids.shape[1]):
                logits[:, i, int((sums[0, i] + windows[0, i]) % 8)] = 3
            return ModelOutput(logits, values, cache)

    bundle = LoadedTextModel(Hybrid(), Tokenizer(), Path("."), "fake/model",
                             "a" * 40, architecture="lfm2", context_length=4096)
    result = forced_token_parity(bundle, prompt="abc", count=8)
    assert result["status"] == "passed", result
    assert result["seeded_rng_isolation"] and result["cancelled_request_isolated"]
    assert all(result["carry_state_matches"])


def test_model_reload_closes_old_cache_and_invalidates_old_handle(fake_mx, tmp_path):
    from mlx_one.engine.cache_errors import CacheOwnershipError
    from mlx_one.server.generation_engine import GenerationEngine

    def bundle(model_id):
        model = make_runner().bundle.model
        model.config = SimpleNamespace(num_key_value_heads=1, num_attention_heads=1, hidden_size=4)
        return SimpleNamespace(model=model, model_id=model_id, architecture="qwen2",
                               revision="a" * 40, context_length=128, path=tmp_path,
                               tokenizer=SimpleNamespace(), quantization={})

    first, second = bundle("first"), bundle("second")
    engine = GenerationEngine(SimpleNamespace(current_model=lambda: first))
    old_runtime = engine._cache_manager_for(first)
    handle = old_runtime.create_handle("old-request")
    engine._cache_manager_for(second)
    assert handle.closed and old_runtime.stats().active_handles == 0
    old_runtime.release(handle)
    with pytest.raises(CacheOwnershipError):
        old_runtime.reserve(handle, tokens=1)
    assert engine.cache_stats()["active_handles"] == 0
    engine.close()


def test_http_runtime_and_observed_generation_use_actual_cache_precision(fake_mx, tmp_path):
    from fastapi.testclient import TestClient

    from mlx_one.server.app import create_app
    from mlx_one.server.generation_engine import GenerationEngine
    from mlx_one.text import LoadedTextModel
    from mlx_one.text.chat import ChatTemplate

    fake_mx.float16 = np.float16
    fake_mx.bfloat16 = np.float32
    fake_mx.argmax = np.argmax
    runner = make_runner()
    model = runner.bundle.model
    model.config = SimpleNamespace(num_key_value_heads=1, num_attention_heads=1, hidden_size=4)
    model.parameters = lambda: {"weight": np.zeros((1,), dtype=np.float16)}
    tokenizer = SimpleNamespace(encode=lambda _: [1, 2], decode=lambda ids: "a" * len(ids),
                                bos_token_id=1, eos_token_id=99)
    bundle = LoadedTextModel(model, tokenizer, tmp_path, "fake/model", "a" * 40,
                             architecture="qwen2", context_length=128,
                             chat_template=ChatTemplate("qwen2", None, tokenizer))
    manager = SimpleNamespace(current_model=lambda: bundle,
                              model_info=lambda: SimpleNamespace(
                                  id="fake/model", context_length=128, to_dict=lambda: {}),
                              unload=lambda: None)
    engine = GenerationEngine(manager, inference_mode="reference_dense",
                              collect_execution_metrics=True)
    config = ServerConfig(inference_mode="reference_dense", warmup=False, cache_prompt=False)
    with TestClient(create_app(manager, engine, config=config, ui_dir=tmp_path)) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "fake/model", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3, "temperature": 0,
        })
        assert response.status_code == 200, response.text
        metrics = response.json()["mlx"]
        assert len(metrics["model_token_times_ms"]) == 3
        assert metrics["model_forward_batch_sizes"] == [1, 1, 1]
        assert metrics["queue_ms"] >= 0 and metrics["prefill_ms"] >= 0
        runtime = client.get("/v1/runtime").json()["inference"]
        assert runtime["scope"]["cache_precision"] == "float16"
        assert runtime["configured_cache_precision"] == {
            "key_type": "f16", "value_type": "f16"
        }
        assert not any(runtime["capabilities"].values())
        assert runtime["runner_qualification"]["equal_offset_batched_decode"][
            "status"] == "experimental"
        assert not engine.cache_stats()["active_handles"]


def test_native_qualification_rechecks_backing_artifacts(tmp_path):
    import hashlib

    from mlx_one.inference_baseline_worker import qualification_evidence

    scope = InferenceScope("model", "a" * 40, "experimental_dense_batch", "topology",
                           "float16", "float16", "0.32.2", "M4", "source")
    payload = tmp_path / "parity.json"
    payload.write_text('{"status":"passed"}')
    qualification_evidence(tmp_path, feature="batched_dense_decode", scope=asdict(scope),
                           parity={"status": "passed"})
    result = tmp_path / "evidence-batched_dense_decode.json"
    from native_evidence import attach_provenance

    result.write_text(json.dumps(attach_provenance(json.loads(result.read_text()), tmp_path)))
    artifact = json.loads(result.read_text())["artifacts"][0]
    assert artifact["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    registry = InferenceFeatureRegistry()
    registry.promote("batched_dense_decode", scope, result,
                     InferenceFeatureStatus.CORRECTNESS_PASSED)
    payload.write_text('{"status":"failed"}')
    assert registry.report("batched_dense_decode", scope)["stale_evidence"]
    with pytest.raises(RegistryError, match="checksum"):
        registry.promote("batched_dense_decode", scope, result,
                         InferenceFeatureStatus.CORRECTNESS_PASSED)
