"""Backend-free direct-page, hybrid, pressure and evidence differential tests."""

import asyncio
import hashlib
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one import BatchingConfig, CacheConfig, Engine
from mlx_one.core.cache import ConvCache, KVCache
from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.batching import ForwardRequest
from mlx_one.engine.cache_errors import (
    CacheCapacityError,
    CacheCorruptionError,
    KVQuantizationUnsupported,
)
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.components import (
    RecurrentComponent,
    SlidingKVComponent,
    load_snapshot,
    save_snapshot,
)
from mlx_one.engine.hybrid import HybridRunner
from mlx_one.engine.paged import KERNEL_VERSION, PagedRunner, dequantize_page, quantize_page
from mlx_one.engine.paged_attention import kernel_source, numpy_paged_attention, page_bindings
from mlx_one.engine.paged_attention import paged_attention as metal_dispatch
from mlx_one.engine.pressure import (
    MemoryPressureConfig,
    MemoryPressureController,
    PressureTelemetry,
    activation_forecast,
)
from mlx_one.engine.selection import (
    compatibility_digest,
    select_auto_mode,
    validate_context_admission,
    validate_execution_envelope,
)


@pytest.fixture
def advanced(native, monkeypatch):
    import mlx.core as mx

    for name in ("pad", "stack", "min", "max", "maximum", "round", "clip", "uint8",
                 "float32", "int32", "where", "repeat", "exp", "abs", "mean", "all"):
        monkeypatch.setattr(mx, name, getattr(np, name), raising=False)
    def softmax(x, axis=-1, precise=True):
        y = np.exp(x.astype(np.float32) - x.max(axis=axis, keepdims=True))
        return (y / y.sum(axis=axis, keepdims=True)).astype(x.dtype)
    monkeypatch.setattr(mx, "softmax", softmax, raising=False)
    monkeypatch.setattr("mlx_one.engine.paged_attention.paged_attention", numpy_paged_attention)

    class Model:
        def __init__(self, hybrid=False):
            self.hybrid = hybrid
            self.config = SimpleNamespace(model_type="lfm2" if hybrid else "qwen2",
                                          num_hidden_layers=2, num_key_value_heads=1,
                                          num_attention_heads=2, hidden_size=8, head_dim=4,
                                          max_position_embeddings=131072, rope_theta=10000,
                                          layer_types=("conv", "full_attention") if hybrid else (),
                                          conv_L_cache=3, vocab_size=64, intermediate_size=16)
            self.weights = np.ones(2, dtype=np.float16)
            self.calls, self.fail, self.pause = [], False, None

        def parameters(self):
            return {"weights": self.weights}

        def set_dtype(self, dtype):
            self.weights = self.weights.astype(dtype)

        def make_cache(self):
            return (ConvCache(3), KVCache()) if self.hybrid else (KVCache(), KVCache())

        def __call__(self, ids, *, cache, attention_mask=None, position_ids=None):
            if self.pause:
                self.pause()
            self.calls.append(ids.shape)
            batch, width = ids.shape
            offset = cache[-1].offset
            if position_ids is None:
                position_ids = np.broadcast_to(np.arange(offset, offset + width), ids.shape)
            data = np.stack((ids / 32, position_ids / 64,
                             ids / 48, np.ones_like(ids) / 8), axis=-1).astype(np.float16)[:, None]
            queries = np.repeat(data, 2, axis=1)
            carry_effect = 0
            for layer_index, layer in enumerate(cache):
                if self.hybrid and layer_index == 0:
                    gated = np.tile(data[:, 0], (1, 1, 2))
                    combined = layer.update(gated)
                    carry_effect = sum(combined[:, i:i + width].astype(np.float32)
                                       for i in range(3)).sum(axis=-1)
                    continue
                if callable(getattr(layer, "attend", None)):
                    attended = layer.attend(queries, data, data, scale=.5,
                                            position_ids=position_ids)
                else:
                    old = layer.offset
                    k, v = layer.update(data, data)
                    k, v = np.repeat(k, 2, axis=1), np.repeat(v, 2, axis=1)
                    scores = (queries @ k.transpose(0, 1, 3, 2)) * .5
                    causal = np.arange(k.shape[2])[None, :] <= (
                        old + np.arange(width)[:, None])
                    valid = causal[None, None]
                    if attention_mask is not None and attention_mask.shape[1] == k.shape[2]:
                        valid = valid & attention_mask[:, None, None, :].astype(bool)
                    scores = np.where(valid, scores, -np.inf)
                    attended = softmax(scores) @ v
            if self.fail:
                raise RuntimeError("injected staged model failure")
            target = (attended.astype(np.float32).sum(axis=(1, 3)) * 5 + carry_effect) % 60
            logits = -np.abs(np.arange(64)[None, None] - target[..., None]).astype(np.float16)
            return ModelOutput(logits, data, cache)

    def build(hybrid=False):
        model = Model(hybrid)
        return replace(native, model=model, architecture="lfm2" if hybrid else "qwen2",
                       context_length=131072)
    return build


def _requests(bundle, runtime, prompts):
    result = []
    for i, ids in enumerate(prompts):
        handle = runtime.create_handle(str(i))
        result.append(ForwardRequest(bundle, tuple(ids), handle.bundle.entries, "prefill",
                                     tuple(ids), runtime, handle, request_id=str(i)))
    return result


@pytest.mark.parametrize("page_width", [16, 32, 64, 128])
def test_direct_pages_ragged_gqa_match_dense_and_never_materialize(advanced, page_width):
    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(backend="paged", block_size_tokens=page_width))
    runner = PagedRunner()
    requests = _requests(bundle, runtime, [range(1, page_width + 2), (4, 5, 6)])
    independent = [bundle.model.make_cache() for _ in requests]
    try:
        result = runner.execute(tuple(requests))
        assert bundle.model.calls[-1][0] == 2
        for i, r in enumerate(requests):
            expected = bundle.model(np.array([r.token_ids]), cache=independent[i]).logits[:, -1:]
            np.testing.assert_allclose(result.logits[i], expected, atol=.03, rtol=.01)
        for step in range(3):
            for r in requests:
                r.phase, r.token_ids = "decode", (step + 9,)
            result = runner.execute(tuple(requests))
            for i, r in enumerate(requests):
                expected = bundle.model(np.array([r.token_ids]),
                                        cache=independent[i]).logits[:, -1:]
                np.testing.assert_allclose(result.logits[i], expected, atol=.03, rtol=.01)
        assert runner.last_trace["history_materializations"] == 0
        assert runner.last_trace["direct_attention_calls"] == 2
        assert requests[0].cache[0].offset == page_width + 4
    finally:
        runtime.close()
    assert runtime.stats().physical_bytes == 0


def test_paged_forward_failure_and_cancel_keep_committed_state(advanced):
    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(backend="paged"))
    requests = _requests(bundle, runtime, [(2, 3), (4,)])
    runner = PagedRunner()
    runner.execute(tuple(requests))
    for r in requests:
        r.phase, r.token_ids = "decode", (9,)
    before = [r.cache[0].offset for r in requests]
    bundle.model.fail = True
    with pytest.raises(RuntimeError, match="staged"):
        runner.execute(tuple(requests))
    assert [r.cache[0].offset for r in requests] == before
    bundle.model.fail = False
    requests[0].cancel = threading.Event()
    bundle.model.pause = requests[0].cancel.set
    runner.execute(tuple(requests))
    assert [r.cache[0].offset for r in requests] == [before[0], before[1] + 1]
    runtime.close()


@pytest.mark.parametrize("bits", [4, 8])
def test_packed_quantization_metadata_and_direct_attention(advanced, bits):
    bundle = advanced()
    rng = np.random.default_rng(12)
    values = rng.normal(size=(1, 2, 16, 37)).astype(np.float16)
    packed = quantize_page(values, bits)
    restored = dequantize_page(packed, bits, 37)
    np.testing.assert_allclose(restored, values, atol=.24 if bits == 4 else .02, rtol=.02)
    assert sum(v.nbytes for v in packed) < values.nbytes
    runtime = CacheManager(bundle, CacheConfig(backend="paged", block_size_tokens=16))
    with pytest.raises(KVQuantizationUnsupported, match="quality evidence"):
        runtime.create_handle("public", key_type=f"q{bits}_0", value_type=f"q{bits}_0")
    runtime.harness_only = True
    handle = runtime.create_handle("fixture", key_type=f"q{bits}_0", value_type=f"q{bits}_0")
    request = ForwardRequest(bundle, tuple(range(20)), handle.bundle.entries, "prefill",
                             tuple(range(20)), runtime, handle, request_id="quant")
    output = PagedRunner().execute((request,))
    assert np.isfinite(output.logits[0]).all()
    assert handle.bundle.entries[0]._blocks[0].block.layout.quantization_bits == bits
    assert handle.bundle.entries[0]._blocks[-1].block.layout.quantization_bits is None
    runtime.close()


def test_segment_identity_validation_and_shared_page_lifetime(advanced):
    runtime = CacheManager(advanced(), CacheConfig(backend="paged", block_size_tokens=16))
    cache = runtime.create_handle("a").bundle.entries[0]
    values = np.ones((1, 1, 17, 4), dtype=np.float16)
    cache.append(values, values)
    branch = cache.clone()
    assert branch._blocks[0].block_id == cache._blocks[0].block_id
    assert branch._blocks[-1].block_id != cache._blocks[-1].block_id
    runtime.close()
    with pytest.raises(CacheCorruptionError):
        page_bindings((branch,))
    # The stale branch cannot resolve any storage after epoch invalidation.


def test_kernel_bindings_are_bounded_and_template_arguments_are_integral():
    for formats in ((None,) * 8, (4,) * 3, (8, None, 4)):
        names, source = kernel_source(formats, 128)
        assert len(names) <= 31
        assert "threadgroup_barrier" in source
        assert "scale[0]" in source


def test_hybrid_ragged_prefill_carries_and_decode_match_independent(advanced):
    bundle = advanced(hybrid=True)
    runtime = CacheManager(bundle, CacheConfig(backend="hybrid"))
    requests = _requests(bundle, runtime, [(1, 2, 3, 4), (8, 9)])
    reference = [bundle.model.make_cache() for _ in requests]
    runner = HybridRunner()
    try:
        result = runner.execute(tuple(requests))
        for i, r in enumerate(requests):
            expected = bundle.model(np.array([r.token_ids]), cache=reference[i]).logits[:, -1:]
            np.testing.assert_allclose(result.logits[i], expected, atol=.03, rtol=.01)
            assert r.cache[0].offset == len(r.token_ids)
            np.testing.assert_array_equal(r.cache[0].values, reference[i][0].values)
        for step in range(4):
            for r in requests:
                r.phase, r.token_ids = "decode", (step + 6,)
            result = runner.execute(tuple(requests))
            for i, r in enumerate(requests):
                expected = bundle.model(np.array([r.token_ids]), cache=reference[i]).logits[:, -1:]
                np.testing.assert_allclose(result.logits[i], expected, atol=.03, rtol=.01)
        assert runner.last_trace["batch_size"] == 2
    finally:
        runtime.close()


@pytest.mark.parametrize("hybrid", [False, True])
def test_new_python_modes_batch_and_restore_principal_prefix(advanced, hybrid):
    bundle = advanced(hybrid)
    mode = "experimental_hybrid_batch" if hybrid else "paged"
    disabled = MemoryPressureConfig(enabled=False)
    with Engine(bundle, inference_mode=mode, cache_namespace="principal:test",
                memory_pressure_config=disabled,
                batching_config=BatchingConfig(max_sequences=2, prefill_chunk_size=16)) as engine:
        prompt = "A repeated prompt with a sealed page. " * 2
        a, b = engine.submit(prompt, max_tokens=5), engine.submit("short", max_tokens=5)
        outputs = a.result(3), b.result(3)
        traces = engine.runtime_stats()["scheduler"]["forward_traces"]
        assert any(t["batch_size"] == 2 for t in traces)
        repeated = engine.generate(prompt, max_tokens=5)
        assert repeated.text == outputs[0].text
        assert engine.runtime_stats()["cache_stats"]["prefix_hits"] > 0
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0


def test_hybrid_snapshot_round_trip_checksum_and_exact_boundary(advanced, tmp_path):
    bundle = advanced(True)
    runtime = CacheManager(bundle, CacheConfig(backend="hybrid", block_size_tokens=16))
    requests = _requests(bundle, runtime, [range(16)])
    HybridRunner().execute(tuple(requests))
    identity = {"revision": bundle.revision, "topology": runtime.plan.topology_digest}
    saved = save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
    restored = load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
    assert restored.offset == 16
    np.testing.assert_array_equal(restored.entries[0].values, requests[0].cache[0].values)
    with pytest.raises(ValueError, match="identity"):
        load_snapshot(tmp_path, identity={"revision": "wrong"},
                      expected_specs=runtime.plan.state_specs)
    (tmp_path / saved["payload"]).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
    restored.release()
    runtime.close()


def test_recurrent_contract_is_isolated_and_math_remains_unavailable(advanced):
    component = RecurrentComponent(np.ones(4, dtype=np.float16), 7)
    clone = component.clone()
    clone.values[0] = 9
    assert component.values[0] == 1
    with pytest.raises(NotImplementedError):
        clone.update(None)
    clone.release()
    clone.release()
    assert component.offset == 7


def test_pressure_hysteresis_reservations_and_missing_telemetry():
    clock = [0.0]
    samples = [PressureTelemetry(rss_bytes=v, installed_bytes=1000)
               for v in (100, 680, 800, 100, 100)]
    controller = MemoryPressureController(telemetry=lambda: samples.pop(0), clock=lambda: clock[0])
    assert controller.poll() == "normal"
    assert controller.available(300, 100) == 350  # owner/RSS overlap, reservations do not
    for state in ("warning", "critical", "critical", "normal"):
        clock[0] += 1
        assert controller.poll() == state
    with pytest.raises(CacheCapacityError, match="ceiling"):
        controller.admission(751)
    with pytest.raises(CacheCapacityError, match="transient"):
        controller.forward(700)
    unknown = MemoryPressureController(telemetry=PressureTelemetry)
    assert unknown.poll() == "unknown"
    assert unknown.available() is None


def test_context_preflight_and_auto_reference_fallback(advanced):
    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(backend="paged"))
    with pytest.raises(ValueError, match="qualification"):
        validate_context_admission(runtime, 128000, 128)
    runtime.harness_only = True
    validate_context_admission(runtime, 128000, 128)
    with pytest.raises(ValueError, match="checkpoint context"):
        validate_context_admission(runtime, 131072, 1)
    runtime.close()
    with Engine(bundle, inference_mode="auto", memory_pressure_config=MemoryPressureConfig(
            enabled=False)) as engine:
        assert engine.runtime_stats()["selection"]["resolved_mode"] == "reference_dense"
        assert engine.generate("abc", max_tokens=2).generation_tokens == 2
    estimate = activation_forecast(bundle.model.config, batch=2, chunk=16, history=32)
    assert estimate["bytes"] > 0 and estimate["measured"] is False


@pytest.mark.parametrize("fail_at", range(1, 7))
def test_every_paged_allocation_layer_rolls_back(advanced, monkeypatch, fail_at):
    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(backend="paged", block_size_tokens=16))
    requests = _requests(bundle, runtime, [(1, 2)])
    runner = PagedRunner()
    runner.execute(tuple(requests))
    before = [c.offset for c in requests[0].cache]
    live = runtime.block_pool.stats()["block_live"]
    original, calls = runtime.block_pool.allocate_ref, [0]
    def allocate(*args, **kwargs):
        calls[0] += 1
        if calls[0] == fail_at:
            raise MemoryError("injected page allocation")
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime.block_pool, "allocate_ref", allocate)
    requests[0].token_ids = tuple(range(32))
    with pytest.raises(MemoryError, match="allocation"):
        runner.execute(tuple(requests))
    assert [c.offset for c in requests[0].cache] == before
    assert runtime.block_pool.stats()["block_live"] == live
    assert runtime.stats().reserved_bytes == 0
    assert runtime.stats().prefix_entries == 0
    runtime.close()
    runtime.close()


def test_hybrid_failure_cancel_and_sliding_components(advanced):
    bundle = advanced(True)
    runtime = CacheManager(bundle, CacheConfig(backend="hybrid"))
    requests = _requests(bundle, runtime, [(1, 2, 3), (4, 5)])
    runner = HybridRunner()
    runner.execute(tuple(requests))
    carries = [r.cache[0].values.copy() for r in requests]
    offsets = [r.cache[0].offset for r in requests]
    for request in requests:
        request.phase, request.token_ids = "decode", (8,)
    bundle.model.fail = True
    with pytest.raises(RuntimeError, match="staged"):
        runner.execute(tuple(requests))
    assert [r.cache[0].offset for r in requests] == offsets
    for r, carry in zip(requests, carries, strict=True):
        np.testing.assert_array_equal(r.cache[0].values, carry)
    bundle.model.fail = False
    requests[0].cancel = threading.Event()
    bundle.model.pause = requests[0].cancel.set
    runner.execute(tuple(requests))
    assert [r.cache[0].offset for r in requests] == [offsets[0], offsets[1] + 1]
    np.testing.assert_array_equal(requests[0].cache[0].values, carries[0])
    runtime.close()
    component = SlidingKVComponent(3)
    data = np.arange(20, dtype=np.float16).reshape(1, 1, 5, 4)
    component.update(data, data)
    assert component.offset == 5 and component.window_start == 2
    np.testing.assert_array_equal(component.keys, data[:, :, -3:])
    fork = component.clone()
    fork.keys[:] = 99
    assert component.keys[0, 0, 0, 0] == 8
    with pytest.raises(ValueError, match="exact checkpoint"):
        component.trim(1)


@pytest.mark.parametrize("hybrid", [False, True])
def test_complete_parity_profiles_run_with_cpu_oracle(advanced, hybrid):
    from mlx_one.engine.qualification import execution_parity

    bundle = advanced(hybrid)
    result = execution_parity(bundle, mode="experimental_hybrid_batch" if hybrid else "paged",
                              prompt="A sealed page and shared prompt. " * 2,
                              count=6, batch_size=2, page_tokens=16, chunk_size=7,
                              atol=.05, rtol=.02)
    assert result["status"] == "passed", result
    assert result["component_comparisons"]
    assert result["prefix_isolation_passed"]
    assert result["cancelled_request_isolated"]
    assert result["cleanup"]["active_handles"] == 0


@pytest.mark.parametrize("cache_type", ["f16", "q4_0"])
def test_auto_exact_evidence_expiry_and_envelope(advanced, tmp_path, monkeypatch, cache_type):
    from mlx_one.engine.reporting import native_report
    from mlx_one.registry import InferenceFeatureRegistry, InferenceFeatureStatus, InferenceScope

    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(key_type=cache_type, value_type=cache_type))
    runtime.mode = "paged"
    config = BatchingConfig(max_sequences=2)
    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
    from mlx_one.inference_qualification import workload_envelope

    envelope = workload_envelope(BaselineProfile(*CHECKPOINTS["qwen"], "paged", slots=2,
        cache_type=cache_type, kv_bits=4 if cache_type == "q4_0" else None,
        ttft_budget_ms=1000, decode_gap_budget_ms=100))
    runtime.batching_config = config
    runtime.pressure = MemoryPressureController()
    runtime.qualification_profile = envelope
    scope = native_report(bundle, runtime, mode="paged", hardware="fixture-host",
                          source_revision="fixture-source")["scope"]
    scope["cache_precision"] = "float16,uint8" if cache_type == "q4_0" else "float16"
    feature = "quantized_paged_kv" if cache_type == "q4_0" else "paged_attention"
    artifact = tmp_path / "measured.json"
    artifact.write_text('{"fixture": true}')
    proof = {"schema_version": "native-inference-qualification-v2", "backend": "mlx-one-native",
             "feature": feature, "scope": scope, "status": "passed",
             "correctness_passed": True, "performance_qualified": True,
             "compatibility_digest": compatibility_digest(runtime),
             "workload_envelope": envelope, "kernel_version": KERNEL_VERSION,
             "quality_passed": True, "kernel_correctness_passed": True,
             "quality_kind": "native-held-out-quality-v1", "quality_coverage_passed": True,
             "artifacts": [{"uri": str(artifact), "kind": "cpu-fixture",
                            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    registry = InferenceFeatureRegistry()
    registry.promote(feature, InferenceScope(**scope), path,
                     InferenceFeatureStatus.PERFORMANCE_QUALIFIED)
    def select(profile=envelope):
        return select_auto_mode(bundle, runtime, registry, profile,
                                hardware="fixture-host", source="fixture-source")
    assert select()["resolved_mode"] == "paged"
    assert select({**envelope, "slots": 4})["resolved_mode"] == "reference_dense"
    assert select({**envelope, "cache_type": "bf16"})["resolved_mode"] == "reference_dense"
    validate_execution_envelope(runtime, config, required=True)
    if cache_type == "q4_0":
        from mlx_one.engine.selection import qualification_matches

        monkeypatch.setattr("mlx_one.engine.selection.host_identity", lambda: "fixture-host")
        monkeypatch.setattr("mlx_one.engine.selection.source_identity", lambda: "fixture-source")
        runtime.qualification_registry, runtime.batching_config = registry, config
        assert qualification_matches(runtime, feature)
    with pytest.raises(ValueError, match="envelope"):
        validate_context_admission(runtime, 2049, 1)
    with pytest.raises(ValueError, match="envelope"):
        validate_execution_envelope(runtime, replace(config, max_sequences=4), required=True)
    artifact.write_text("stale")
    assert select()["resolved_mode"] == "reference_dense"
    runtime.close()


@pytest.mark.parametrize("hybrid", [False, True])
def test_http_new_modes_batch_sse_and_principal_isolation(advanced, hybrid):
    import httpx

    from mlx_one.engine.actor import InferenceActor
    from mlx_one.engine.dispatch import DispatchRunner
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine

    class Template:
        template = "plain"
        def render(self, messages, **kwargs):
            return " ".join(m["content"] for m in messages)
    bundle = replace(advanced(hybrid), chat_template=Template())
    info = SimpleNamespace(id=bundle.model_id, context_length=bundle.context_length,
                           to_dict=lambda: {"id": bundle.model_id},
                           to_openai=lambda: {"id": bundle.model_id})
    manager = SimpleNamespace(current_model=lambda: bundle, model_info=lambda: info)
    mode = "experimental_hybrid_batch" if hybrid else "paged"
    actor = InferenceActor(BatchingConfig(max_sequences=2, prefill_chunk_size=16),
                           runner=DispatchRunner())
    engine = GenerationEngine(manager, inference_mode=mode, prefill_chunk_size=16,
                              memory_pressure_config=MemoryPressureConfig(enabled=False),
                              collect_execution_metrics=True)
    app = create_app(manager, engine, scheduler=actor, config=ServerConfig(
        inference_mode=mode, parallel=2, cache_reuse=1, warmup=False, timeout=5,
        api_keys=("alice", "bob")))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            async def one(key="alice"):
                response = await client.post("/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"}, json={"model": bundle.model_id,
                    "messages": [{"role": "user", "content": "Same shared document. " * 3}],
                    "stream": True, "max_tokens": 8, "temperature": 0})
                assert response.status_code == 200, response.text
                assert "data: [DONE]" in response.text
                return response.text.replace(" ", "")
            await asyncio.gather(one(), one())
            assert '"prompt_cache_hit":true' in await one()
            assert '"prompt_cache_hit":false' in await one("bob")
            runtime = (await client.get("/v1/runtime", headers={
                "Authorization": "Bearer alice"})).json()
            assert runtime["inference"]["engine_mode"] == mode
            assert any(t["batch_size"] == 2 and t["phase"] == "decode"
                       for t in runtime["scheduler"]["forward_traces"])
            assert "alice" not in str(runtime["cache"])
    try:
        asyncio.run(run())
    finally:
        actor.close(finalizer=engine.close)


def test_cache_aware_admission_guards_deadlines_age_and_unfit_candidates(monkeypatch):
    from collections import deque

    from mlx_one.engine.actor import InferenceActor, _Job

    actor = object.__new__(InferenceActor)
    actor.config = BatchingConfig(max_sequences=1, admission_policy="cache_aware")
    actor._condition, actor._active = threading.Condition(), []
    monkeypatch.setattr("mlx_one.engine.actor.time.monotonic", lambda: 100.0)
    def job(created, deadline, prefix):
        result = _Job(lambda: iter(()), threading.Event())
        result.created_at, result.deadline = created, deadline
        result.prefix_preview = lambda: prefix
        return result
    urgent, aged = job(99.9, 100.2, 0), job(97, 200, 0)
    locality, smaller = job(99.9, 200, 128), job(99.9, 101, 16)
    actor._pending = deque([smaller, locality, aged, urgent])
    for expected in (urgent, aged, locality, smaller):
        actor._admit()
        assert actor._active == [expected]
        actor._active.clear()
    locality.retry_at = 101
    actor._pending = deque([locality, smaller])
    actor._admit()
    assert actor._active == [smaller] and list(actor._pending) == [locality]


def test_partitioned_dispatch_mask_gqa_and_partial_pages_match_dense(advanced, monkeypatch):
    from mlx_one.engine.paged import PagedKVCache

    bundle = advanced()
    runtime = CacheManager(bundle, CacheConfig(backend="paged", block_size_tokens=16))
    rng = np.random.default_rng(12)
    rows = []
    histories = []
    for length in (259, 37):
        keys = rng.normal(size=(1, 1, length, 4)).astype(np.float16)
        values = rng.normal(size=keys.shape).astype(np.float16)
        cache = PagedKVCache(runtime.block_pool)
        cache.append(keys, values)
        rows.append(cache)
        histories.append((keys, values))
    query = rng.normal(size=(2, 2, 3, 4)).astype(np.float16)
    positions = ((256, 257, 258), (36, 37, 38))
    widths = (3, 1)
    launches = []
    def kernel(formats, dim, row_count):
        def execute(*, inputs, **kwargs):
            launches.append(len(formats))
            q, segment_ids, pos, real, lengths, scale, *remaining = inputs
            tables, arrays = remaining[:row_count], remaining[row_count:]
            partial = np.zeros((*q.shape[:-1], dim + 2), dtype=np.float32)
            partial[..., dim] = -np.inf
            for row in range(q.shape[0]):
                for head in range(q.shape[1]):
                    for qi in range(real[row]):
                        scores, vals = [], []
                        for i in range(len(formats)):
                            for sid, slot, start, count in tables[row]:
                                if sid != segment_ids[i]:
                                    continue
                                for t in range(min(count, lengths[row] - start)):
                                    if start + t <= pos[row, qi]:
                                        scores.append(float(q[row, head, qi].astype(np.float32)
                                                            @ arrays[i * 2][slot, 0, t]) * scale[0])
                                        vals.append(arrays[i * 2 + 1][slot, 0, t])
                        if scores:
                            maximum = max(scores)
                            weights = np.exp(np.array(scores) - maximum)
                            partial[row, head, qi, :dim] = weights @ np.array(vals)
                            partial[row, head, qi, dim:] = (maximum, sum(weights))
            return [partial]
        return execute
    monkeypatch.setattr("mlx_one.engine.paged_attention._kernel", kernel)
    # This exercises the production partition merger with a CPU launch fixture.
    import mlx.core as mx
    monkeypatch.setattr(mx, "fast", SimpleNamespace(metal_kernel=lambda **kwargs: None),
                        raising=False)
    try:
        output = metal_dispatch(query, rows, positions, widths, scale=.5)
        expected = numpy_paged_attention(query, rows, positions, widths, scale=.5)
        np.testing.assert_allclose(output, expected, atol=.002, rtol=.002)
        assert len(launches) == 2 and max(launches) <= 8
        assert np.all(output[1, :, 1:] == 0)
    finally:
        for row in rows:
            row.reset()
        runtime.close()


def test_snapshot_shape_position_and_dtype_fail_before_adoption(advanced, tmp_path):
    runtime = CacheManager(advanced(True), CacheConfig(backend="hybrid"))
    requests = _requests(runtime.bundle, runtime, [range(32)])
    HybridRunner().execute(tuple(requests))
    identity = {"topology": runtime.plan.topology_digest}
    metadata = save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
    path = tmp_path / "state.json"
    for layer, field, value in ((0, "offset", 31), (0, "kernel_size", 7),
                                (1, "kind", "recurrent")):
        changed = json.loads(json.dumps(metadata))
        changed["layers"][layer][field] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
        assert requests[0].handle.bundle.offset == 32
    runtime.close()


def test_live_telemetry_samples_off_actor_and_closes_idempotently():
    sampled = threading.Event()
    threads = []
    def telemetry():
        threads.append(threading.current_thread().name)
        if threads[-1] == "mlx-one-pressure":
            sampled.set()
        return PressureTelemetry(rss_bytes=100, installed_bytes=1000)
    controller = MemoryPressureController(MemoryPressureConfig(sample_interval_s=.01),
                                           telemetry=telemetry, clock=lambda: 0)
    try:
        controller.start()
        assert sampled.wait(.5)
        for _ in range(20):
            assert controller.poll() == "normal"
        assert threads.count(threading.current_thread().name) == 1
    finally:
        controller.close()
        controller.close()
    assert controller._sampler is None


def test_activation_calibration_requires_exact_dimensions_identity_and_freshness(
        advanced, tmp_path, monkeypatch):
    from mlx_one.engine.pressure import execution_forecast
    from mlx_one.engine.reporting import native_report
    from mlx_one.registry import InferenceFeatureRegistry, InferenceFeatureStatus, InferenceScope

    monkeypatch.setattr("mlx_one.engine.selection.host_identity", lambda: "fixture-host")
    monkeypatch.setattr("mlx_one.engine.selection.source_identity", lambda: "fixture-source")
    runtime = CacheManager(advanced(), CacheConfig(backend="paged"))
    runtime.mode = "paged"
    scope = native_report(runtime.bundle, runtime, mode="paged", hardware="fixture-host",
                          source_revision="fixture-source")["scope"]
    scope["cache_precision"] = "float16"
    path = tmp_path / "peak.json"
    proof = {"schema_version": "native-inference-qualification-v2", "backend": "mlx-one-native",
             "scope": scope, "feature": "activation_peak", "status": "passed",
             "correctness_passed": True, "compatibility_digest": compatibility_digest(runtime),
             "kernel_version": KERNEL_VERSION, "arena_version": "segmented-arena-v1",
             "activation_peak": {"dimensions": {"batch": 2, "chunk": 4, "history": 5,
                "history_lengths": [5, 5], "widths": [4, 4], "phase": "prefill"},
                "policy": "activation-peak-v1-margin20", "incremental_bytes": 4096}}
    path.write_text(json.dumps(proof))
    registry = InferenceFeatureRegistry()
    registry.promote("activation_peak", InferenceScope(**scope), path,
                     InferenceFeatureStatus.CORRECTNESS_PASSED)
    runtime.qualification_registry = registry
    config = runtime.bundle.model.config
    matched = execution_forecast(runtime, config, batch=2, chunk=4, history=5)
    assert matched == {"bytes": 4096, "source": "matching-measured-peak"}
    assert execution_forecast(runtime, config, batch=2, chunk=8, history=5)[
        "source"] == "conservative-analytical"
    path.write_text("{}")
    assert execution_forecast(runtime, config, batch=2, chunk=4, history=5)[
        "source"] == "conservative-analytical"
    runtime.close()


@pytest.mark.parametrize("hybrid", [False, True])
def test_deadline_expiring_in_forward_does_not_commit_or_publish(advanced, monkeypatch, hybrid):
    bundle = advanced(hybrid)
    runtime = CacheManager(bundle, CacheConfig(backend="hybrid" if hybrid else "paged"))
    requests = _requests(bundle, runtime, [range(32)])
    request = requests[0]
    request.handle.metadata["fingerprint"] = runtime.fingerprint(namespace="principal")
    request.cancel, request.deadline = threading.Event(), 1.0
    clock = [0.0]
    monkeypatch.setattr("mlx_one.engine.batching.time.monotonic", lambda: clock[0])
    bundle.model.pause = lambda: clock.__setitem__(0, 2.0)
    (HybridRunner() if hybrid else PagedRunner()).execute(tuple(requests))
    assert request.cancel.is_set() and request.cache[0].offset == 0
    assert runtime.stats().prefix_entries == 0 and runtime.stats().reserved_bytes == 0
    runtime.close()


def test_hybrid_chunk_budget_is_preserved_with_final_aligned_snapshot(advanced):
    bundle = advanced(True)
    with Engine(bundle, inference_mode="experimental_hybrid_batch", cache_namespace="tenant",
                batching_config=BatchingConfig(prefill_chunk_size=512),
                memory_pressure_config=MemoryPressureConfig(enabled=False)) as engine:
        prompt = "Document " * 145
        original = engine.generate(prompt, max_tokens=2)
        traces = engine.runtime_stats()["scheduler"]["forward_traces"]
        assert max(t["real_tokens"] for t in traces if t["phase"] == "prefill") == 512
        repeated = engine.generate(prompt, max_tokens=2)
        assert repeated.text == original.text
        assert engine.runtime_stats()["cache_stats"]["prefix_hits"] > 0


def test_arena_segments_tables_and_leases_are_bounded(advanced):
    runtime = CacheManager(advanced(), CacheConfig(backend="paged", block_size_tokens=16))
    cache = runtime.create_handle("arena").bundle.entries[0]
    values = np.ones((1, 1, 257, 4), dtype=np.float16)
    cache.append(values, values)
    assert len(runtime.block_pool.segments) == 2
    prefix_table = cache.page_table.chunks[0]
    branch = cache.clone()
    before = runtime.block_pool.table_update_bytes
    branch.append(values[:, :, :1], values[:, :, :1])
    assert branch.page_table.chunks[0] is prefix_table
    assert runtime.block_pool.table_update_bytes - before == 16 * 4 * 4
    assert branch._blocks[0].block_id == cache._blocks[0].block_id
    arena = runtime.block_pool
    with arena.lease():
        branch.append(values[:, :, :1], values[:, :, :1])
        assert arena.stats()["replacement_overlap_bytes"] > 0
    assert arena.stats()["replacement_overlap_bytes"] == 0
    assert arena.stats()["table_bytes"] > 0
    assert arena.stats()["page_payload_bytes"] < arena.stats()["physical_bytes"]
    branch.reset()
    runtime.close()
    assert arena.stats()["physical_bytes"] == 0


def test_arena_evaluation_failure_does_not_publish_segment(advanced, monkeypatch):
    import mlx.core as mx

    runtime = CacheManager(advanced(), CacheConfig(backend="paged", block_size_tokens=16))
    cache = runtime.create_handle("arena").bundle.entries[0]
    values = np.ones((1, 1, 16, 4), dtype=np.float16)
    cache.append(values, values)
    arena = runtime.block_pool
    before = {sid: s.state for sid, s in arena.segments.items()}
    live = set(arena._blocks)
    def fail(*args):
        raise MemoryError("injected arena evaluation")
    monkeypatch.setattr(mx, "eval", fail)
    with pytest.raises(MemoryError):
        arena.allocate_ref(tuple(values.copy() for _ in range(2)), 16,
                           layout=cache._blocks[0].block.layout)
    assert set(arena._blocks) == live
    assert all(arena.segments[sid].state is state for sid, state in before.items())
    runtime.close()


def test_cache_aware_age_guard_under_continuous_locality_arrivals(monkeypatch):
    from collections import deque

    from mlx_one.engine.actor import InferenceActor, _Job

    actor = object.__new__(InferenceActor)
    actor.config = BatchingConfig(max_sequences=1, admission_policy="cache_aware")
    actor._condition, actor._active, actor._pending = threading.Condition(), [], deque()
    now = [0.]
    monkeypatch.setattr("mlx_one.engine.actor.time.monotonic", lambda: now[0])
    oldest = _Job(lambda: iter(()), threading.Event(), created_at=0.)
    actor._pending.append(oldest)
    selected = None
    for tick in range(31):
        now[0] = tick / 10
        arriving = _Job(lambda: iter(()), threading.Event(), created_at=now[0])
        arriving.prefix_preview = lambda: 1024
        actor._pending.append(arriving)
        actor._admit()
        if oldest in actor._active:
            selected = now[0]
            break
        actor._active.clear()
    assert selected is not None and selected <= 2.0


def test_uniform_component_contract_and_atomic_snapshot_failure(advanced, tmp_path, monkeypatch):
    from mlx_one.engine.components import StateComponent

    runtime = CacheManager(advanced(True), CacheConfig(backend="hybrid"))
    requests = _requests(runtime.bundle, runtime, [range(32)])
    HybridRunner().execute(tuple(requests))
    identity = {"revision": runtime.bundle.revision}
    first = save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
    for component, spec in zip(requests[0].cache, runtime.plan.state_specs, strict=True):
        assert isinstance(component, StateComponent)
        assert component.capabilities().can_batch
        assert component.committed_position == 32
        component.validate(spec)
    old_manifest = (tmp_path / "state.json").read_bytes()
    original = __import__("os").replace
    def fail_commit(source, target):
        if str(target).endswith("state.json"):
            raise OSError("injected manifest commit failure")
        return original(source, target)
    monkeypatch.setattr("mlx_one.engine.components.os.replace", fail_commit)
    with pytest.raises(OSError):
        save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
    assert (tmp_path / "state.json").read_bytes() == old_manifest
    assert (tmp_path / first["payload"]).exists()
    restored = load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
    assert restored.offset == 32
    restored.release()
    runtime.close()


def test_snapshot_atomic_adoption_rejects_cancelled_and_wrong_dtype(advanced, tmp_path):
    from mlx_one.engine.lifecycle import adopt_snapshot

    runtime = CacheManager(advanced(True), CacheConfig(backend="hybrid"))
    requests = _requests(runtime.bundle, runtime, [range(32)])
    HybridRunner().execute(tuple(requests))
    save_snapshot(tmp_path, requests[0].handle.bundle, identity={"fixture": 1})
    handle = runtime.create_handle("restored")
    staged = load_snapshot(tmp_path, identity={"fixture": 1},
                           expected_specs=runtime.plan.state_specs)
    cancelled = threading.Event()
    cancelled.set()
    assert not adopt_snapshot(runtime, handle, staged, cancel=cancelled)
    assert handle.bundle.offset == 0 and staged.closed
    wrong = tuple(replace(s, dtype="float32") for s in runtime.plan.state_specs)
    with pytest.raises(ValueError, match="shape mismatch|dtype mismatch"):
        load_snapshot(tmp_path, identity={"fixture": 1}, expected_specs=wrong)
    runtime.close()


def test_hybrid_lifecycle_profile_against_independent_cpu_native_fixture(advanced, tmp_path):
    from mlx_one.engine.lifecycle import hybrid_lifecycle
    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile

    profile = BaselineProfile(*CHECKPOINTS["lfm"], "experimental_hybrid_batch",
                              workload="lifecycle", cache_type="f16")
    result = hybrid_lifecycle(advanced(True), profile, tmp_path,
                             reload_bundle=lambda: advanced(True))
    assert result["status"] == "passed", result
    assert result["checks"]["stale_handle_rejected"]
    assert all(row["active_handles"] == 0 for row in result["cleanup"])


def test_snapshot_reader_waits_for_writer_and_legacy_stays_readable(
        advanced, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    runtime = CacheManager(advanced(True), CacheConfig(backend="hybrid"))
    requests = _requests(runtime.bundle, runtime, [range(32)])
    HybridRunner().execute(tuple(requests))
    identity = {"fixture": "locking"}
    first = save_snapshot(tmp_path, requests[0].handle.bundle, identity=identity)
    metadata = dict(first, schema_version="hybrid-state-v1")
    (tmp_path / "tensors.npz").write_bytes((tmp_path / first["payload"]).read_bytes())
    metadata.pop("payload")
    (tmp_path / "state.json").write_text(json.dumps(metadata))
    legacy = load_snapshot(tmp_path, identity=identity, expected_specs=runtime.plan.state_specs)
    assert legacy.offset == 32
    legacy.release()
    entered, release = threading.Event(), threading.Event()
    original = __import__("os").replace
    def paused_commit(source, target):
        if str(target).endswith("state.json"):
            entered.set()
            assert release.wait(3)
        return original(source, target)
    monkeypatch.setattr("mlx_one.engine.components.os.replace", paused_commit)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(save_snapshot, tmp_path, requests[0].handle.bundle, identity=identity)
        assert entered.wait(3)
        reader = pool.submit(load_snapshot, tmp_path, identity=identity,
                             expected_specs=runtime.plan.state_specs)
        assert not reader.done()
        release.set()
        writer.result()
        restored = reader.result()
        assert restored.offset == 32
        restored.release()
    runtime.close()


def test_peak_producer_resets_each_measured_forward_and_records_ragged_scope(advanced, monkeypatch):
    import mlx.core as mx

    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
    from mlx_one.peak_calibration import native_peak_calibration

    resets = []
    for name, value in (("synchronize", lambda: None),
                        ("reset_peak_memory", lambda: resets.append(True)),
                        ("get_peak_memory", lambda: 100_000),
                        ("get_active_memory", lambda: 10_000),
                        ("get_cache_memory", lambda: 100), ("clear_cache", lambda: None)):
        monkeypatch.setattr(mx, name, value, raising=False)
    item = BaselineProfile(*CHECKPOINTS["qwen"], "paged", context_tokens=32,
                           chunk_tokens=16, slots=2, workload="peak_calibration",
                           calibration_phase="prefill", cache_type="f16")
    result = native_peak_calibration(advanced(), item)
    assert len(resets) == 6
    assert result["status"] == "passed"
    assert result["dimensions"] == {"batch": 2, "chunk": 16, "history": 16,
                                    "history_lengths": [16, 16], "widths": [16, 16],
                                    "phase": "prefill"}
    assert result["actual_cache_dtypes"] == ["float16"]


def test_paged_prefix_cannot_substitute_quantized_layout(advanced):
    from mlx_one.engine.modes import ModelCapabilityUnsupported
    from mlx_one.engine.paged import PagedKVCache

    runtime = CacheManager(advanced(), CacheConfig(backend="paged", block_size_tokens=16))
    donor = runtime.create_handle("float").bundle.entries[0]
    values = np.ones((1, 1, 16, 4), dtype=np.float16)
    donor.append(values, values)
    borrower = PagedKVCache(runtime.block_pool, bits=4)
    references = tuple(ref.fork() for ref in donor._blocks)
    with pytest.raises(ModelCapabilityUnsupported, match="precision"):
        borrower.adopt(references)
    assert borrower.offset == 0
    for reference in references:
        reference.close()
    runtime.close()


def test_held_out_native_fixture_scores_all_input_tokens(advanced, tmp_path):
    from test_quality_dataset import manifest

    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
    from mlx_one.quality_dataset import load_quality_manifest, native_dataset_quality

    path = manifest(tmp_path)
    dataset = load_quality_manifest(path)
    bundle = replace(advanced(), tokenizer_identity="immutable-cpu-fixture-tokenizer")
    item = BaselineProfile(*CHECKPOINTS["qwen"], "paged", workload="quality",
                           cache_type="f16", output_tokens=2, quality_manifest=str(path),
                           quality_dataset_identity=dataset["identity"])
    result, retrieval = native_dataset_quality(bundle, item)
    assert result["scored_tokens"] == len(bundle.tokenizer.encode("Held out text.")) - 1
    assert result["document_count"] == 1
    assert abs(result["held_out_nll_mean_delta"]) < .05
    assert not result["coverage"]["eligible"]
    assert len(retrieval["fixtures"]) == 1
