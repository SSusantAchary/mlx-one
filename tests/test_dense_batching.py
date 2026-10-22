"""CPU differential tests for dense packing, actor ownership, and cache transactions."""

import asyncio
import sys
import time
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one import BatchingConfig, Engine
from mlx_one.core.cache import KVCache
from mlx_one.core.outputs import ModelOutput
from mlx_one.engine.actor import InferenceActor
from mlx_one.engine.batching import BatchedDenseRunner, BatchScheduler, ForwardRequest
from mlx_one.engine.block_cache import BlockPool
from mlx_one.engine.block_kv import BlockKVCache
from mlx_one.engine.cache_errors import CacheCapacityError, CacheCorruptionError
from mlx_one.engine.cache_manager import CacheManager
from mlx_one.engine.cache_specs import CacheConfig
from mlx_one.engine.contracts import BatchPlan, EngineRequest, Phase, RequestKind, ScheduledSequence
from mlx_one.text.loading import LoadedTextModel


@pytest.fixture
def native(monkeypatch, tmp_path):
    core = ModuleType("mlx.core")
    core.array, core.zeros, core.concatenate = np.array, np.zeros, np.concatenate
    core.float16, core.bfloat16 = np.float16, np.float32
    core.argmax = np.argmax
    core.eval = lambda *args: None
    core.clear_cache = lambda: None
    package = ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 63
        pad_token_id = 0

        def encode(self, text):
            return [3 + ord(c) % 20 for c in text]

        def decode(self, ids):
            return "".join(chr(65 + int(i) % 26) for i in ids)

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(model_type="qwen2", num_hidden_layers=2,
                                          num_key_value_heads=1, num_attention_heads=1,
                                          hidden_size=2, max_position_embeddings=32768,
                                          rope_theta=10000)
            self.calls = []
            self.weights = np.ones((2,), dtype=np.float16)
            self.fail = False
            self.pause = None

        def parameters(self):
            return {"weights": self.weights}

        def set_dtype(self, dtype):
            self.weights = self.weights.astype(dtype)

        def make_cache(self):
            return (KVCache(), KVCache())

        def __call__(self, ids, *, cache, attention_mask=None, position_ids=None):
            if self.pause is not None:
                self.pause()
            batch, width = ids.shape
            history = cache[0].offset
            if position_ids is None:
                position_ids = np.broadcast_to(np.arange(history, history + width), ids.shape)
            data = np.stack((ids, position_ids), axis=-1).astype(np.float16)[:, None]
            self.calls.append((ids.shape, position_ids.copy()))
            for layer in cache:
                keys, _ = layer.update(data, data)
            if self.fail:
                raise RuntimeError("injected forward failure")
            logits = np.empty((batch, width, 64), dtype=np.float16)
            for row in range(batch):
                for q in range(width):
                    values = keys[row, 0, :history + q + 1]
                    valid = (np.ones(len(values), dtype=bool) if attention_mask is None else
                             attention_mask[row, :history + q + 1] != 0)
                    target = int(values[valid].sum()) % 60
                    logits[row, q] = -np.abs(np.arange(64) - target)
            return ModelOutput(logits, data, cache)

    return LoadedTextModel(Model(), Tokenizer(), tmp_path, "test/qwen", "a" * 40,
                           architecture="qwen2", context_length=32768)


def request(bundle, tokens, name, phase="prefill", cache=None):
    return ForwardRequest(bundle, tuple(tokens), cache or bundle.model.make_cache(), phase,
                          tuple(tokens), request_id=name)


def test_ragged_prefill_decode_matches_independent_forced_tokens(native):
    runner = BatchedDenseRunner()
    requests = [request(native, (4, 5, 6), "a"), request(native, (7,), "b")]
    independent = [request(native, r.token_ids, r.request_id) for r in requests]
    result = runner.execute(tuple(requests))
    assert native.model.calls[-1][0] == (2, 3)
    for row, reference in enumerate(independent):
        expected = runner.execute((reference,))
        np.testing.assert_array_equal(result.logits[row], expected.logits[0])
    for step in range(4):
        for group in (requests, independent):
            for r in group:
                r.phase, r.token_ids = "decode", (10 + step,)
        result = runner.execute(tuple(requests))
        assert native.model.calls[-1][0] == (2, 1)
        for row, reference in enumerate(independent):
            expected = runner.execute((reference,))
            np.testing.assert_array_equal(result.logits[row], expected.logits[0])
            assert requests[row].cache[0].offset == reference.cache[0].offset
    # Compact caches own allocations rather than retaining a packed B>1 buffer view.
    assert requests[0].cache[0].keys.base is None


def test_runner_contract_preserves_unequal_committed_boundaries(native):
    runner = BatchedDenseRunner(native)
    states = [runner.prepare(EngineRequest(RequestKind.TEXT, native.model_id,
                                          {"prompt": prompt}, request_id=name))
              for name, prompt in (("short", "ab"), ("long", "abcdef"))]
    for state in states:
        state.transition(Phase.PREFILL)
    plan = BatchPlan(tuple(ScheduledSequence(s.request_id, "prefill", 0, s.prompt_len)
                           for s in states), sum(s.prompt_len for s in states))
    result = runner.prefill(plan)
    assert result.request_ids == ("short", "long")
    assert native.model.calls[-1][0] == (2, 6)
    assert [s.computed_len for s in states] == [2, 6]
    for state in states:
        state.transition(Phase.DECODE)
        state.generated_ids.append(12)
    assert [s.computed_len for s in states] == [2, 6]
    result = runner.decode(BatchPlan(tuple(
        ScheduledSequence(s.request_id, "decode", s.computed_len, 1) for s in states), 2))
    assert result.committed_token_counts == (1, 1)
    assert native.model.calls[-1][0] == (2, 1)
    assert [s.computed_len for s in states] == [3, 7]
    for state in states:
        runner.finalize(state)
        runner.finalize(state)
        assert state.cache.closed


def test_copy_telemetry_counts_real_ragged_history_separately_from_padding(native):
    runner = BatchedDenseRunner(native)
    a, b = request(native, (4, 5, 6), "a"), request(native, (7,), "b")
    runner.execute((a, b))
    a.phase = b.phase = "decode"
    a.token_ids = b.token_ids = (8,)
    runner.execute((a, b))
    trace = runner.last_trace
    per_token = runner.adapter.plan.bytes_per_token()
    assert trace["history_pack_bytes"] == trace["history_scatter_bytes"] == 4 * per_token
    assert trace["packed_history_bytes"] == 6 * per_token
    assert trace["kv_delta_write_bytes"] == 2 * per_token
    assert trace["pack_allocated_bytes"] == 8 * per_token
    assert all(trace[key] >= 0 for key in ("pack_graph_time_ms", "forward_eval_time_ms",
                                          "stage_eval_time_ms", "commit_time_ms"))
    assert trace["gpu_sync_time_ms"] is None


def test_scatter_failure_and_invalid_mappings_are_atomic(native, monkeypatch):
    runner = BatchedDenseRunner()
    a, b = request(native, (1, 2), "a"), request(native, (3,), "b")
    with pytest.raises(ValueError, match="unique"):
        runner.execute((a, replace(b, request_id="a")))
    native.model.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute((a, b))
    assert all(c.offset == 0 for r in (a, b) for c in r.cache)
    native.model.fail = False
    original = KVCache.update
    def fail_scatter(self, keys, values):
        if keys.shape[0] == 1:
            raise RuntimeError("scatter failed")
        return original(self, keys, values)
    monkeypatch.setattr(KVCache, "update", fail_scatter)
    with pytest.raises(RuntimeError, match="scatter"):
        runner.execute((a, b))
    assert all(c.offset == 0 for r in (a, b) for c in r.cache)


def test_combined_budget_and_rotation(native):
    cfg = BatchingConfig(max_scheduled_tokens=4, max_prefill_tokens=2, prefill_chunk_size=2)
    policy = BatchScheduler(cfg)
    decodes = [request(native, (1,), str(i), "decode") for i in range(5)]
    prefills = [request(native, (1, 2), f"p{i}") for i in range(3)]
    seen = set()
    for _ in range(8):
        groups = policy.select(decodes + prefills)
        assert sum(len(r.token_ids) for g in groups for r in g) <= 4
        seen.update(r.request_id for g in groups for r in g)
    assert seen == {r.request_id for r in decodes + prefills}


def test_python_concurrent_submissions_batch_and_preserve_output(native):
    config = BatchingConfig(max_sequences=4, prefill_chunk_size=2, max_prefill_tokens=4,
                            max_scheduled_tokens=8)
    with Engine(native, inference_mode="experimental_dense_batch",
                batching_config=config) as engine:
        handles = [engine.submit(prompt, max_tokens=8, seed=i)
                   for i, prompt in enumerate(("abc", "abcdef", "xy", "longer"))]
        results = [handle.result(timeout=5) for handle in handles]
        trace = engine.runtime_stats()["scheduler"]["forward_traces"]
        assert any(t["batch_size"] > 1 and t["phase"] == "decode" for t in trace)
        assert any(len(set(t["history_lengths"])) > 1 for t in trace if t["batch_size"] > 1)
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
        assert engine.runtime_stats()["cache_stats"]["reserved_bytes"] == 0
    with Engine(native) as reference:
        expected = [reference.generate(prompt, max_tokens=8, seed=i).text
                    for i, prompt in enumerate(("abc", "abcdef", "xy", "longer"))]
    assert [result.text for result in results] == expected


def test_actor_pressure_shedding_runs_on_the_device_owner(native):
    calls = []
    config = BatchingConfig(max_sequences=2, prefill_chunk_size=1,
                            max_scheduled_tokens=1, max_prefill_tokens=1)
    with Engine(native, inference_mode="experimental_dense_batch",
                batching_config=config) as engine:
        engine._actor.pressure = SimpleNamespace(
            state="normal", poll=lambda: "normal", stats=lambda: {},
            shed=lambda runtime: calls.append(engine._actor.device_executor.is_owner))
        with engine._actor._condition:
            handles = [engine.submit(prompt, max_tokens=6) for prompt in ("abc", "unequal")]
        for handle in handles:
            handle.result(timeout=5)
    assert calls and all(calls)


def test_actor_bounded_output_cancellation_and_barrier(native):
    actor = InferenceActor(BatchingConfig(max_sequences=2, output_buffer_events=2,
                                         output_stall_timeout_s=.05, timeout=1))
    try:
        slow = actor._enqueue(lambda cancel: iter(range(100)))
        fast = actor._enqueue(lambda cancel: iter(("ok",)))
        assert fast.output.get(timeout=1) == "ok"
        fast.terminal.result(timeout=1)
        assert slow.output.qsize() <= 2
        with pytest.raises(TimeoutError, match="stalled"):
            slow.terminal.result(timeout=1)
        epoch = actor.stats()["model_epoch"]
        assert actor.execute(lambda: "reload") == "reload"
        assert actor.stats()["model_epoch"] == epoch + 1
    finally:
        actor.close()


def test_block_rollback_ids_fp16_and_dense_prefix(native, monkeypatch):
    pool = BlockPool(1024, block_tokens=2)
    cache = BlockKVCache(pool)
    values = np.ones((1, 1, 1, 2), dtype=np.float32)
    cache.update(values, values)
    before = cache.state()[0].copy()
    original = pool.allocate_ref
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CacheCapacityError("injected allocation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(pool, "allocate_ref", fail)
    with pytest.raises(CacheCapacityError):
        cache.update(np.ones((1, 1, 4, 2)), np.ones((1, 1, 4, 2)))
    assert cache.offset == 1
    np.testing.assert_array_equal(cache.state()[0], before)
    assert cache.state()[0].dtype == np.float16
    old_id = cache._blocks[0].block_id
    cache.reset()
    monkeypatch.setattr(pool, "allocate_ref", original)
    cache.update(values, values)
    with pytest.raises(CacheCorruptionError):
        pool.get(old_id)
    manager = CacheManager(native)
    donor = manager.create_handle("donor")
    tokens = (1, 2, 3, 4)
    data = np.ones((1, 1, 4, 2), dtype=np.float16)
    for layer in donor.bundle.entries:
        layer.update(data, data)
    fingerprint = manager.fingerprint(namespace="tenant:a")
    assert manager.publish_prefix(donor, fingerprint, tokens)
    borrower = manager.create_handle("borrower")
    assert manager.adopt_prefix(borrower, fingerprint, tokens) == 3
    other = manager.create_handle("other")
    assert manager.adopt_prefix(other, manager.fingerprint(namespace="tenant:b"), tokens) == 0
    assert manager.adopt_prefix(other, fingerprint, (1, 2, 9, 10)) == 2
    manager.release(donor)
    assert borrower.bundle.offset == 3
    manager.close()


def test_actor_cancels_one_request_while_peer_progresses(native):
    with Engine(native, inference_mode="experimental_dense_batch",
                batching_config=BatchingConfig(max_sequences=2, prefill_chunk_size=2)) as engine:
        cancelled = engine.submit("cancel this", max_tokens=100)
        live = engine.submit("continue", max_tokens=12)
        cancelled.cancel()
        assert live.result(timeout=5).generation_tokens == 12
        deadline = time.monotonic() + 2
        while (engine.runtime_stats()["cache_stats"]["active_handles"]
               and time.monotonic() < deadline):
            time.sleep(.001)
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0


def test_precision_and_mode_conflicts(native):
    with pytest.raises(ValueError, match="conflicts"):
        Engine(native, inference_mode="experimental_dense_batch",
               cache_config=CacheConfig(backend="block"))
    native.model.weights = native.model.weights.astype(np.float32)
    with pytest.raises(ValueError, match="observed FP16"):
        Engine(native, inference_mode="experimental_dense_batch")
    with Engine(native, inference_mode="experimental_dense_batch", execution_dtype="float16"):
        assert native.model.weights.dtype == np.float16


@pytest.mark.parametrize("mode", ["experimental_block", "experimental_dense_batch"])
def test_optimized_parity_profiles_with_cpu_math(native, mode, monkeypatch):
    from mlx_one.engine.qualification import execution_parity

    core = sys.modules["mlx.core"]
    for name in ("abs", "max", "mean", "all", "allclose"):
        monkeypatch.setattr(core, name, getattr(np, name), raising=False)
    core.float32 = np.float32
    def softmax(value, axis=-1):
        exponent = np.exp(value - np.max(value, axis=axis, keepdims=True))
        return exponent / exponent.sum(axis=axis, keepdims=True)
    core.softmax = softmax
    result = execution_parity(native, mode=mode, prompt="A sufficiently long shared prefix. " * 3,
                              count=6, chunk_size=7)
    assert result["status"] == "passed", result
    assert all(row["reused_tokens"] > 0 for row in result["prefix_restoration"])
    assert all(row["cached_single_request"]["within_tolerance"]
               for row in result["prefix_restoration"])
    assert result["seeded_rng_isolation"]


def test_http_batched_requests_principal_isolation_and_sse(native):
    import httpx

    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.generation_engine import GenerationEngine

    class Template:
        template = "plain"
        def render(self, messages, **kwargs):
            return " ".join(m["content"] for m in messages)
    bundle = replace(native, chat_template=Template())
    info = SimpleNamespace(id=bundle.model_id, context_length=bundle.context_length,
                           to_dict=lambda: {"id": bundle.model_id},
                           to_openai=lambda: {"id": bundle.model_id})
    manager = SimpleNamespace(current_model=lambda: bundle, model_info=lambda: info,
                              unload=lambda: None)
    config = BatchingConfig(max_sequences=2, prefill_chunk_size=2, max_scheduled_tokens=8,
                            max_prefill_tokens=4)
    actor = InferenceActor(config)
    engine = GenerationEngine(manager, inference_mode="experimental_dense_batch",
                              prefill_chunk_size=2, collect_execution_metrics=True)
    app = create_app(manager, engine, scheduler=actor, config=ServerConfig(
        inference_mode="experimental_dense_batch", parallel=2, cache_prompt=True, cache_reuse=1,
        timeout=5, warmup=False, api_keys=("alice", "bob")))
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            async def one(prompt, key="alice", streaming=False):
                response = await client.post("/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"}, json={"model": bundle.model_id,
                    "messages": [{"role": "user", "content": prompt}], "max_tokens": 8,
                    "stream": streaming, "temperature": 0})
                assert response.status_code == 200, response.text
                return response
            first, second = await asyncio.gather(
                one("shared prefix a"), one("shared prefix longer"))
            assert first.json()["usage"]["completion_tokens"] == 8
            same = await one("shared prefix a", streaming=True)
            assert "data: [DONE]" in same.text
            assert '"prompt_cache_hit":true' in same.text.replace(" ", "")
            separate = await one("shared prefix a", key="bob", streaming=True)
            assert '"prompt_cache_hit":false' in separate.text.replace(" ", "")
            runtime = (await client.get(
                "/v1/runtime", headers={"Authorization": "Bearer alice"})).json()
            assert runtime["inference"]["engine_mode"] == "experimental_dense_batch"
            assert runtime["inference"]["capabilities"]["continuous_batching"]
            assert any(trace["batch_size"] == 2 and trace["phase"] == "decode"
                       for trace in runtime["scheduler"]["forward_traces"])
            assert "alice" not in str(runtime["cache"])
    try:
        asyncio.run(run())
    finally:
        actor.close(finalizer=engine.close)


def test_multi_layer_prefix_adoption_rolls_back_every_reference(native, monkeypatch):
    manager = CacheManager(native, CacheConfig(backend="block", block_size_tokens=16))
    donor = manager.create_handle("donor")
    values = np.ones((1, 1, 32, 2), dtype=np.float16)
    for cache in donor.bundle.entries:
        cache.update(values, values)
    fp = manager.fingerprint(namespace="same")
    assert manager.publish_prefix(donor, fp, tuple(range(32)))
    refs = [ref.block.references for cache in donor.bundle.entries for ref in cache._blocks]
    borrower = manager.create_handle("borrower")
    original = BlockKVCache.adopt
    def fail_second(self, references):
        if self.layer_id == 1:
            raise RuntimeError("second layer adoption failed")
        return original(self, references)
    monkeypatch.setattr(BlockKVCache, "adopt", fail_second)
    with pytest.raises(RuntimeError, match="second layer"):
        manager.adopt_prefix(borrower, fp, tuple(range(32)))
    assert borrower.bundle.offset == 0
    assert refs == [ref.block.references for cache in donor.bundle.entries for ref in cache._blocks]
    manager.close()


def test_memory_reconciliation_and_transient_failure_leave_no_reservations(native):
    manager = CacheManager(native, CacheConfig(memory_budget_bytes=8192))
    handle = manager.create_handle("memory")
    reservation = manager.reserve(handle, tokens=8, requested_bytes=4096)
    data = np.ones((1, 1, 8, 2), dtype=np.float16)
    for cache in handle.bundle.entries:
        cache.update(data, data)
    manager.reconcile(handle)
    stats = manager.stats()
    assert stats.physical_bytes + stats.reserved_bytes == 4096
    with pytest.raises(CacheCapacityError):
        with manager.transient_reservation(8192):
            raise AssertionError("unreachable")
    assert manager.stats().reserved_bytes == 0
    manager.release_reservation(reservation)
    manager.release_reservation(reservation)
    manager.release(handle)
    other = CacheManager(native)
    with pytest.raises(RuntimeError, match="another manager"):
        other.release_reservation(reservation)
    other.close()
    manager.close()


def test_cancellation_during_forward_does_not_commit_cancelled_cache(native):
    import threading

    a, b = request(native, (1, 2), "cancel"), request(native, (3, 4), "live")
    a.cancel = threading.Event()
    native.model.pause = a.cancel.set
    BatchedDenseRunner().execute((a, b))
    assert all(cache.offset == 0 for cache in a.cache)
    assert all(cache.offset == 2 for cache in b.cache)


def test_actor_deadlines_expire_queued_and_paused_requests(native):
    actor = InferenceActor(BatchingConfig(max_sequences=1, timeout=.02,
                                         output_buffer_events=1, output_stall_timeout_s=1))
    try:
        active = actor._enqueue(lambda cancel: iter(range(10)))
        queued = actor._enqueue(lambda cancel: iter((1,)))
        for job in (active, queued):
            with pytest.raises(TimeoutError, match="timed out"):
                job.terminal.result(timeout=1)
        assert actor.stats()["active_slots"] == 0
    finally:
        actor.close()


def test_unfit_admission_releases_slot_for_feasible_request(native):
    actor = InferenceActor(BatchingConfig(max_sequences=1, timeout=.1))
    try:
        blocked = actor._enqueue(lambda cancel: iter(("blocked",)), admission=lambda: "queue")
        feasible = actor._enqueue(lambda cancel: iter(("feasible",)))
        assert feasible.output.get(timeout=1) == "feasible"
        feasible.terminal.result(timeout=1)
        with pytest.raises(TimeoutError):
            blocked.terminal.result(timeout=1)
        assert actor.stats()["skipped_admissions"] > 0
    finally:
        actor.close()


def test_python_submit_queue_is_bounded_before_collector_execution(native):
    import threading

    from mlx_one.engine.execution_context import QueueFullError

    entered, release = threading.Event(), threading.Event()
    with Engine(native, inference_mode="experimental_dense_batch",
                batching_config=BatchingConfig(queue_size=1)) as engine:
        def barrier():
            entered.set()
            release.wait(2)
        thread = threading.Thread(target=lambda: engine._actor.execute(barrier))
        thread.start()
        assert entered.wait(1)
        try:
            handle = engine.submit("first", max_tokens=2)
            with pytest.raises(QueueFullError):
                engine.submit("overflow", max_tokens=2)
        finally:
            release.set()
            thread.join()
        assert handle.result(timeout=2).generation_tokens == 2
        assert engine.runtime_stats()["scheduler"]["rejected_requests"] == 1


@pytest.mark.parametrize("field,value", [
    ("namespace", "other-principal"), ("model_identity", "other-weights"),
    ("adapter_identity", "other-adapter"), ("special_tokens_identity", "other-special-tokens"),
    ("tokenizer_identity", "other-tokenizer"), ("template_identity", "other-template"),
    ("rendering_identity", "other-render-options"), ("rope_identity", "other-rope"),
    ("context_tokens", 99),
])
def test_prefix_identity_changes_never_share_pages(native, field, value):
    manager = CacheManager(native, CacheConfig(backend="block", block_size_tokens=16))
    donor = manager.create_handle("donor")
    values = np.ones((1, 1, 32, 2), dtype=np.float16)
    for cache in donor.bundle.entries:
        cache.update(values, values)
    ids = tuple(range(32))
    fingerprint = manager.fingerprint(namespace="principal")
    assert manager.publish_prefix(donor, fingerprint, ids)
    borrower = manager.create_handle("borrower")
    assert manager.adopt_prefix(borrower, replace(fingerprint, **{field: value}), ids) == 0
    assert borrower.bundle.offset == 0
    manager.close()


def test_multimodal_restoration_is_explicitly_unavailable(native):
    from mlx_one.engine.cache_errors import PrefixReuseUnsupported

    manager = CacheManager(native)
    handle = manager.create_handle("multimodal")
    fingerprint = manager.fingerprint(namespace="same", multimodal_identity="image-sha256")
    with pytest.raises(PrefixReuseUnsupported):
        manager.adopt_prefix(handle, fingerprint, (1, 2))
    manager.close()


def test_one_token_budget_alternates_prefill_and_decode(native):
    policy = BatchScheduler(BatchingConfig(max_scheduled_tokens=1, prefill_chunk_size=1))
    requests = [request(native, (1,), "decode", "decode"), request(native, (1,), "prefill")]
    selected = [policy.select(requests)[0][0].phase for _ in range(4)]
    assert selected == ["prefill", "decode", "prefill", "decode"]
