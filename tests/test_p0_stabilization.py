"""P0 regressions use the same instrumented native-model fixture as batching."""

import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_dense_batching import native as native

from mlx_one.p0_qualification import dense_snapshot_parity, inventory, p0_profiles, write_inventory


def test_p0_inventory_is_offline_and_never_qualifies(tmp_path):
    proof = inventory(tmp_path, checkpoints=[("missing", tmp_path / "absent")])
    assert proof["blockers"][0]["reason"] == "local checkpoint missing"
    assert proof["hardware_qualification"] == "pending"
    assert not proof["qualification_promoted"]
    write_inventory(tmp_path / "run", proof)
    checksums = json.loads((tmp_path / "run/checksums.json").read_text())
    assert checksums["readiness.json"] == hashlib.sha256(
        (tmp_path / "run/readiness.json").read_bytes()).hexdigest()
    assert all(p.slots == 1 and p.cache_type == "f16" for p in p0_profiles())


@pytest.mark.parametrize("length,boundary", [(2, 1), (33, 32), (257, 256)])
def test_dense_full_cached_and_trimmed_oracle(native, length, boundary):
    sys.modules["mlx.core"].float32 = np.float32
    proof = dense_snapshot_parity(native, [4] * length, boundary)
    assert proof["passed"] and proof["donor_position"] == length
    assert all(row["offsets"] == [length + i] * 2 for i, row in enumerate(proof["rows"]))


@pytest.mark.parametrize("cooperative", [False, True])
def test_semantic_publication_reuses_real_prefill_and_preserves_donor(native, cooperative):
    from mlx_one.engine.batching import BatchedDenseRunner
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.cache_specs import CacheConfig
    from mlx_one.engine.prompts import prepare_chat
    from mlx_one.text.chat import ChatTemplate
    from mlx_one.text.generation import stream_chat
    from mlx_one.text.schemas import TextGenerationOptions

    native = replace(native, chat_template=ChatTemplate("qwen2", None, native.tokenizer))
    messages = [{"role": "system", "content": "stable knowledge " * 3},
                {"role": "user", "content": "suffix A"}]
    prepared = prepare_chat(native, messages, 2, 2048)
    assert 0 < prepared.semantic_boundary < len(prepared.token_ids)
    runtime = CacheManager(native, CacheConfig())
    handle = runtime.create_handle("donor")
    fp = runtime.fingerprint(namespace="tenant", context_tokens=2048)
    runtime.adopt_prefix(handle, fp, prepared.token_ids)
    handle.metadata.update(prepared_prompt=prepared, semantic_boundary=prepared.semantic_boundary)
    iterator = stream_chat(native, messages, options=TextGenerationOptions(max_tokens=2),
                           cache=handle.bundle.entries, _cache_runtime=runtime,
                           _cache_handle=handle, _cooperative=cooperative,
                           on_cache_update=lambda tokens, _: runtime.publish_prefix(
                               handle, fp, tokens))
    runner = BatchedDenseRunner()
    try:
        from mlx_one.engine.batching import ForwardRequest

        event = next(iterator)
        while True:
            if isinstance(event, ForwardRequest):
                event.request_id = "donor"
                result = runner.execute((event,))
                event = iterator.send(SimpleNamespace(logits=result.logits[0],
                                                      batch_size=1))
            else:
                event = next(iterator)
    except StopIteration:
        pass
    assert runtime.stats().reserved_bytes == 0
    assert runtime.snapshot_prefix.stats()["entries"] == 1
    other = prepare_chat(native, messages[:-1] + [{"role": "user", "content": "suffix B"}],
                         2, 2048)
    borrower = runtime.create_handle("borrower")
    reused = runtime.adopt_prefix(borrower, fp, other.token_ids)
    assert reused == prepared.semantic_boundary
    assert borrower.bundle.offset == reused < handle.bundle.offset
    assert sum(shape[1] for shape, _ in native.model.calls[:-1]) == len(prepared.token_ids)
    runtime.close()


def test_cancelled_staged_snapshot_is_never_published(native):
    from mlx_one.engine.cache_manager import CacheManager

    runtime = CacheManager(native)
    handle = runtime.create_handle("r")
    fp = runtime.fingerprint(namespace="principal")
    native.model(np.array([[4, 5]]), cache=handle.bundle.entries)
    assert runtime.stage_prefix(handle, fp, (4, 5))
    assert runtime.stats().reserved_bytes > 0
    handle.metadata["cancelled"] = True
    assert not runtime.publish_staged_prefix(handle)
    assert runtime.stats().reserved_bytes == 0
    assert runtime.snapshot_prefix.stats()["entries"] == 0
    runtime.release(handle)
    runtime.close()


def test_preview_uses_prepared_context_and_identity_changes(native):
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.prompts import prepare_chat
    from mlx_one.text.chat import ChatTemplate

    native = replace(native, chat_template=ChatTemplate("qwen2", None, native.tokenizer))
    prepared = prepare_chat(native, [{"role": "system", "content": "keep"},
                                    {"role": "user", "content": "old" * 50},
                                    {"role": "assistant", "content": "answer"},
                                    {"role": "user", "content": "new"}],
                            2, 100, context_shift=True)
    assert [m["content"] for m in prepared.messages] == ["keep", "new"]
    runtime = CacheManager(native)
    before = runtime.fingerprint(namespace="p")
    native.model.weights = native.model.weights.astype(np.float32)
    assert before != runtime.fingerprint(namespace="p")
    native.tokenizer.special_tokens = {"new": 4}
    after = runtime.fingerprint(namespace="p")
    assert before.special_tokens_identity != after.special_tokens_identity
    runtime.close()


@pytest.mark.parametrize("surface", ["python", "server"])
def test_auto_paged_constructs_the_selected_pool(native, monkeypatch, surface):
    from mlx_one import Engine
    from mlx_one.engine.arena import SegmentedPageArena
    from mlx_one.server.generation_engine import GenerationEngine

    monkeypatch.setattr("mlx_one.engine.selection.select_auto_mode", lambda *a, **k: {
        "requested_mode": "auto", "resolved_mode": "paged", "reason": "test selection",
        "evidence_scope": None})
    owner = (Engine(native, inference_mode="auto") if surface == "python" else
             GenerationEngine(SimpleNamespace(current_model=lambda: native), inference_mode="auto"))
    try:
        runtime = owner._cache_manager if surface == "python" else owner._cache_manager_for(native)
        assert runtime.config.backend == "paged"
        assert isinstance(runtime.block_pool, SegmentedPageArena)
    finally:
        owner.close()


def test_bundle_cleanup_attempts_every_layer_and_preserves_corruption():
    from mlx_one.engine.cache import CacheBundle
    from mlx_one.engine.cache_errors import CacheCorruptionError

    called = []
    class State:
        def __init__(self, index):
            self.index = index

        def reset(self):
            called.append(self.index)
            if self.index == 0:
                raise CacheCorruptionError("corrupt reference")
    bundle = CacheBundle((State(0), State(1)))
    with pytest.raises(CacheCorruptionError):
        bundle.release()
    assert called == [0, 1]
    assert bundle.closed
    bundle.release()
    assert called == [0, 1]


@pytest.mark.parametrize("times,tps,mean", [([], None, None), ([5], None, None),
                                           ([5, 5.1, 5.3], 2 / .3, 150)])
def test_request_timing_excludes_prefill_and_first_token(times, tps, mean):
    from mlx_one.engine.timing import RequestTiming

    timing = RequestTiming(0, trace=True, limit=2)
    timing.observe({"kind": "forward", "phase": "prefill", "duration_ms": 5000,
                    "input_tokens": 8192, "batch_size": 1, "logits_dtype": "float16"})
    for at in times:
        timing.observe({"kind": "token", "at": at})
    result = timing.summary()
    assert result["decode_tokens_per_second"] == pytest.approx(tps) if tps else (
        result["decode_tokens_per_second"] is None)
    assert result["inter_token_latency_ms"] == pytest.approx(mean) if mean else (
        result["inter_token_latency_ms"] is None)
    assert timing.prefill_tokens == 8192
    assert len(timing.tokens) <= 2


def test_host_admission_does_not_double_count_and_recovers(native, monkeypatch):
    from mlx_one.engine.cache_errors import CacheCapacityError
    from mlx_one.engine.cache_manager import CacheManager
    from mlx_one.engine.pressure import PressureTelemetry

    telemetry = PressureTelemetry(rss_bytes=500, installed_bytes=2000,
                                  mlx_active_bytes=400, mlx_cached_bytes=100)
    monkeypatch.setattr("mlx_one.engine.pressure.sample_telemetry", lambda: telemetry)
    runtime = CacheManager(native)
    with runtime.transient_reservation(900):
        assert runtime.stats().reserved_bytes == 900
    with pytest.raises(CacheCapacityError) as failure, runtime.transient_reservation(1001):
        pass
    assert not failure.value.retryable
    assert runtime.stats().reserved_bytes == 0
    with runtime.transient_reservation(100):
        pass
    assert runtime.capacity_diagnostics["working_set_bytes"] == 500
    runtime.close()
