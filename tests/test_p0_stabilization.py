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


def test_prefix_gate_requires_native_token_ids_and_actual_skipped_work():
    from mlx_one.p0_qualification import prefix_gate

    rows = []
    for phase in ("cold", "warm"):
        for _ in range(3):
            rows.append({"status": "completed", "temperature_phase": phase,
                         "output_token_ids": [1, 2, 3],
                         "client_ttft_ms": 100 if phase == "cold" else 85,
                         "metrics": {"actual_prefill_tokens": 8192 if phase == "cold" else 192,
                                     "reused_prompt_tokens": 0 if phase == "cold" else 8000}})
    assert prefix_gate(rows, 8000)["status"] == "passed"
    rows[-1]["metrics"]["actual_prefill_tokens"] = 8192
    assert not prefix_gate(rows, 8000)["checks"]["actual_skipped_prefill"]
    rows[-1]["output_token_ids"] = None
    assert not prefix_gate(rows, 8000)["checks"]["token_id_parity"]


def test_http_benchmark_auth_is_not_serialized(monkeypatch):
    from benchmarks.cache_runtime_bench import authentication_headers, exact_prefix_metadata
    from mlx_one.text.chat import ChatTemplate

    monkeypatch.setenv("P0_TEST_SECRET", "opaque-secret")
    assert authentication_headers("P0_TEST_SECRET") == {"Authorization": "Bearer opaque-secret"}
    assert authentication_headers(None) == {}
    with pytest.raises(SystemExit, match="unset or empty"):
        authentication_headers("P0_MISSING_SECRET")
    tokenizer = SimpleNamespace(encode=lambda text: list(text.encode()))
    metadata = exact_prefix_metadata(tokenizer, ChatTemplate("qwen2", None, tokenizer),
                                     ["prefix-A", "prefix-B"])
    assert metadata["verified_common_prefix_tokens"] > len("prefix-")
    assert metadata["rendered_token_sha256"][0] != metadata["rendered_token_sha256"][1]


@pytest.mark.parametrize("schema,kind", [("native-inference-qualification-v2", "native-hardware"),
                                        ("native-inference-qualification-v4", "synthetic")])
def test_unmeasured_or_legacy_evidence_cannot_promote(tmp_path, schema, kind):
    from dataclasses import asdict

    from mlx_one.engine.selection import load_evidence
    from mlx_one.registry import InferenceFeatureRegistry, InferenceScope, RegistryError

    scope = InferenceScope("model", "a" * 40, "paged", "topology", "float16", "float16",
                           "0.32.2", "M4", "source")
    proof = {"schema_version": schema, "validation_kind": kind, "scope": asdict(scope),
             "backend": "mlx-one-native", "feature": "paged_attention", "status": "passed",
             "correctness_passed": True, "performance_qualified": True}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    with pytest.raises(RegistryError):
        InferenceFeatureRegistry().promote("paged_attention", scope, path, "performance_qualified")
    if schema.endswith("v2"):
        assert load_evidence([path]).report("paged_attention", scope)["status"] == "experimental"


def test_feature_details_distinguish_implementation_from_validation(native):
    from mlx_one import Engine

    with Engine(native) as engine:
        report = engine.runtime_stats()
        dense = report["feature_details"]["dense_execution"]
        assert dense["implemented"] and dense["available_for_opt_in"]
        assert not dense["hardware_correctness_passed"]
        assert not dense["performance_qualified"]
        assert not report["feature_details"]["quantized_paged_kv"]["available_for_opt_in"]
        assert report["selection"]["resolved_mode"] == "reference_dense"
        assert report["scope"]["source_revision"]


@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_p0_concurrency_matrix_preserves_independent_output_and_cleanup(native, slots):
    from mlx_one import BatchingConfig, Engine

    prompts = ["x" * (i + 1) for i in range(slots * 2)]
    config = BatchingConfig(max_sequences=slots, queue_size=slots * 2,
                            prefill_chunk_size=2, max_prefill_tokens=8,
                            max_scheduled_tokens=16)
    with Engine(native, inference_mode="experimental_dense_batch",
                batching_config=config) as engine:
        tasks = [engine.submit(prompt, max_tokens=5, seed=i) for i, prompt in enumerate(prompts)]
        actual = [task.result(timeout=5).text for task in tasks]
        stats = engine.runtime_stats()["cache_stats"]
        assert stats["active_handles"] == stats["reserved_bytes"] == 0
    with Engine(native) as reference:
        expected = [reference.generate(prompt, max_tokens=5, seed=i).text
                    for i, prompt in enumerate(prompts)]
    assert actual == expected


def test_auto_loads_missing_and_incomplete_evidence_as_unqualified(tmp_path):
    from mlx_one.engine.selection import load_evidence

    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":"native-inference-qualification-v4"}')
    assert not load_evidence([tmp_path / "missing.json", path]).records()


def test_model_timing_excludes_buffered_stop_tokens(native):
    from mlx_one.engine.timing import RequestTiming
    from mlx_one.text.generation import _stream_bundle
    from mlx_one.text.schemas import TextGenerationOptions

    stop_ids = native.tokenizer.encode("abc")
    timing = RequestTiming(0, trace=True)
    chunks = list(_stream_bundle(native, "x", TextGenerationOptions(max_tokens=4, stop=("abc",)),
                                 None, _forced_token_ids=(4, *stop_ids),
                                 on_execution=timing.observe))
    assert chunks[-1].finish_reason == "stop"
    assert timing.count == 1
    assert tuple(timing.token_ids) == (4,)
    assert timing.summary()["decode_tokens_per_second"] is None
