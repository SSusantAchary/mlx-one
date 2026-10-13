"""Resident allocation, complete reconciliation, and stalled-consumer isolation."""
import threading

import pytest

from mlx_one.engine.actor import InferenceActor
from mlx_one.engine.batching import BatchingConfig
from mlx_one.stress_qualification import MEMORY_METRICS, TouchedMemory, reconcile_memory


def test_real_pages_are_touched_and_allocation_failure_rolls_back():
    memory = TouchedMemory(8192, page_bytes=4096)
    memory.grow(8192)
    assert memory.buffers[0][0] == memory.buffers[0][4096] == 1
    with pytest.raises(MemoryError, match="cap"):
        memory.grow(1)
    assert memory.allocated_bytes == 8192
    memory.close()
    memory.close()
    assert not memory.buffers and memory.allocated_bytes == 0
    calls = []
    def fail(size):
        calls.append(size)
        if len(calls) == 2:
            raise MemoryError("injected allocation")
        return bytearray(size)
    memory = TouchedMemory(32 * 1024 * 1024, allocator=fail)
    with pytest.raises(MemoryError, match="injected"):
        memory.grow(32 * 1024 * 1024)
    assert not memory.buffers and memory.allocated_bytes == 0


def test_reconciliation_cannot_hide_rss_retention_or_missing_telemetry():
    initial = dict.fromkeys(MEMORY_METRICS, 0)
    final = dict(initial)
    tolerances = dict(initial)
    assert reconcile_memory(initial, final, tolerances)["status"] == "passed"
    final["rss_bytes"] = 1024
    assert reconcile_memory(initial, final, tolerances)["status"] == "failed"
    tolerances["rss_bytes"] = 1024
    assert reconcile_memory(initial, final, tolerances)["status"] == "passed"
    final["mlx_cached_bytes"] = None
    assert reconcile_memory(initial, final, tolerances)["status"] == "pending"
    final["mlx_cached_bytes"] = 0
    final["queued_events"] = 1
    tolerances["queued_events"] = 10
    assert reconcile_memory(initial, final, tolerances)["status"] == "failed"


def test_single_execution_slot_pauses_full_consumer_without_blocking_peer(monkeypatch):
    clock = [100.]
    monkeypatch.setattr("mlx_one.engine.actor.time.monotonic", lambda: clock[0])
    actor = InferenceActor(BatchingConfig(max_sequences=1, timeout=0,
                                         output_buffer_events=32, output_stall_timeout_s=30))
    blocked = threading.Event()
    actor.set_event_sink(lambda e: blocked.set() if e["event"] == "blocked" else None)
    try:
        slow = actor._enqueue(lambda cancel: iter(range(1000)))
        assert blocked.wait(2)
        peer = actor._enqueue(lambda cancel: iter(("peer progressed",)))
        assert peer.output.get(timeout=2) == "peer progressed"
        peer.terminal.result(timeout=2)
        assert slow.output.qsize() == 32 and not slow.terminal.done()
        assert actor.stats()["paused_slots"] == 1
        clock[0] = 131
        with pytest.raises(TimeoutError, match="stalled"):
            slow.terminal.result(timeout=2)
        assert slow.output.qsize() == 32
    finally:
        actor.close()


def test_barrier_cancels_parked_consumers_before_epoch_changes():
    actor = InferenceActor(BatchingConfig(max_sequences=1, timeout=0, output_buffer_events=1))
    blocked = threading.Event()
    actor.set_event_sink(lambda e: blocked.set() if e["event"] == "blocked" else None)
    try:
        slow = actor._enqueue(lambda cancel: iter(range(1000)))
        assert blocked.wait(2)
        before = actor.stats()["model_epoch"]
        assert actor.execute(lambda: "reload barrier") == "reload barrier"
        assert slow.cancel.is_set() and slow.terminal.done()
        assert actor.stats()["model_epoch"] == before + 1
    finally:
        actor.close()


def test_shutdown_finalization_refreshes_backing_checksums(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from mlx_one.inference_baseline import CHECKPOINTS, BaselineProfile
    from mlx_one.inference_qualification import finalize_shutdown

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"failed": 0}))
    proof_path = tmp_path / "evidence-paged_attention.json"
    proof_path.write_text(json.dumps({"feature": "paged_attention", "status": "passed",
        "correctness_passed": True, "artifacts": [{"uri": str(path), "kind": "native-result",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    scheduler = SimpleNamespace(_thread=SimpleNamespace(is_alive=lambda: False))
    item = BaselineProfile(*CHECKPOINTS["qwen"], "paged")
    finalize_shutdown(tmp_path, item, scheduler, None)
    proof = json.loads(proof_path.read_text())
    assert proof["artifacts"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert any(a["uri"].endswith("shutdown.json") for a in proof["artifacts"])
    assert json.loads(path.read_text())["shutdown"] == {
        "actor_alive": False, "server_alive": False}
