import time
from types import SimpleNamespace

import pytest

from mlx_one.engine import (
    AdmissionStatus,
    ASRRunner,
    BlockAllocator,
    CacheBundle,
    CausalLMRunner,
    EncoderRunner,
    EngineRequest,
    ExecutionBatch,
    MemoryBudget,
    MemoryUsage,
    PageTable,
    PrefixCache,
    RequestKind,
    SequenceState,
    TokenBudgetScheduler,
    UnifiedMemoryPlanner,
    VLMRunner,
    forecast_kv_bytes,
)


class FakeArray:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes


class FakeCache:
    def __init__(self, size: int, offset: int = 0) -> None:
        self.value = FakeArray(size) if size else None
        self.offset = offset
        self.released = False

    @property
    def nbytes(self) -> int:
        return 0 if self.value is None else self.value.nbytes

    def clone(self):
        return FakeCache(self.nbytes, self.offset)

    def reset(self) -> None:
        self.value = None
        self.offset = 0
        self.released = True

    def trim(self, count: int) -> bool:
        self.offset = max(0, self.offset - count)
        return True


def request(**overrides):
    values = {
        "kind": RequestKind.TEXT,
        "model": "test/model",
        "payload": {"prompt": "hello"},
        "request_id": "req-test",
        "created_at": time.monotonic(),
        "max_tokens": 4,
    }
    values.update(overrides)
    return EngineRequest(**values)


def test_engine_request_and_sequence_state_contract() -> None:
    item = CausalLMRunner(
        SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda value: [1, 2], bos_token_id=None),
            model=SimpleNamespace(make_cache=lambda: (FakeCache(8),)),
        )
    ).prepare(request())
    assert item.state is SequenceState.WAITING
    assert item.remaining_prompt_tokens == 2
    item.transition(SequenceState.PREFILL)
    item.computed_tokens = 2
    item.transition(SequenceState.DECODING)
    item.transition(SequenceState.FINISHED)
    assert item.state.terminal
    with pytest.raises(ValueError, match="invalid sequence transition"):
        item.transition(SequenceState.DECODING)


def test_request_rejects_invalid_deadline_and_token_limit() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        request(max_tokens=-1)
    created = time.monotonic()
    with pytest.raises(ValueError, match="deadline"):
        request(created_at=created, deadline=created - 1)


def test_non_autoregressive_task_runners_dispatch_by_capability() -> None:
    embedding = EncoderRunner(lambda **payload: payload["texts"])
    value = embedding.execute(
        request(kind=RequestKind.EMBEDDING, payload={"texts": ("one",)})
    )
    assert value == ("one",)
    assert embedding.capabilities.batchable

    vlm = VLMRunner(lambda **payload: payload["images"])
    assert vlm.execute(
        request(kind=RequestKind.VISION, payload={"images": ("image",)})
    ) == ("image",)
    assert vlm.capabilities.modalities == frozenset({"text", "image"})

    asr = ASRRunner(lambda **payload: payload["audio"])
    assert asr.execute(
        request(kind=RequestKind.TRANSCRIPTION, payload={"audio": b"audio"})
    ) == b"audio"
    with pytest.raises(ValueError, match="ASR runner"):
        asr.execute(request())


def test_cache_bundle_accounts_clones_trims_and_releases() -> None:
    bundle = CacheBundle((FakeCache(12, 5), FakeCache(20, 5)))
    assert bundle.nbytes == 32
    assert bundle.offset == 5
    cloned = bundle.snapshot()
    assert cloned is not bundle
    assert cloned.entries[0] is not bundle.entries[0]
    assert cloned.trim(2)
    assert cloned.offset == 3
    cloned.release()
    assert cloned.nbytes == 0
    assert bundle.nbytes == 32


def test_prefix_cache_selects_longest_compatible_prefix_and_evicts_lru() -> None:
    cache = PrefixCache(capacity_bytes=24)
    compatibility = ("model", "revision", "f16")
    assert cache.put(compatibility, (1, 2), CacheBundle((FakeCache(8),)))
    assert cache.put(compatibility, (1, 2, 3), CacheBundle((FakeCache(8),)))
    match = cache.longest_prefix(compatibility, (1, 2, 3, 4), minimum_tokens=2)
    assert match is not None and match[0] == (1, 2, 3)
    assert cache.longest_prefix(("other",), (1, 2, 3), minimum_tokens=1) is None
    assert cache.put(compatibility, (8, 9), CacheBundle((FakeCache(16),)))
    stats = cache.stats()
    assert stats["bytes"] <= stats["capacity_bytes"]
    assert stats["evictions"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_execution_batch_validates_phase_and_budget() -> None:
    sequence = CausalLMRunner(
        SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda value: [1], bos_token_id=None),
            model=SimpleNamespace(make_cache=lambda: (FakeCache(0),)),
        )
    ).prepare(request())
    batch = ExecutionBatch((sequence,), "prefill", 512)
    assert batch.token_budget == 512
    with pytest.raises(ValueError, match="cannot be empty"):
        ExecutionBatch((), "prefill", 1)
    with pytest.raises(ValueError, match="positive"):
        ExecutionBatch((sequence,), "decode", 0)


def test_token_scheduler_prioritizes_decode_and_progresses_oldest_prefill() -> None:
    runner = CausalLMRunner(
        SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda value: [1, 2, 3], bos_token_id=None),
            model=SimpleNamespace(make_cache=lambda: (FakeCache(0),)),
        )
    )
    decoding = runner.prepare(request(request_id="decode"))
    decoding.transition(SequenceState.PREFILL)
    decoding.computed_tokens = 3
    decoding.generated_tokens.append(4)
    decoding.transition(SequenceState.DECODING)
    prefill = runner.prepare(request(request_id="prefill"))
    scheduler = TokenBudgetScheduler(max_batch_tokens=8, prefill_chunk_size=2)
    scheduler.active.append(decoding)
    scheduler.add(prefill)
    batches = scheduler.schedule()
    assert [batch.phase for batch in batches] == ["decode", "prefill"]
    assert batches[1].token_budget == 2


def test_token_scheduler_expires_and_cancels_requests() -> None:
    runner = CausalLMRunner(
        SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda value: [1], bos_token_id=None),
            model=SimpleNamespace(make_cache=lambda: (FakeCache(0),)),
        )
    )
    created = time.monotonic()
    expired = runner.prepare(request(request_id="expired", deadline=created + 1))
    cancelled = runner.prepare(request(request_id="cancelled"))
    cancelled.cancel.set()
    scheduler = TokenBudgetScheduler()
    scheduler.add(expired)
    scheduler.add(cancelled)
    assert scheduler.schedule(now=created + 2) == ()
    assert expired.state is SequenceState.TIMED_OUT
    assert cancelled.state is SequenceState.CANCELLED


def test_memory_planner_forecasts_and_distinguishes_queue_from_reject() -> None:
    gib = 1024**3
    budget = MemoryBudget.from_total(16 * gib)
    planner = UnifiedMemoryPlanner(budget)
    available = planner.available_bytes(MemoryUsage())
    assert planner.admit(available, MemoryUsage()).status is AdmissionStatus.ADMIT
    busy = MemoryUsage(model_bytes=available - 1)
    assert planner.admit(2, busy).status is AdmissionStatus.QUEUE
    assert planner.admit(available + 1, MemoryUsage()).status is AdmissionStatus.REJECT

    config = SimpleNamespace(
        num_hidden_layers=2,
        num_key_value_heads=2,
        num_attention_heads=4,
        hidden_size=16,
    )
    assert forecast_kv_bytes(config, 10) == 2 * 10 * 2 * 4 * 2 * 2
    assert forecast_kv_bytes(config, 10, key_bits=4, value_bits=8) < forecast_kv_bytes(
        config, 10
    )


def test_block_allocator_reference_counts_copy_on_write_and_capacity() -> None:
    allocator = BlockAllocator(24, block_tokens=4)
    first = allocator.allocate(FakeArray(8), 4)
    table = PageTable(allocator, [first])
    cloned = table.clone()
    assert allocator.get(first).references == 2
    replacement = allocator.copy_on_write(first, FakeArray(8), 3)
    cloned.block_ids[0] = replacement
    assert replacement != first
    assert allocator.get(first).references == 1
    assert table.token_count == 4
    assert cloned.token_count == 3
    with pytest.raises(MemoryError, match="pool is full"):
        allocator.allocate(FakeArray(20), 1)
    table.release()
    cloned.release()
    assert allocator.nbytes == 0
