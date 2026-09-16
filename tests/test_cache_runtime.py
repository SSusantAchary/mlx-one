import asyncio
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from mlx_one.core.cache import ConvCache, KVCache
from mlx_one.engine import (
    BlockKVCache,
    BlockPool,
    CacheConfig,
    CacheCorruptionError,
    CacheKind,
    CacheManager,
    CacheOwnershipError,
    CacheReservationError,
    CacheTopologyUnsupported,
    ReservationState,
    build_cache_plan,
)
from mlx_one.server.scheduler import GenerationScheduler


def install_fake_mlx(monkeypatch) -> None:
    package = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.zeros = np.zeros
    core.concatenate = np.concatenate
    core.array = np.array
    core.float16 = np.float16
    core.bfloat16 = np.float32
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, value):
        return list(range(1, len(value) + 1))

    def decode(self, values):
        return "".join(chr(96 + int(value) % 26) for value in values)


class Model:
    def __init__(self, caches=1):
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=caches,
            num_key_value_heads=2,
            num_attention_heads=4,
            hidden_size=16,
            max_position_embeddings=128,
            rope_theta=10000.0,
        )
        self.caches = caches

    def make_cache(self):
        return tuple(KVCache() for _ in range(self.caches))


def bundle(tmp_path: Path, *, model=None, architecture="qwen2"):
    return SimpleNamespace(
        model=model or Model(),
        tokenizer=Tokenizer(),
        path=tmp_path,
        model_id="test/model",
        revision="revision",
        architecture=architecture,
        context_length=128,
        chat_template=SimpleNamespace(template="{{ messages }}"),
    )


def test_cache_plan_maps_attention_and_hybrid_topologies(tmp_path) -> None:
    plan = build_cache_plan(bundle(tmp_path))
    assert plan.layer_specs[0].kind is CacheKind.ATTENTION_KV
    assert plan.layer_specs[0].bytes_per_token() == 32
    assert plan.block_compatible

    hybrid = Model()
    hybrid.make_cache = lambda: (KVCache(), ConvCache(3))
    hybrid.config.num_hidden_layers = 2
    hybrid_plan = build_cache_plan(bundle(tmp_path, model=hybrid, architecture="lfm2"))
    assert [item.kind for item in hybrid_plan.layer_specs] == [
        CacheKind.ATTENTION_KV,
        CacheKind.CONV_STATE,
    ]
    assert not hybrid_plan.block_compatible
    assert not hybrid_plan.prefix_shareable


def test_manager_owns_dense_handles_accounting_and_reservations(tmp_path, monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    manager = CacheManager(
        bundle(tmp_path),
        CacheConfig(memory_budget_bytes=1024 * 1024, prefix_cache_budget_bytes=4096),
    )
    handle = manager.create_handle("request")
    values = np.ones((1, 2, 3, 4), dtype=np.float16)
    handle.bundle.entries[0].update(values, values)
    assert handle.bundle.nbytes == values.nbytes * 2
    assert handle.bundle.allocated_nbytes > handle.bundle.nbytes
    reservation = manager.reserve(handle, tokens=4, requested_bytes=128)
    assert reservation.state is ReservationState.GRANTED
    manager.commit(reservation, 64)
    assert reservation.state is ReservationState.COMMITTED
    assert manager.stats().reserved_bytes == 64
    manager.release(handle)
    assert manager.stats().active_handles == 0
    with pytest.raises(CacheOwnershipError):
        manager.release(handle)
    with pytest.raises(CacheReservationError):
        manager.release_reservation(reservation)


def test_block_pool_pins_reuses_and_rejects_invalid_release() -> None:
    pool = BlockPool(256, block_tokens=16)
    state = np.zeros((1, 1, 16, 2), dtype=np.float16)
    reference = pool.allocate_ref((state, state.copy()), 16)
    reference.pin()
    block_id = reference.block_id
    reference.close()
    assert pool.get(block_id).references == 0
    assert pool.stats()["block_prefix"] == 1
    pool.unpin(block_id)
    assert pool.stats()["block_free"] == 1
    reused = pool.allocate_ref((state.copy(), state.copy()), 1)
    assert reused.block_id == block_id
    replacement = np.ones_like(state)
    reused.close()
    reused = pool.allocate_ref((replacement, replacement.copy()), 1)
    assert np.array_equal(reused.block.state[0], replacement)
    reused.close()
    with pytest.raises(CacheCorruptionError):
        reused.close()


def test_block_kv_growth_materialization_clone_and_trim(monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    pool = BlockPool(4096, block_tokens=16)
    cache = BlockKVCache(pool)
    first = np.arange(20 * 4, dtype=np.float16).reshape(1, 1, 20, 4)
    keys, values = cache.update(first, first)
    assert cache.offset == 20
    assert len(cache.full_block_refs) == 1
    assert np.array_equal(keys, first)
    assert np.array_equal(values, first)
    clone = cache.clone()
    clone.update(np.ones((1, 1, 1, 4), dtype=np.float16), np.ones((1, 1, 1, 4)))
    assert clone.offset == 21
    assert cache.offset == 20
    assert clone.trim(5)
    assert clone.offset == 16
    cache.release()
    clone.release()


def test_block_prefix_survives_donor_and_borrower_lifecycles(tmp_path, monkeypatch) -> None:
    install_fake_mlx(monkeypatch)
    manager = CacheManager(
        bundle(tmp_path),
        CacheConfig(
            backend="block",
            memory_budget_bytes=1024 * 1024,
            prefix_cache_budget_bytes=1024 * 1024,
            block_size_tokens=16,
        ),
    )
    fingerprint = manager.fingerprint()
    tokens = tuple(range(40))
    donor = manager.create_handle("donor")
    values = np.arange(40 * 8, dtype=np.float16).reshape(1, 2, 40, 4)
    donor.bundle.entries[0].update(values, values)
    assert manager.publish_prefix(donor, fingerprint, tokens)

    borrower = manager.create_handle("borrower")
    assert manager.adopt_prefix(borrower, fingerprint, tokens) == 32
    mismatch = manager.create_handle("mismatch")
    assert manager.adopt_prefix(
        mismatch, replace(fingerprint, adapter_identity="different"), tokens
    ) == 0
    manager.release(mismatch)
    manager.release(donor)
    restored = borrower.bundle.entries[0].state()[0]
    assert np.array_equal(restored, values[:, :, :32, :])
    borrower.bundle.entries[0].update(
        values[:, :, 32:33, :], values[:, :, 32:33, :]
    )
    manager.release(borrower)
    assert manager.stats().prefix_entries > 0
    manager.clear_prefixes()
    assert manager.stats().prefix_entries == 0


def test_block_pool_deterministic_stress_balances_references() -> None:
    rng = random.Random(7)
    pool = BlockPool(4096, block_tokens=16)
    state = np.zeros((1, 1, 16, 2), dtype=np.float16)
    references = []
    for _ in range(500):
        if not references or (len(references) < 8 and rng.random() < 0.6):
            references.append(
                pool.allocate_ref((state.copy(), state.copy()), rng.randint(1, 16))
            )
        else:
            references.pop(rng.randrange(len(references))).close()
    for reference in references:
        reference.close()
    assert pool.stats()["block_live"] == 0
    pool.clear()
    assert pool.stats()["block_total"] == 0


def test_explicit_block_backend_rejects_unqualified_model(tmp_path) -> None:
    manager = CacheManager(
        bundle(tmp_path, architecture="qwen3"),
        CacheConfig(backend="block", memory_budget_bytes=4096),
    )
    with pytest.raises(CacheTopologyUnsupported):
        manager.create_handle("request")


def test_scheduler_retries_queued_admission_before_execution() -> None:
    scheduler = GenerationScheduler(timeout=1)
    attempts = 0

    def admission():
        nonlocal attempts
        attempts += 1
        return "queue" if attempts == 1 else "admit"

    async def collect():
        return [
            item
            async for item in scheduler.schedule(
                lambda _: iter(("ok",)), admission=admission
            )
        ]

    try:
        assert asyncio.run(collect()) == ["ok"]
        assert attempts == 2
    finally:
        scheduler.close()
