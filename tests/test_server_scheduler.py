import asyncio
import threading

import pytest

from mlx_one.server.scheduler import GenerationScheduler, QueueFullError


def test_scheduler_streams_items_and_cancels_consumer() -> None:
    scheduler = GenerationScheduler()

    async def run() -> None:
        seen = []

        def generate(cancel: threading.Event):
            for value in range(3):
                if cancel.is_set():
                    return
                yield value

        async for value in scheduler.schedule(generate):
            seen.append(value)
        assert seen == [0, 1, 2]

    try:
        asyncio.run(run())
    finally:
        scheduler.close()


def test_scheduler_executes_jobs_in_fifo_order() -> None:
    scheduler = GenerationScheduler()
    active = threading.Event()
    release = threading.Event()
    order: list[int] = []

    def generate(value: int):
        def run(cancel: threading.Event):
            order.append(value)
            if value == 1:
                active.set()
                release.wait(timeout=2)
            if not cancel.is_set():
                yield value

        return run

    first = scheduler.schedule(generate(1))
    assert active.wait(timeout=1)
    second = scheduler.schedule(generate(2))
    third = scheduler.schedule(generate(3))
    release.set()

    async def collect(stream):
        return [item async for item in stream]

    try:
        assert asyncio.run(_gather(collect(first), collect(second), collect(third))) == [
            [1],
            [2],
            [3],
        ]
        assert order == [1, 2, 3]
    finally:
        scheduler.close()


def test_scheduler_rejects_more_than_eight_pending_jobs() -> None:
    scheduler = GenerationScheduler(max_pending=8)
    active = threading.Event()
    release = threading.Event()

    def blocked(cancel: threading.Event):
        active.set()
        release.wait(timeout=2)
        if not cancel.is_set():
            yield 0

    scheduler.schedule(blocked)
    assert active.wait(timeout=1)
    for _ in range(8):
        scheduler.schedule(lambda cancel: iter(()))
    with pytest.raises(QueueFullError):
        scheduler.schedule(lambda cancel: iter(()))
    release.set()
    scheduler.close()


def test_scheduler_cancels_abandoned_generation_and_isolates_failures() -> None:
    scheduler = GenerationScheduler()
    cancelled = threading.Event()

    def cancellable(cancel: threading.Event):
        yield "first"
        cancel.wait(timeout=2)
        if cancel.is_set():
            cancelled.set()

    async def abandon() -> None:
        stream = scheduler.schedule(cancellable)
        assert await anext(stream) == "first"
        await stream.aclose()
        assert await asyncio.to_thread(cancelled.wait, 1)

    def failing(cancel: threading.Event):
        del cancel
        raise ValueError("isolated")
        yield

    async def failure_then_success() -> None:
        failed = scheduler.schedule(failing)
        healthy = scheduler.schedule(lambda cancel: iter(("ok",)))
        with pytest.raises(ValueError, match="isolated"):
            await anext(failed)
        assert [item async for item in healthy] == ["ok"]

    try:
        asyncio.run(abandon())
        asyncio.run(failure_then_success())
    finally:
        scheduler.close()


async def _gather(*awaitables):
    return await asyncio.gather(*awaitables)
