import asyncio
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from mlx_one.cli import main
from mlx_one.server.app import create_app
from mlx_one.server.config import ServerConfig, cache_type_from_bits
from mlx_one.server.generation_engine import (
    GenerationEvent,
    RuntimeStats,
    _fit_messages,
    _ReasoningParser,
)
from mlx_one.server.model_manager import ModelMetadata
from mlx_one.server.scheduler import GenerationScheduler, RequestTimeoutError


class _Bundle:
    model_id = "source/model"
    context_length = 128


class _Manager:
    def current_model(self):
        return _Bundle()

    def model_info(self):
        return ModelMetadata("public-name", "qwen3", 128, 10, {}, None)

    def list_models(self):
        return (self.model_info(),)

    def unload(self):
        pass


class _Engine:
    stats = RuntimeStats()

    def stream(self, **kwargs):
        del kwargs
        yield GenerationEvent(text="ok")
        yield GenerationEvent(finish_reason="stop", metrics={})


def test_server_config_and_cli_expose_native_controls() -> None:
    assert cache_type_from_bits(4) == "q4_0"
    assert ServerConfig().timeout == 600
    with pytest.raises(ValueError, match="timeout"):
        ServerConfig(timeout=-1)
    result = CliRunner().invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--alias",
        "--api-key",
        "--context-length",
        "--queue-size",
        "--parallel",
        "--reasoning-budget",
        "--cache-prompt",
        "--context-shift",
        "--kv-cache-bits",
        "--spec-type",
    ):
        assert flag in result.output


def test_api_key_protects_v1_but_not_health_or_ui(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ui", encoding="utf-8")
    config = ServerConfig(api_keys=("secret",), warmup=False, cache_prompt=False, timeout=0)
    app = create_app(_Manager(), _Engine(), ui_dir=tmp_path, config=config)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        denied = client.get("/v1/models")
        assert denied.status_code == 401
        assert denied.json()["error"]["type"] == "authentication_error"
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        payload = {
            "model": "public-name",
            "messages": [{"role": "user", "content": "hello"}],
        }
        assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200


def test_scheduler_rotates_parallel_jobs_and_enforces_deadline() -> None:
    scheduler = GenerationScheduler(parallel=2, timeout=0)
    order: list[str] = []

    def values(name: str):
        def generate(cancel: threading.Event):
            del cancel
            for index in range(2):
                order.append(f"{name}{index}")
                yield index

        return generate

    async def collect():
        return await asyncio.gather(
            _collect(scheduler.schedule(values("a"))),
            _collect(scheduler.schedule(values("b"))),
        )

    try:
        assert asyncio.run(collect()) == [[0, 1], [0, 1]]
        assert order == ["a0", "b0", "a1", "b1"]
    finally:
        scheduler.close()

    expired = GenerationScheduler(timeout=0.01)

    def slow(cancel: threading.Event):
        del cancel
        yield 1
        time.sleep(0.02)
        yield 2

    async def timeout():
        with pytest.raises(RequestTimeoutError):
            await _collect(expired.schedule(slow))

    try:
        asyncio.run(timeout())
    finally:
        expired.close()


def test_scheduler_lifecycle_call_is_a_barrier() -> None:
    scheduler = GenerationScheduler(parallel=1, timeout=0)
    order: list[str] = []
    first_step = threading.Event()
    release_first = threading.Event()

    def first(cancel: threading.Event):
        del cancel
        order.append("first-0")
        first_step.set()
        yield 0
        release_first.wait(1)
        order.append("first-1")
        yield 1

    def second(cancel: threading.Event):
        del cancel
        order.append("second")
        yield 2

    async def collect_with_barrier():
        first_result = asyncio.create_task(_collect(scheduler.schedule(first)))
        await asyncio.to_thread(first_step.wait, 1)
        barrier = asyncio.create_task(
            asyncio.to_thread(scheduler.execute, lambda: order.append("barrier"))
        )
        while scheduler._jobs.qsize() == 0 and scheduler._deferred_call is None:
            await asyncio.sleep(0)
        second_result = asyncio.create_task(_collect(scheduler.schedule(second)))
        release_first.set()
        return await asyncio.gather(first_result, barrier, second_result)

    try:
        assert asyncio.run(collect_with_barrier()) == [[0, 1], None, [2]]
        assert order == ["first-0", "first-1", "barrier", "second"]
    finally:
        scheduler.close()


async def _collect(stream):
    return [item async for item in stream]


def test_reasoning_parser_separates_incremental_thoughts() -> None:
    parser = _ReasoningParser("deepseek", False, -1)
    parts = [parser.feed(value) for value in ("<thi", "nk>work", "</think>answer")]
    parts.append(parser.flush())
    assert "".join(part[0] for part in parts) == "answer"
    assert "".join(part[1] for part in parts) == "work"


def test_legacy_reasoning_keeps_tags_and_also_separates_thoughts() -> None:
    parser = _ReasoningParser(
        "deepseek-legacy", False, -1, start_in_reasoning=True
    )
    first = parser.feed("work</think>answer")
    trailing = parser.flush()
    assert first[0] + trailing[0] == "<think>work</think>answer"
    assert first[1] + trailing[1] == "work"


def test_context_shift_drops_oldest_complete_turn() -> None:
    class Tokenizer:
        def encode(self, text):
            return list(text)

    class Template:
        def render(self, messages, **kwargs):
            del kwargs
            return "|".join(item["content"] for item in messages)

    class Bundle:
        tokenizer = Tokenizer()
        chat_template = Template()

    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "new"},
    ]
    selected, prompt = _fit_messages(Bundle(), messages, 2, 8, True, "auto")
    assert [item["content"] for item in selected] == ["S", "new"]
    assert prompt == "S|new"
