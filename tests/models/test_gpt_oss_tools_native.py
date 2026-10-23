"""Opt-in tiny native-model tool lifecycle tests; never downloads weights."""

import os
import threading
from dataclasses import replace
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from fastapi.testclient import TestClient
from test_gpt_oss_tools import ARGS, CALL, FINAL, TOOL, USER, Manager, Tokenizer, chunks
from test_large_moe_contracts import tiny_config

from mlx_one.core.registry import get_registration
from mlx_one.server.app import create_app
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEngine
from mlx_one.server.model_manager import ModelMetadata
from mlx_one.server.scheduler import GenerationScheduler
from mlx_one.text import TextGenerationOptions
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.loading import LoadedTextModel


@pytest.fixture
def native(tmp_path, monkeypatch):
    scheduler = GenerationScheduler()

    def build():
        mx.random.seed(11)
        config = replace(tiny_config("gpt_oss"), vocab_size=512, max_position_embeddings=4096)
        model = get_registration("gpt_oss").model_class()(config)
        tokenizer = Tokenizer()
        tokenizer.special = {key: 256 + i for i, key in enumerate(tokenizer.special)}
        tokenizer.inverse = {value: key for key, value in tokenizer.special.items()}
        tokenizer.eos_token_id = tokenizer.special["<|return|>"]
        bundle = LoadedTextModel(
            model,
            tokenizer,
            Path(tmp_path),
            "test-alias",
            None,
            architecture="gpt_oss",
            context_length=4096,
            eos_token_ids=(tokenizer.eos_token_id, tokenizer.special["<|call|>"]),
            chat_template=ChatTemplate("gpt_oss", None, tokenizer),
        )
        model.eval()
        mx.eval(model.parameters())
        mx.eval(model(mx.array([[1]])).logits)
        manager = Manager()
        manager.current_model = lambda: bundle
        manager.model_info = lambda: ModelMetadata("test-alias", "gpt_oss", 4096, 1, {}, None)
        engine = GenerationEngine(
            manager, prefix_cache_bytes=0, prefill_chunk_size=64, inference_mode="reference_dense"
        )
        return manager, engine, tokenizer

    manager, engine, tokenizer = scheduler.execute(build)

    def script(text):
        tokens = iter(tokenizer.encode(text))
        monkeypatch.setattr("mlx_one.text.generation.select_token", lambda *a, **kw: next(tokens))

    yield manager, engine, script, scheduler
    scheduler.close(finalizer=engine.close)


def test_native_tool_round_trips_and_cache_cleanup(native, tmp_path):
    manager, engine, script, scheduler = native
    app = create_app(
        manager,
        engine,
        ui_dir=tmp_path,
        scheduler=scheduler,
        config=ServerConfig(warmup=False, cache_prompt=False, inference_mode="reference_dense"),
    )
    with TestClient(app) as client:
        for _ in range(3):
            script(CALL)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-alias",
                    "messages": [USER],
                    "tools": [TOOL],
                    "max_tokens": 512,
                },
            )
            assert response.status_code == 200, response.text
            data = response.json()["choices"][0]
            assert data["finish_reason"] == "tool_calls"
            call = data["message"]["tool_calls"][0]
            assert call["function"]["arguments"] == ARGS
            assert engine.cache_stats()["active_handles"] == 0
            script(FINAL)
            history = [
                USER,
                data["message"],
                {"role": "tool", "tool_call_id": call["id"], "content": "File written."},
            ]
            result = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-alias",
                    "messages": history,
                    "tools": [TOOL],
                    "max_tokens": 512,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            assert result.status_code == 200
            text = "".join(
                choice["delta"].get("content", "")
                for event in chunks(result)
                for choice in event["choices"]
            )
            assert text == "Done."
            assert "private analysis" not in result.text
            assert "data: [DONE]" in result.text
            assert engine.cache_stats()["active_handles"] == 0


def test_native_cancel_and_malformed_output_release_state(native, monkeypatch):
    from mlx_one.text.harmony_tools import HarmonyToolParser

    partial_arguments = []
    original_feed = HarmonyToolParser.feed

    def observe_partial(self, token):
        result = original_feed(self, token)
        if not self.in_header and self.recipient and self.body_ids:
            partial_arguments.append(True)
        return result

    monkeypatch.setattr(HarmonyToolParser, "feed", observe_partial)
    manager, engine, script, scheduler = native
    script(CALL)
    cancel = threading.Event()
    stream = engine.stream(
        model="test-alias",
        messages=[USER],
        tools=[TOOL],
        options=TextGenerationOptions(max_tokens=512),
        cancel=cancel,
        context_length=4096,
    )
    for _ in range(512):
        assert not scheduler.execute(lambda: next(stream)).tool_calls
        if partial_arguments:
            break
    assert partial_arguments, "cancellation must occur during partial tool JSON parsing"
    cancel.set()
    scheduler.execute(lambda: list(stream))
    scheduler.execute(stream.close)
    assert engine.cache_stats()["active_handles"] == 0
    script(CALL.replace(ARGS, '{"path":'))
    with pytest.raises(ValueError, match="JSON"):
        scheduler.execute(
            lambda: list(
                engine.stream(
                    model="test-alias",
                    messages=[USER],
                    tools=[TOOL],
                    options=TextGenerationOptions(max_tokens=512),
                    cancel=threading.Event(),
                    context_length=4096,
                )
            )
        )
    assert engine.cache_stats()["active_handles"] == 0
    script(CALL)
    healthy = scheduler.execute(
        lambda: list(
            engine.stream(
                model="test-alias",
                messages=[USER],
                tools=[TOOL],
                options=TextGenerationOptions(max_tokens=512),
                cancel=threading.Event(),
                context_length=4096,
            )
        )
    )
    assert healthy[-1].finish_reason == "tool_calls"
    assert engine.cache_stats()["active_handles"] == 0


def test_native_text_stops_do_not_modify_tool_arguments(native, monkeypatch):
    monkeypatch.setattr(
        "mlx_one.text.harmony.final_text",
        lambda *args, **kwargs: pytest.fail("tool requests must use token parsing"),
    )
    _, engine, script, scheduler = native
    script(CALL)
    events = scheduler.execute(
        lambda: list(
            engine.stream(
                model="test-alias",
                messages=[USER],
                tools=[TOOL],
                options=TextGenerationOptions(max_tokens=512, stop=("hello",)),
                cancel=threading.Event(),
                context_length=4096,
            )
        )
    )
    assert events[-1].finish_reason == "tool_calls"
    arguments = "".join(
        delta["function"]["arguments"] for event in events for delta in event.tool_calls or []
    )
    assert arguments == ARGS
    script("<|channel|>final<|message|>helloSTOPignored<|return|>")
    events = scheduler.execute(
        lambda: list(
            engine.stream(
                model="test-alias",
                messages=[USER],
                tools=[TOOL],
                options=TextGenerationOptions(max_tokens=512, stop=("STOP",)),
                cancel=threading.Event(),
                context_length=4096,
            )
        )
    )
    assert "".join(event.text for event in events) == "hello"
    assert events[-1].finish_reason == "stop"
    assert engine.cache_stats()["active_handles"] == 0
