"""Raw HTTP completion contracts, including queued/active stream cleanup."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_server import FakeEngineEngine, FakeManager

from mlx_one.server.app import ClientDisconnected, _until_disconnected, create_app
from mlx_one.server.config import ServerConfig


class CompletionManager(FakeManager):
    def current_model(self):
        # No chat template is needed for a raw completion.
        return SimpleNamespace(context_length=4096, chat_template=None,
                               tokenizer=SimpleNamespace(encode=lambda text: text.split()))


class CompletionEngine(FakeEngineEngine):
    closed = False

    def stream_completion(self, **settings):
        assert settings["prompt"] == "raw prompt"
        assert settings["options"].temperature == 0
        assert "messages" not in settings
        try:
            yield from self.stream(**settings)
        finally:
            self.closed = True


@pytest.mark.parametrize("streaming", [False, True])
def test_raw_completion_shape_usage_and_stream_cleanup(tmp_path, streaming):
    engine = CompletionEngine()
    with TestClient(create_app(CompletionManager(), engine, ui_dir=tmp_path)) as client:
        response = client.post("/v1/completions", json={
            "model": "test/model", "prompt": "raw prompt", "temperature": 0,
            "max_tokens": 1, "stream": streaming,
        })
        assert response.status_code == 200, response.text
        if streaming:
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            assert all(chunk["object"] == "text_completion" for chunk in chunks)
            assert "".join(c["choices"][0]["text"] for c in chunks) == "Hello"
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            result = response.json()
            assert result["object"] == "text_completion"
            assert result["choices"] == [{"index": 0, "text": "Hello", "logprobs": None,
                                          "finish_reason": "stop"}]
            assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 1,
                                       "total_tokens": 3}
    assert engine.closed


@pytest.mark.parametrize("update,status", [
    ({"model": "missing"}, 404), ({"prompt": ""}, 422), ({"prompt": ["a", "b"]}, 422),
    ({"max_tokens": 0}, 422), ({"max_tokens": 4096}, 400), ({"stop": [""]}, 422),
    ({"n": 2}, 422), ({"prompt": "p " * 4096}, 400),
])
def test_invalid_raw_requests_never_start_generation(tmp_path, update, status):
    engine = CompletionEngine()
    with TestClient(create_app(CompletionManager(), engine, ui_dir=tmp_path)) as client:
        result = client.post("/v1/completions", json={
            "model": "test/model", "prompt": "raw prompt", "temperature": 0,
            "max_tokens": 1, **update,
        })
        assert result.status_code == status, result.text
        assert "error" in result.json()
    assert not engine.closed


def test_raw_completion_uses_same_auth_policy(tmp_path):
    app = create_app(CompletionManager(), CompletionEngine(), ui_dir=tmp_path,
                     config=ServerConfig(api_keys=("secret",), warmup=False))
    with TestClient(app) as client:
        result = client.post("/v1/completions", json={"model": "test/model", "prompt": "x"})
        assert result.status_code == 401


def test_disconnect_cancels_a_pending_next_event_and_closes_upstream():
    async def run():
        closed = asyncio.Event()

        async def upstream():
            try:
                await asyncio.Event().wait()
                yield None
            finally:
                closed.set()

        async def disconnected():
            return True

        stream = _until_disconnected(SimpleNamespace(is_disconnected=disconnected), upstream())
        with pytest.raises(ClientDisconnected):
            await asyncio.wait_for(anext(stream), timeout=2)
        assert closed.is_set()

    asyncio.run(run())


def test_unicode_sse_survives_every_single_byte_transport_boundary(tmp_path):
    import codecs

    from mlx_one.server.generation_engine import GenerationEvent

    class UnicodeEngine(CompletionEngine):
        def stream(self, **settings):
            yield GenerationEvent(text="café")
            yield GenerationEvent(text=" 😀中文")
            yield GenerationEvent(finish_reason="stop", metrics={
                "prompt_tokens": 2, "generated_tokens": 2, "context_used": 4,
            })

    engine = UnicodeEngine()
    with TestClient(create_app(CompletionManager(), engine, ui_dir=tmp_path)) as client:
        with client.stream("POST", "/v1/completions", json={
            "model": "test/model", "prompt": "raw prompt", "temperature": 0,
            "max_tokens": 2, "stream": True,
        }) as response:
            assert response.status_code == 200
            decoder = codecs.getincrementaldecoder("utf-8")()
            text = "".join(decoder.decode(byte) for byte in response.iter_bytes(chunk_size=1))
            text += decoder.decode(b"", final=True)
    events = [json.loads(line[6:]) for line in text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(event["choices"][0]["text"] for event in events) == "café 😀中文"
    assert text.endswith("data: [DONE]\n\n") and engine.closed
