import threading
from pathlib import Path

from fastapi.testclient import TestClient

from mlx_one.server.app import create_app
from mlx_one.server.generation_engine import GenerationEvent, RuntimeStats
from mlx_one.server.model_manager import ModelMetadata


class FakeBundle:
    model_id = "test/model"


class FakeManager:
    def current_model(self):
        return FakeBundle()

    def list_models(self):
        return (self.model_info(),)

    def model_info(self):
        return ModelMetadata("test/model", "qwen3", 4096, 10, {"bits": 4}, None)

    def unload(self):
        pass


class FakeEngineEngine:
    stats = RuntimeStats()

    def stream(self, **kwargs):
        cancel: threading.Event = kwargs["cancel"]
        if not cancel.is_set():
            yield GenerationEvent(text="Hello")
            yield GenerationEvent(
                finish_reason="stop",
                metrics={"prompt_tokens": 2, "generated_tokens": 1, "context_used": 3},
            )


class FailingEngine(FakeEngineEngine):
    def stream(self, **kwargs):
        del kwargs
        raise RuntimeError("generation failed")
        yield


def test_health_models_runtime_and_static_ui(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>mlx-one</h1>", encoding="utf-8")
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["runtime"] == "mlx"
        models = client.get("/v1/models").json()["data"]
        assert models[0]["id"] == "test/model"
        assert models[0]["mlx"]["quantization"] == {"bits": 4}
        assert client.get("/v1/runtime").json()["backend"] == "mlx"
        assert "mlx-one" in client.get("/").text
        assert "mlx-one" in client.get("/chat/anything").text


def test_non_streaming_and_streaming_chat(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "Hi"}]}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello"
        streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200
        assert '"content":"Hello"' in streamed.text
        assert "data: [DONE]" in streamed.text


def test_chat_validation_and_invalid_model(tmp_path: Path) -> None:
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    with TestClient(app) as client:
        invalid = client.post(
            "/v1/chat/completions",
            json={"model": "other", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert invalid.status_code == 404
        malformed = client.post(
            "/v1/chat/completions",
            json={"model": "test/model", "messages": [{"role": "tool", "content": "x"}]},
        )
        assert malformed.status_code == 422
        oversized = client.post(
            "/v1/chat/completions",
            content=b"x" * (1024 * 1024 + 1),
            headers={"content-type": "application/json"},
        )
        assert oversized.status_code == 413


def test_runtime_errors_use_openai_error_shape(tmp_path: Path) -> None:
    app = create_app(FakeManager(), FailingEngine(), ui_dir=tmp_path)
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "Hi"}]}
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "server_error"
        streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200
        assert '"type":"server_error"' in streamed.text
        assert "data: [DONE]" in streamed.text
