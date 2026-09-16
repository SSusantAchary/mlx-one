import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from mlx_one.engine.memory import AdmissionStatus
from mlx_one.server.app import create_app
from mlx_one.server.generation_engine import GenerationEvent, RuntimeStats
from mlx_one.server.model_manager import ModelMetadata


class FakeBundle:
    model_id = "test/model"
    chat_template = SimpleNamespace(render=lambda messages: str(messages))
    tokenizer = SimpleNamespace(encode=lambda prompt: list(range(len(prompt.split()))))


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


def embedding_service(texts, **kwargs):
    assert kwargs["input_type"] == "document"
    return {
        "embeddings": [[float(index), 1.0] for index, _ in enumerate(texts)],
        "dimensions": 2,
        "input_type": kwargs["input_type"],
        "prompt_tokens": len(texts),
    }


def rerank_service(query, documents, **kwargs):
    assert query == "q"
    assert kwargs["top_k"] == 1
    return {
        "items": [
            {"index": 0, "document": documents[0], "score": 1.0, "probability": 0.9}
        ]
    }


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
        assert response.status_code == 200, response.text
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


def test_additive_embedding_and_rerank_endpoints(tmp_path: Path) -> None:
    app = create_app(
        FakeManager(),
        FakeEngineEngine(),
        ui_dir=tmp_path,
        task_services={"embedding": embedding_service, "rerank": rerank_service},
    )
    with TestClient(app) as client:
        embedded = client.post(
            "/v1/embeddings",
            json={"model": "test/model", "input": ["one", "two"]},
        )
        assert embedded.status_code == 200
        assert embedded.json()["data"][1]["embedding"] == [1.0, 1.0]
        assert embedded.json()["mlx"]["dimensions"] == 2

        reranked = client.post(
            "/v1/rerank",
            json={
                "model": "test/model",
                "query": "q",
                "documents": ["document"],
                "top_n": 1,
            },
        )
        assert reranked.status_code == 200
        assert reranked.json()["results"][0]["document"] == "document"


def test_private_transcription_capability_and_endpoint(tmp_path: Path) -> None:
    class TranscriptionService:
        def capabilities(self):
            return {
                "enabled": True,
                "accepted_formats": ["audio/webm"],
                "max_bytes": 25 * 1024**2,
                "language_detection": True,
            }

        def __call__(self, audio, *, media_type, language, task, cancel):
            assert audio == b"browser audio"
            assert media_type == "audio/webm"
            assert language == "en"
            assert task == "transcribe"
            assert not cancel.is_set()
            return {"text": "hello from the microphone", "language": "en"}

    app = create_app(
        FakeManager(),
        FakeEngineEngine(),
        ui_dir=tmp_path,
        task_services={"transcription": TranscriptionService()},
    )
    with TestClient(app) as client:
        runtime = client.get("/v1/runtime").json()
        assert runtime["transcription"]["enabled"] is True
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/model", "language": "en"},
            files={"file": ("recording.webm", b"browser audio", "audio/webm")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "hello from the microphone"


def test_transcription_rejects_invalid_model_and_missing_capability(tmp_path: Path) -> None:
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    with TestClient(app) as client:
        assert client.get("/v1/runtime").json()["transcription"]["enabled"] is False
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "test/model"},
            files={"file": ("recording.webm", b"audio", "audio/webm")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "capability_error"


def test_multimodal_chat_requires_vision_service(tmp_path: Path) -> None:
    payload = {
        "model": "test/model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ],
    }
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "capability_error"


def test_memory_admission_rejects_before_generation(tmp_path: Path) -> None:
    class Admission:
        def stats(self):
            return {"process_bytes": 1}

        def decide(self, bundle, tokens, **kwargs):
            del bundle, tokens, kwargs
            return SimpleNamespace(
                status=AdmissionStatus.REJECT,
                reason="request exceeds the configured unified-memory capacity",
            )

    app = create_app(
        FakeManager(),
        FakeEngineEngine(),
        ui_dir=tmp_path,
        memory_admission=Admission(),  # type: ignore[arg-type]
    )
    payload = {"model": "test/model", "messages": [{"role": "user", "content": "Hi"}]}
    with TestClient(app) as client:
        assert client.get("/v1/runtime").json()["memory_budget"]["process_bytes"] == 1
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "capacity_error"
