"""Real-checkpoint classification task, service, CLI and HTTP contracts."""

import json
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/distilbert-sst2"
pytestmark = pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="pinned local SST-2 checkpoint absent"
)


@pytest.fixture
def engine():
    from mlx_one.classification import ClassificationEngine

    with ClassificationEngine(CHECKPOINT) as engine:
        yield engine


def test_shared_service_registry_batch_limits_and_cleanup(engine):
    from mlx_one.core.registry import get_registration
    from mlx_one.engine.adapter_registry import get_runtime_registration

    assert get_registration("distilbert").tasks == frozenset({"text-classification"})
    assert get_runtime_registration("distilbert").integration_status == "integrated"
    text = "I enjoyed this film."
    single = engine.classify(text)
    batch = engine.classify([text, "A tedious and disappointing film."])
    assert single.predictions[0]["labels"][0]["label"] == "POSITIVE"
    assert batch.predictions[1]["labels"][0]["label"] == "NEGATIVE"
    for one, grouped in zip(
        single.predictions[0]["labels"], batch.predictions[0]["labels"], strict=True
    ):
        assert one["score"] == pytest.approx(grouped["score"], abs=1e-4, rel=1e-4)
    assert engine.classify(text).to_dict() == single.to_dict()
    short = engine.classify("word " * 600, max_length=8, top_k=1)
    assert short.predictions[0]["token_count"] == 8
    assert short.predictions[0]["truncated"]
    assert len(short.predictions[0]["labels"]) == 1
    assert engine.service.pool.stats()["active_leases"] == 0
    engine.service.unload("classification")
    assert engine.service.pool.stats()["resident_models"] == 0
    assert engine.classify(text).to_dict() == single.to_dict()


@pytest.mark.parametrize(
    "text", ["", "  \t\n", [], ["ok", "ok", "ok"], ["ok", 42], "\ud800", "a" * 32769]
)
def test_reject_invalid_input_and_recover(engine, text):
    with pytest.raises(ValueError):
        engine.classify(text)
    assert engine.service.pool.stats()["active_leases"] == 0
    assert engine.classify("Great.").predictions


def test_unicode_html_options_and_concurrent_tokenizer_isolation(engine):
    for text in ["Café — très bien!", "A fine movie 🎬", "<b>I loved it</b>"]:
        assert engine.classify(text).predictions[0]["token_count"] >= 2
    with pytest.raises(ValueError):
        engine.classify("ok", max_length=True)
    with pytest.raises(ValueError):
        engine.classify("ok", top_k=3)

    def call(limit):
        return engine.classify("word " * 100, max_length=limit).predictions[0]["token_count"]

    with ThreadPoolExecutor(4) as workers:
        assert list(workers.map(call, [8, 16, 8, 16])) == [8, 16, 8, 16]
    assert engine.service.pool.stats()["active_leases"] == 0


def test_cancel_deadline_and_timeout_cleanup(engine, monkeypatch):
    from mlx_one.engine.execution_context import RequestTimeoutError

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CancelledError):
        engine.classify("Good.", cancel=cancel)
    with pytest.raises(RequestTimeoutError):
        engine.classify("Good.", deadline=time.monotonic() - 1)
    actor, lease = engine.service._task_lease("classification", "text-classification")
    model = lease.runtime.value.model
    lease.release()
    original = type(model).__call__

    def slow(self, *args, **kwargs):
        time.sleep(0.03)
        return original(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(model), "__call__", slow)
        with pytest.raises(RequestTimeoutError):
            engine.classify("Good.", deadline=time.monotonic() + 0.01)
    assert engine.classify("Good.").predictions
    assert engine.service.pool.stats()["active_leases"] == 0


def test_python_cli_http_same_scores_and_errors(engine):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one.cli import main
    from mlx_one.server import ModelManager, ServerConfig, create_app

    text = "I loved this film."
    python = engine.classify(text).to_dict()
    cli = CliRunner().invoke(main, ["classify", str(CHECKPOINT), text, "--json-output"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == json.loads(json.dumps(python))
    manager = ModelManager(alias="sentiment")
    manager.load(CHECKPOINT, offline=True)
    config = ServerConfig(api_keys=("secret",), warmup=False, timeout=30)
    with TestClient(create_app(model_manager=manager, config=config)) as client:
        body = {"model": "sentiment", "input": text}
        assert client.post("/v1/tasks/text-classification", json=body).status_code == 401
        client.headers["Authorization"] = "Bearer secret"
        response = client.post("/v1/tasks/text-classification", json=body)
        assert response.status_code == 200, response.text
        assert response.json() == json.loads(json.dumps(python))
        assert (
            client.post(
                "/v1/tasks/text-classification", json={**body, "model": "missing"}
            ).status_code
            == 404
        )
        for value in ["", "  ", ["a", "b", "c"], 42]:
            response = client.post("/v1/tasks/text-classification", json={**body, "input": value})
            assert response.status_code == 422
            assert response.json()["error"]["type"] == "invalid_request_error"
        assert (
            client.post("/v1/tasks/text-classification", json={**body, "extra": 1}).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/tasks/text-classification",
                content="{bad",
                headers={"content-type": "application/json"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/tasks/text-classification", json={**body, "input": "x" * (1024 * 1024 + 1)}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/v1/completions", json={"model": "sentiment", "prompt": "Hello"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "sentiment", "messages": [{"role": "user", "content": "Hello"}]},
            ).status_code
            == 400
        )
        assert client.post("/v1/tasks/text-classification", json=body).json() == json.loads(
            json.dumps(python)
        )
    with pytest.raises(RuntimeError):
        manager.current_model()


def test_http_timeout_and_recovery(monkeypatch):
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from mlx_one.classification import ClassificationEngine
    from mlx_one.server import ModelManager, ServerConfig, create_app

    manager = ModelManager(alias="sentiment")
    bundle = manager.load(CHECKPOINT, offline=True)
    with ClassificationEngine(bundle.classification, timeout=60) as classifier:
        # Compile outside the timeout experiment, then introduce a bounded delay.
        classifier.classify("Great.")
        actor = classifier.service._task_actor
        actor.config = replace(actor.config, timeout=0.005)
        original = type(bundle.model).__call__

        def slow(self, *args, **kwargs):
            time.sleep(0.03)
            return original(self, *args, **kwargs)

        app = create_app(
            model_manager=manager,
            config=ServerConfig(warmup=False),
            task_services={"text-classification": classifier},
        )
        with TestClient(app) as client:
            with monkeypatch.context() as patch:
                patch.setattr(type(bundle.model), "__call__", slow)
                response = client.post(
                    "/v1/tasks/text-classification", json={"model": "sentiment", "input": "Great."}
                )
                assert response.status_code == 408
            actor = classifier.service._task_actor
            actor.config = replace(actor.config, timeout=60)
            assert (
                client.post(
                    "/v1/tasks/text-classification", json={"model": "sentiment", "input": "Great."}
                ).status_code
                == 200
            )
            assert classifier.service.pool.stats()["active_leases"] == 0


def test_http_disconnect_helper_signals_and_drains_worker():
    import asyncio

    from mlx_one.server.ocr import _run_request

    cancelled = threading.Event()
    drained = threading.Event()

    class Disconnected:
        async def is_disconnected(self):
            return True

    def worker():
        cancelled.wait(2)
        drained.set()

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await _run_request(Disconnected(), worker, cancelled)

    asyncio.run(run())
    assert cancelled.is_set() and drained.wait(2)
