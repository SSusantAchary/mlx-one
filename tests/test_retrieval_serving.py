"""Real serving integration contracts without downloading checkpoints."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_server import FakeEngineEngine, FakeManager

from mlx_one.server import create_app
from mlx_one.server.model_manager import _load_serving_model
from mlx_one.server.retrieval import RetrievalServingBundle


@pytest.mark.parametrize("task", ["embedding", "reranking"])
def test_explicit_serving_task_loads_retrieval_not_language_model(tmp_path, monkeypatch, task):
    from mlx_one.retrieval import loading

    calls = []
    bundle = SimpleNamespace(
        task=task,
        architecture="bert",
        model=SimpleNamespace(config=SimpleNamespace(sentence_max_length=256)),
    )
    monkeypatch.setattr(
        loading, "load_retrieval_model", lambda *a, **kw: calls.append(kw) or bundle
    )
    loaded = _load_serving_model(tmp_path, retrieval_task=task, offline=True)
    assert loaded.retrieval is bundle and loaded.context_length == 256
    assert calls == [dict(task=task, offline=True)]


def test_sentence_transformer_metadata_routes_without_task_override(tmp_path, monkeypatch):
    from mlx_one.retrieval import loading

    (tmp_path / "config.json").write_text('{"model_type":"bert"}')
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling/config.json").write_text("{}")
    (tmp_path / "modules.json").write_text(
        json.dumps([dict(type="sentence_transformers.models.Pooling")])
    )
    bundle = SimpleNamespace(architecture="bert", task="embedding")
    monkeypatch.setattr(loading, "load_retrieval_model", lambda *a, **kw: bundle)
    assert _load_serving_model(tmp_path).retrieval is bundle


@pytest.mark.parametrize("task", ["embedding", "reranking"])
def test_native_retrieval_services_are_automatically_wired_and_hold_lease(monkeypatch, task):
    from mlx_one import retrieval

    class Manager(FakeManager):
        active = 0

        def current_model(self):
            return RetrievalServingBundle(SimpleNamespace(task=task))

        @contextmanager
        def model_lease(self):
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1

    manager = Manager()

    def operation(bundle, *args, **kwargs):
        assert manager.active == 1 and bundle.task == task
        return (
            dict(embeddings=[[1.0, 2.0]], dimensions=2)
            if task == "embedding"
            else dict(items=[dict(index=0, document="document", score=1.0, probability=0.9)])
        )

    monkeypatch.setattr(retrieval, "embed" if task == "embedding" else "rerank", operation)
    with TestClient(create_app(manager, FakeEngineEngine())) as client:
        response = client.post(
            "/v1/embeddings" if task == "embedding" else "/v1/rerank",
            json=dict(model="test/model", input="hello")
            if task == "embedding"
            else dict(model="test/model", query="query", documents=["document"]),
        )
        assert response.status_code == 200, response.text
    assert manager.active == 0
