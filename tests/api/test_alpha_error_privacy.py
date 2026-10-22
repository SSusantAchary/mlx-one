"""Backend exceptions must not expose private paths or credentials to clients."""

import pytest
from fastapi.testclient import TestClient
from test_server import FakeEngineEngine, FakeManager

from mlx_one.server.app import create_app


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("after_first", [False, True])
def test_unexpected_chat_error_redacts_details_and_recovers(tmp_path, streaming, after_first):
    marker = "PRIVATE_SENTINEL /Users/private/checkpoint credential=not-a-real-secret"

    class Engine(FakeEngineEngine):
        fail = True

        def stream(self, **kwargs):
            if self.fail:
                if after_first:
                    yield next(super().stream(**kwargs))
                raise RuntimeError(marker)
            yield from super().stream(**kwargs)

    engine = Engine()
    body = {"model": "test/model", "messages": [{"role": "user", "content": "Hello"}],
            "stream": streaming}
    with TestClient(create_app(FakeManager(), engine, ui_dir=tmp_path)) as client:
        result = client.post("/v1/chat/completions", json=body)
        assert result.status_code == (200 if streaming else 500)
        assert "internal server error" in result.text
        assert "PRIVATE_SENTINEL" not in result.text and "credential=" not in result.text
        assert "/Users/private" not in result.text and "Traceback" not in result.text
        if streaming:
            assert result.text.count("data: [DONE]") == 1
        engine.fail = False
        result = client.post("/v1/chat/completions", json={**body, "stream": False})
        assert result.status_code == 200


def test_500_error_helper_redacts_all_task_handlers():
    from mlx_one.server.app import _error

    assert _error(500, "private artifact details", "server_error").body == (
        b'{"error":{"message":"internal server error","type":"server_error","code":500}}')
