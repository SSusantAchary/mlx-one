"""HTTP metadata and bounded ASR service errors with no model execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from test_server import FakeEngineEngine, FakeManager

from mlx_one import __version__
from mlx_one.models.audio.whisper.processing import AudioProcessingError
from mlx_one.server.app import create_app
from mlx_one.server.transcription import AudioUploadError, NativeTranscriptionService


def test_openapi_preserves_alpha_version_and_existing_route_scope(tmp_path):
    app = create_app(FakeManager(), FakeEngineEngine(), ui_dir=tmp_path)
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["version"] == __version__
        assert {"/v1/chat/completions", "/v1/embeddings", "/v1/rerank",
                "/v1/audio/transcriptions", "/v1/models"} <= schema["paths"].keys()
        assert "/v1/completions" in schema["paths"]


@pytest.mark.parametrize("failure", ["duration limit", "malformed codec", "timed out"])
def test_decoder_failure_removes_upload_and_never_invokes_model(monkeypatch, failure):
    observed = []

    def decode(path, **kwargs):
        observed.append(Path(path))
        assert kwargs["max_duration_seconds"] == 300
        assert kwargs["timeout_seconds"] == 120
        raise AudioProcessingError(failure)

    monkeypatch.setattr("mlx_one.server.transcription.shutil.which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr("mlx_one.server.transcription.decode_audio", decode)
    monkeypatch.setattr("mlx_one.server.transcription.transcribe",
                        lambda *a, **kw: pytest.fail("unsafe audio reached model"))
    service = NativeTranscriptionService(SimpleNamespace())
    with pytest.raises(AudioUploadError, match=failure):
        service(b"invalid audio", media_type="audio/wav")
    assert observed and not observed[0].exists()
