from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_one.server.transcription import AudioUploadError, NativeTranscriptionService


def test_native_transcription_uses_and_removes_a_safe_temporary_file(monkeypatch) -> None:
    observed: list[Path] = []

    def fake_transcribe(bundle, path, **kwargs):
        assert bundle is fake_bundle
        assert kwargs["task"] == "transcribe"
        source = Path(path)
        assert source.suffix == ".webm"
        assert source.read_bytes() == b"audio"
        observed.append(source)
        return {"text": "hello"}

    fake_bundle = SimpleNamespace()
    monkeypatch.setattr("mlx_one.server.transcription.shutil.which", lambda _: "/bin/ffmpeg")
    monkeypatch.setattr("mlx_one.server.transcription.transcribe", fake_transcribe)
    service = NativeTranscriptionService(fake_bundle)  # type: ignore[arg-type]
    assert service(b"audio", media_type="audio/webm;codecs=opus")["text"] == "hello"
    assert observed and not observed[0].exists()


def test_native_transcription_rejects_unsupported_audio(monkeypatch) -> None:
    monkeypatch.setattr("mlx_one.server.transcription.shutil.which", lambda _: "/bin/ffmpeg")
    service = NativeTranscriptionService(SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(AudioUploadError, match="unsupported"):
        service(b"audio", media_type="application/octet-stream")
