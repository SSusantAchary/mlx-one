from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_one.server.transcription import AudioUploadError, NativeTranscriptionService


def test_native_transcription_uses_and_removes_a_safe_temporary_file(monkeypatch) -> None:
    observed: list[Path] = []

    waveform = object()

    def fake_decode(path, **kwargs):
        assert kwargs["max_duration_seconds"] == 300
        assert kwargs["max_file_bytes"] == 25 * 1024**2
        assert kwargs["input_format"] == "matroska"
        source = Path(path)
        assert source.suffix == ".webm"
        assert source.read_bytes() == b"audio"
        observed.append(source)
        return waveform

    def fake_transcribe(bundle, audio, **kwargs):
        assert bundle is fake_bundle
        assert audio is waveform
        assert kwargs["task"] == "transcribe"
        assert kwargs["sample_rate"] == 16000
        return {"text": "hello"}

    fake_bundle = SimpleNamespace()
    monkeypatch.setattr("mlx_one.server.transcription.shutil.which", lambda _: "/bin/ffmpeg")
    monkeypatch.setattr("mlx_one.server.transcription.transcribe", fake_transcribe)
    monkeypatch.setattr("mlx_one.server.transcription.decode_audio", fake_decode)
    service = NativeTranscriptionService(fake_bundle)  # type: ignore[arg-type]
    assert service(b"audio", media_type="audio/webm;codecs=opus")["text"] == "hello"
    assert observed and not observed[0].exists()


def test_native_transcription_rejects_unsupported_audio(monkeypatch) -> None:
    monkeypatch.setattr("mlx_one.server.transcription.shutil.which", lambda _: "/bin/ffmpeg")
    service = NativeTranscriptionService(SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(AudioUploadError, match="unsupported"):
        service(b"audio", media_type="application/octet-stream")
