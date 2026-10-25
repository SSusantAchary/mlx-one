"""Source-level audio contracts, input bounds and public interface validation."""

import io
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from mlx_one.audio import AudioInput, SpeechGenerationResult
from mlx_one.audio.input import prepare_audio, resample
from mlx_one.cli import main
from mlx_one.engine.contracts import TaskKind, canonical_task
from mlx_one.server import ServerConfig, create_app
from mlx_one.server.audio import NativeAudioService


def wav_bytes(samples=800, rate=8000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.zeros(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


@pytest.mark.parametrize("value", [np.zeros(1), np.zeros((100, 2)), np.zeros(100, dtype=np.int16)])
def test_input_rate_channel_and_duration(value):
    out = prepare_audio(value, sample_rate=8000)
    assert out.dtype == np.float32 and out.ndim == 1 and len(out) == len(value) * 2
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "value,rate",
    [
        (np.array([np.nan]), 16000),
        (np.array([np.inf]), 16000),
        (np.zeros((4, 4)), 16000),
        ([0, 1], 16000),
        (np.zeros(4), None),
        (np.zeros(4), True),
        (np.zeros(4), 200000),
    ],
)
def test_bad_waveforms_fail(value, rate):
    with pytest.raises(ValueError):
        prepare_audio(value, sample_rate=rate)


def test_sinc_downsampling_suppresses_aliases():
    time = np.arange(48000) / 48000
    low = resample(np.sin(2 * np.pi * 1000 * time), 48000, 16000)[100:-100]
    high = resample(np.sin(2 * np.pi * 12000 * time), 48000, 16000)[100:-100]
    assert np.sqrt(np.mean(low**2)) > 0.65
    assert np.sqrt(np.mean(high**2)) < 0.01


def test_input_bound_cancellation_and_invalid_mime():
    with pytest.raises(ValueError, match="duration"):
        prepare_audio(np.zeros(16001), sample_rate=16000, max_seconds=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        prepare_audio(np.zeros(1), sample_rate=16000, cancelled=lambda: True)
    with pytest.raises(ValueError, match="MIME"):
        prepare_audio(AudioInput(b"data", media_type="application/octet-stream"))
    with pytest.raises(ValueError, match="offset"):
        prepare_audio(AudioInput(np.zeros(1), 16000, offset_seconds=-1))


def test_speech_wav_header_actual_duration_and_no_overwrite(tmp_path):
    result = SpeechGenerationResult(np.zeros(2400), 24000, "hello", "/model", "a" * 40)
    assert result.duration == 0.1
    with wave.open(io.BytesIO(result.audio_bytes()), "rb") as handle:
        assert handle.getnframes() == 2400 and handle.getframerate() == 24000
    path = tmp_path / "output.wav"
    result.save(path)
    with pytest.raises(FileExistsError):
        result.save(path)
    with pytest.raises(ValueError):
        result.audio_bytes("mp3")
    invalid = SpeechGenerationResult(np.array([np.inf]), 24000, "", "/model", None)
    with pytest.raises(ValueError, match="finite"):
        invalid.audio_bytes()


@pytest.mark.parametrize(
    "alias,task",
    [
        ("audio.transcribe", TaskKind.TRANSCRIPTION),
        ("audio.align", TaskKind.ALIGNMENT),
        ("audio.generate_speech", TaskKind.SPEECH_GENERATION),
        ("audio.speech_to_speech", TaskKind.SPEECH_TO_SPEECH),
    ],
)
def test_task_aliases(alias, task):
    assert canonical_task(alias) == task


def test_audio_cli_help_and_validation_without_loading(tmp_path):
    runner = CliRunner()
    for command in ["transcribe", "align", "speak"]:
        result = runner.invoke(main, ["audio", command, "--help"])
        assert result.exit_code == 0 and "--offline" in result.output
    result = runner.invoke(
        main,
        [
            "audio",
            "speak",
            "--model",
            "/missing",
            "--output",
            str(tmp_path / "out.wav"),
            "--offline",
        ],
    )
    assert result.exit_code != 0 and "either --text" in result.output


class Manager:
    def model_info(self):
        return SimpleNamespace(id="audio-test", architecture="whisper", tasks=("transcription",))

    def current_model(self):
        raise RuntimeError("no automatic primary service in this route fixture")

    def unload(self):
        pass


class Service:
    family = "lfm2_audio"

    def __call__(self, data, **kwargs):
        return {"text": "fixture transcript", "duration": 1, "timings": {}}

    def speak(self, **kwargs):
        return SpeechGenerationResult(np.zeros(2400), 24000, "", "audio-test", "a" * 40)

    def align(self, data, **kwargs):
        return SimpleNamespace(to_dict=lambda: {"units": [], "unmatched_words": ["hello"]})


def client(services):
    return TestClient(
        create_app(
            model_manager=Manager(),
            task_services=services,
            config=ServerConfig(api_keys=("secret",), warmup=False, timeout=0),
        )
    )


def test_http_auth_model_selection_formats_and_unsupported_translation():
    with client({"transcription": Service()}) as c:
        assert c.post("/v1/audio/transcriptions").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        response = c.post(
            "/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
            data={"model": "wrong"},
        )
        assert response.status_code == 404
        for format in ["text", "json", "verbose_json"]:
            response = c.post(
                "/v1/audio/transcriptions",
                headers=headers,
                files={"file": ("a.wav", wav_bytes(), "audio/wav")},
                data={"model": "audio-test", "response_format": format},
            )
            assert response.status_code == 200 and "fixture transcript" in response.text
        response = c.post(
            "/v1/audio/translations",
            headers=headers,
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
            data={"model": "audio-test"},
        )
        assert response.status_code == 400


def test_http_speech_waveform_and_request_validation():
    with client({"speech": Service(), "alignment": Service()}) as c:
        headers = {"Authorization": "Bearer secret"}
        data = {"model": "audio-test", "input": "hello", "voice": "us_male"}
        response = c.post("/v1/audio/speech", headers=headers, json=data)
        assert response.status_code == 200 and response.content[:4] == b"RIFF"
        assert response.headers["x-audio-sample-rate"] == "24000"
        for update in [{"response_format": "mp3"}, {"speed": 2}, {"model": "wrong"}]:
            response = c.post("/v1/audio/speech", headers=headers, json={**data, **update})
            assert response.status_code in {400, 404}
        response = c.post(
            "/v1/audio/alignments",
            headers=headers,
            files={"file": ("a.wav", wav_bytes(), "audio/wav")},
            data={"model": "audio-test"},
        )
        assert response.status_code == 400


def test_http_media_validation_precedes_decoding():
    service = NativeAudioService(
        SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(model_type="whisper")))
    )
    with pytest.raises(ValueError, match="content type"):
        service.pcm(b"abc", "text/html", threading.Event())


def test_alignment_offsets_preserve_duration_and_unmatched_words():
    from mlx_one.audio.public import apply_offset
    from mlx_one.audio.schemas import AlignmentUnit, AudioAlignment

    local = AudioAlignment(
        (AlignmentUnit("hello", 0.1, 0.4),),
        "hello world",
        "English",
        1.0,
        "/fixture",
        "a" * 40,
        0.08,
        unmatched_words=("world",),
    )
    absolute = apply_offset(local, 12.0)
    assert absolute.units[0].start == 12.1 and absolute.units[0].end == 12.4
    assert absolute.duration == 1.0 and absolute.offset_seconds == 12.0
    assert absolute.unmatched_words == ("world",)
