"""Decoder safety paths execute with fake subprocesses and no Metal."""

import struct
import subprocess
from types import SimpleNamespace

import pytest

from mlx_one.models.audio.whisper import processing
from mlx_one.models.audio.whisper.processing import AudioProcessingError, decode_audio


def decoder(monkeypatch, tmp_path, output=b"", *, hangs=False):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"fixture")
    state = SimpleNamespace(returncode=0, terminated=False, commands=[])

    def communicate(**kwargs):
        if hangs and not state.terminated:
            raise subprocess.TimeoutExpired("ffmpeg", kwargs["timeout"])
        return output, b""

    state.communicate = communicate
    state.terminate = lambda: setattr(state, "terminated", True)
    state.kill = state.terminate

    def popen(command, **kwargs):
        state.commands.append(command)
        return state

    monkeypatch.setattr(processing.shutil, "which", lambda _: "/fake/ffmpeg")
    monkeypatch.setattr(processing.subprocess, "Popen", popen)
    return source, state


def test_duration_limit_rejects_instead_of_silently_truncating(monkeypatch, tmp_path):
    source, state = decoder(monkeypatch, tmp_path, struct.pack("<3f", 0, 0, 0))
    with pytest.raises(AudioProcessingError, match="duration limit"):
        decode_audio(source, max_duration_seconds=2 / 16000, input_format="wav")
    command = state.commands[0]
    assert "file,pipe" in command  # no network protocols for nested media
    assert float(command[command.index("-t") + 1]) == 3 / 16000
    assert command[command.index("-i") - 2:command.index("-i")] == ["-f", "wav"]


@pytest.mark.parametrize("output", [b"", b"abc", struct.pack("<f", float("nan"))])
def test_bad_pcm_rejected_without_mlx(monkeypatch, tmp_path, output):
    source, _ = decoder(monkeypatch, tmp_path, output)
    with pytest.raises(AudioProcessingError, match="empty|malformed|non-finite"):
        decode_audio(source)


def test_cancellation_and_timeout_terminate_decoder(monkeypatch, tmp_path):
    source, state = decoder(monkeypatch, tmp_path, hangs=True)
    with pytest.raises(AudioProcessingError, match="cancelled"):
        decode_audio(source, cancelled=lambda: True)
    assert state.terminated
    state.terminated = False
    ticks = iter([0, 2])
    monkeypatch.setattr(processing.time, "monotonic", lambda: next(ticks))
    with pytest.raises(AudioProcessingError, match="timed out"):
        decode_audio(source, timeout_seconds=1)
    assert state.terminated


@pytest.mark.parametrize("kwargs", [{"max_duration_seconds": float("nan")},
    {"timeout_seconds": 0}, {"max_file_bytes": True}, {"max_file_bytes": 1},
    {"input_format": "hls"}])
def test_invalid_limits_or_large_file_fail_before_process(monkeypatch, tmp_path, kwargs):
    source, state = decoder(monkeypatch, tmp_path)
    with pytest.raises(AudioProcessingError):
        decode_audio(source, **kwargs)
    assert not state.commands
