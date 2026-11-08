"""Exercise advertised speech CLI, HTTP/NDJSON and Python stream cleanup."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import wave
from pathlib import Path


def qualify(model, data, report, client, cli):
    import numpy as np
    from click.testing import CliRunner

    from mlx_one.audio.public import AudioEngine

    out = Path(report["evidence_directory"]) / "audio-outputs"
    out.mkdir(exist_ok=True)
    interfaces = report["interfaces"]
    text = "Hello from the speech test."
    source = data / "audio/sts-question.wav"

    def check_wav(content):
        with wave.open(io.BytesIO(content), "rb") as wav:
            assert wav.getframerate() == 24000 and wav.getnchannels() == 1
            assert wav.getsampwidth() == 2 and wav.getnframes() > 0
            return dict(
                samples=wav.getnframes(),
                sample_rate=24000,
                sha256=hashlib.sha256(content).hexdigest(),
            )

    for task in ["tts", "sts"]:
        path = out / f"interface-cli-{task}.wav"
        args = ["audio", "speak", "--model", str(model), "--offline", "--output", str(path)]
        args += ["--text", text] if task == "tts" else ["--audio", str(source)]
        result = CliRunner().invoke(cli, args)
        record = dict(exit_code=result.exit_code, output=result.output)
        interfaces[f"speech_cli_{task}"] = record
        assert result.exit_code == 0, result.output
        record.update(status="PASS", **check_wav(path.read_bytes()))

    response = client.post("/v1/audio/speech", json=dict(model="qualification", input=text))
    assert response.status_code == 200, response.text[:300]
    interfaces["speech_http_tts"] = dict(status="PASS", **check_wav(response.content))
    (out / "interface-http-tts.wav").write_bytes(response.content)
    response = client.post(
        "/v1/audio/speech-to-speech",
        data=dict(model="qualification"),
        files={"file": (source.name, source.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200, response.text[:300]
    interfaces["speech_http_sts"] = dict(status="PASS", **check_wav(response.content))
    (out / "interface-http-sts.wav").write_bytes(response.content)
    response = client.post(
        "/v1/audio/speech", json=dict(model="qualification", input=text, stream=True)
    )
    assert response.status_code == 200
    records = [json.loads(line) for line in response.text.splitlines() if line]
    assert sum(row["kind"] == "completion" for row in records) == 1
    assert not any(row["kind"] == "error" for row in records)
    assert [row["sequence"] for row in records] == list(range(len(records)))
    cursor = 0
    for row in records:
        if row["kind"] == "audio_chunk":
            assert row["sample_offset"] == cursor and row["sample_rate"] == 24000
            cursor += len(base64.b64decode(row["pcm_s16le_base64"], validate=True)) // 2
    assert cursor > 0
    (out / "interface-http-speech.ndjson").write_text(response.text)
    interfaces["speech_http_stream"] = dict(status="PASS", events=len(records), samples=cursor)
    with AudioEngine(model, offline=True) as engine:
        stream = engine.stream_generate(text)
        first = next(stream)
        assert first.kind in {"audio_chunk", "text_chunk"}
        stream.close()
        result = engine.generate_speech("This request still completes.")
        assert result.duration > 0 and np.isfinite(result.waveform).all()
    interfaces["speech_python_early_close_and_recovery"] = "PASS"
    invalid = client.post("/v1/audio/speech", json=dict(model="qualification", input=text, speed=2))
    assert invalid.status_code == 400
    interfaces["speech_invalid_speed"] = "PASS_400"
