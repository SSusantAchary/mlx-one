"""Pinned real-checkpoint VAD state, source timestamps and task interfaces."""

import ast
import io
import json
import threading
import time
import wave
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "local/native_tasks/checkpoints/silero-vad"
RUN = ROOT / "local/native_tasks/runs/m3-20261103"
pytestmark = pytest.mark.skipif(not CHECKPOINT.exists(), reason="pinned Silero assets absent")


def source_timestamps():
    source = ast.parse((CHECKPOINT / "utils_vad.py").read_text())
    node = next(
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_speech_timestamps_from_probs"
    )
    namespace = {"List": list}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "pinned-utils_vad.py", "exec"), namespace
    )
    return namespace[node.name]


@pytest.fixture
def engine():
    from mlx_one import VADEngine

    with VADEngine(CHECKPOINT) as e:
        yield e


def test_vad_offline_streaming_tail_reset_and_interleaved_state(engine):
    from mlx_one.audio.vad import prepare_vad

    pcm = prepare_vad(CHECKPOINT / "test.wav")[:17001]
    expected = engine.detect(pcm)
    a, b = engine.session(), engine.session()
    for offset in range(0, len(pcm) // 512 * 512, 512):
        a.process_chunk(pcm[offset : offset + 512])
        b.process_chunk(np.zeros(512, np.float32))
    actual = a.finalize(pcm[len(pcm) // 512 * 512 :])
    assert actual.to_dict() == expected.to_dict()
    assert not b.finalize().segments
    assert not np.array_equal(a.state, b.state)
    with pytest.raises(ValueError):
        a.process_chunk(pcm[:512])
    with pytest.raises(ValueError):
        a.finalize()
    a.reset()
    assert a.process_chunk(pcm[:512]) == expected.probabilities[0]
    assert engine.service.pool.stats()["active_leases"] == 0


def test_vad_real_oracle_and_source_timestamp_policy(engine):
    path = RUN / "silero-full-reference.npz"
    if not path.exists():
        pytest.skip("real reference traces absent")
    with np.load(path, allow_pickle=False) as f:
        pcm, probabilities = f["pcm"], f["probabilities"].reshape(-1)
    oracle = source_timestamps()
    for options in (
        {},
        {"threshold": 0.3},
        {"speech_pad_ms": 0},
        {"min_speech_duration_ms": 100, "min_silence_duration_ms": 50},
    ):
        result = engine.detect(pcm, **options)
        assert np.allclose(result.probabilities, probabilities, atol=1e-4, rtol=1e-4)
        assert list(result.segments) == oracle(
            probabilities, audio_length_samples=len(pcm), **options
        )
    assert engine.detect(np.zeros(0, np.float32)).segments == ()


def test_vad_source_policy_edge_cases_and_bounded_errors(engine):
    from mlx_one.audio.vad_postprocessing import get_speech_timestamps_from_probs as native
    from mlx_one.engine.execution_context import RequestTimeoutError

    oracle = source_timestamps()
    for probabilities in [
        [],
        [0.0] * 20,
        [1.0] * 20,
        [0.0] * 3 + [1.0] * 12 + [0.0] * 12 + [1.0] * 10,
    ]:
        for options in (
            {},
            {"max_speech_duration_s": 0.2},
            {"speech_pad_ms": 100},
            {"return_seconds": True, "time_resolution": 2},
        ):
            assert native(probabilities, **options) == oracle(probabilities, **options)
    for pcm, options in [
        (np.zeros(3), {"sample_rate": 8000}),
        (np.ones((10, 2)), {}),
        (np.array([np.nan], np.float32), {}),
        (np.zeros(3), {"threshold": True}),
        (np.zeros(3), {"threshold": 0}),
        (np.zeros(3), {"unknown": 1}),
    ]:
        with pytest.raises(ValueError):
            engine.detect(pcm, **options)
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledError):
        engine.detect(np.zeros(512, np.float32), cancel=event)
    with pytest.raises(RequestTimeoutError):
        engine.detect(np.zeros(512, np.float32), deadline=time.monotonic() - 1)
    with ThreadPoolExecutor(2) as pool:
        result = list(pool.map(lambda n: engine.detect(np.zeros(n, np.float32)), [512, 1024]))
    assert [r.sample_count for r in result] == [512, 1024]


def test_vad_python_cli_http_and_unload(engine, tmp_path):
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one.cli import main
    from mlx_one.server import ModelManager, ServerConfig, create_app

    data = io.BytesIO()
    with wave.open(data, "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(np.zeros(1025, dtype="<i2").tobytes())
    path = tmp_path / "silence.wav"
    path.write_bytes(data.getvalue())
    expected = json.loads(engine.detect(path).to_json())
    cli = CliRunner().invoke(main, ["vad", str(CHECKPOINT), str(path), "--json-output"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output) == expected
    manager = ModelManager(alias="vad")
    manager.load(CHECKPOINT, offline=True)
    with TestClient(
        create_app(model_manager=manager, config=ServerConfig(api_keys=("key",), warmup=False))
    ) as c:
        route = "/v1/tasks/voice-activity-detection"
        files = {"file": ("silence.wav", data.getvalue(), "audio/wav")}
        body = {"model": "vad"}
        assert c.post(route, data=body, files=files).status_code == 401
        c.headers["Authorization"] = "Bearer key"
        for _ in range(2):
            response = c.post(route, data=body, files=files)
            assert response.status_code == 200, response.text
            assert response.json() == expected
        assert c.post(route, json=body).status_code == 415
        assert c.post(route, data={"model": "wrong"}, files=files).status_code == 404
        assert c.post(route, data={**body, "threshold": "nan"}, files=files).status_code == 422
        assert (
            c.post(route, data={**body, "file_path": "/etc/passwd"}, files=files).status_code == 422
        )
        assert (
            c.post(route, data=body, files={"file": ("bad.wav", b"bad", "audio/wav")}).status_code
            == 422
        )
        assert (
            c.post(
                "/v1/chat/completions",
                json={"model": "vad", "messages": [{"role": "user", "content": "hello"}]},
            ).status_code
            == 400
        )
    with pytest.raises(RuntimeError):
        manager.current_model()
