#!/usr/bin/env python3
"""Installed-wheel Python versus real CLI-server HTTP checks for four tasks.

Run with -I using a native-only wheel environment. Requests use loopback and
local locked weights. There are no model downloads or throughput measurements.
"""

import argparse
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("fixtures", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
f = a.fixtures.resolve()
a.output.mkdir(parents=True, exist_ok=False)
for name in ("torch", "transformers", "onnxruntime", "mlx_lm", "omlx"):
    assert importlib.util.find_spec(name) is None, name
from mlx_one import VADEngine, classify, token_classify, zero_shot_classify  # noqa: E402

text = "I enjoyed this film."
image = f / "runs/m2-20261103/clip/real-images/chelsea.png"
labels = ["cat", "dog"]
audio = io.BytesIO()
with wave.open(audio, "wb") as stream:
    stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    stream.writeframes(np.zeros(1025, dtype="<i2").tobytes())
with VADEngine(f / "checkpoints/silero-vad") as engine:
    vad = engine.detect(audio.getvalue()).to_dict()
cases = [
    (
        "text-classification",
        "distilbert-sst2",
        {"json": {"model": "test", "input": text}},
        classify(f / "checkpoints/distilbert-sst2", text).to_dict(),
    ),
    (
        "token-classification",
        "bert-ner",
        {
            "json": {
                "model": "test",
                "input": "Marie Curie lived in Paris.",
                "aggregation_strategy": "simple",
            }
        },
        token_classify(
            f / "checkpoints/bert-ner", "Marie Curie lived in Paris.", aggregation_strategy="simple"
        ).to_dict(),
    ),
    (
        "zero-shot-image-classification",
        "clip-converted",
        {
            "data": {"model": "test", "candidate_labels": json.dumps(labels)},
            "files": {"file": ("cat.png", image.read_bytes(), "image/png")},
        },
        zero_shot_classify(f / "checkpoints/clip-converted", image, labels).to_dict(),
    ),
    (
        "voice-activity-detection",
        "silero-vad",
        {
            "data": {"model": "test"},
            "files": {"file": ("silence.wav", audio.getvalue(), "audio/wav")},
        },
        vad,
    ),
]
results = []
env = {**os.environ, "HF_HUB_OFFLINE": "1", "MLX_ENABLE_TF32": "0"}
env.pop("PYTHONPATH", None)
for task, checkpoint, request, expected in cases:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-I",
        "-m",
        "mlx_one.cli",
        "serve",
        str(f / "checkpoints" / checkpoint),
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "test",
        "--api-key",
        "integration-fixture-key",
        "--no-warmup",
        "--timeout",
        "30",
    ]
    with (a.output / (task + ".log")).open("x") as log:
        process = subprocess.Popen(
            command, cwd="/private/tmp", env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False
            ) as client:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("server did not become healthy: " + task)
                    time.sleep(0.1)
                route = "/v1/tasks/" + task
                assert client.post(route, **request).status_code == 401
                client.headers["Authorization"] = "Bearer integration-fixture-key"
                response = client.post(route, **request)
                assert response.status_code == 200, response.text
                assert response.json() == json.loads(json.dumps(expected))
                assert client.post(route, **request).json() == response.json()
                results.append(
                    dict(
                        task=task,
                        http_status=200,
                        unauth_status=401,
                        python_http_equal=True,
                        repeat_equal=True,
                    )
                )
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        assert process.returncode == 0, (task, process.returncode)
        results[-1]["clean_server_exit"] = True
    print("PASS", task, flush=True)
assert not [
    n
    for n in sys.modules
    if n.split(".")[0] in {"torch", "transformers", "onnxruntime", "mlx_lm", "omlx"}
]
(a.output / "summary.json").write_text(
    json.dumps(dict(results=results, forbidden_runtime_imports=[]), indent=2) + "\n"
)
