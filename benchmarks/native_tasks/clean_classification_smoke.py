#!/usr/bin/env python3
"""Run with the clean wheel interpreter and -I, outside the source checkout."""

import importlib.util
import json
import socket
import subprocess
import sys

for name in ("torch", "transformers", "onnxruntime", "mlx_lm", "omlx"):
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError(f"reference framework present in clean environment: {name}")


def deny_network(*args, **kwargs):
    raise RuntimeError("network disabled by clean-wheel smoke")


socket.socket.connect = deny_network
from fastapi.testclient import TestClient  # noqa: E402

import mlx_one  # noqa: E402
from mlx_one.classification import classify  # noqa: E402
from mlx_one.server import ModelManager, ServerConfig, create_app  # noqa: E402

checkpoint = sys.argv[1]
text = "I loved this film."
python = classify(checkpoint, text, offline=True).to_dict()
manager = ModelManager(alias="sentiment")
manager.load(checkpoint, offline=True)
with TestClient(
    create_app(
        model_manager=manager, config=ServerConfig(api_keys=("key",), warmup=False, timeout=30)
    )
) as client:
    body = {"model": "sentiment", "input": text}
    assert client.post("/v1/tasks/text-classification", json=body).status_code == 401
    response = client.post(
        "/v1/tasks/text-classification", json=body, headers={"Authorization": "Bearer key"}
    )
    assert response.status_code == 200, response.text
    assert response.json() == json.loads(json.dumps(python))
cli = subprocess.run(
    [
        sys.executable,
        "-I",
        "-m",
        "mlx_one.cli",
        "classify",
        checkpoint,
        text,
        "--offline",
        "--json-output",
    ],
    capture_output=True,
    text=True,
    check=True,
)
assert json.loads(cli.stdout) == json.loads(json.dumps(python))
forbidden = sorted(
    name
    for name in sys.modules
    if name.split(".")[0] in {"torch", "transformers", "onnxruntime", "mlx_lm", "omlx"}
)
assert not forbidden, forbidden
assert "/site-packages/" in mlx_one.__file__, mlx_one.__file__
print(
    json.dumps(
        {
            "package": mlx_one.__file__,
            "python": sys.version,
            "network": "socket connections denied in parent; Hub offline in all processes",
            "forbidden_modules": forbidden,
            "python_cli_http_equal": True,
            "result": python,
        },
        indent=2,
    )
)
