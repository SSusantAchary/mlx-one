#!/usr/bin/env python3
"""Exercise the installed CLI server over loopback and shut it down cleanly."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("python", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("log", type=Path)
args = parser.parse_args()
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
command = [
    str(args.python.absolute()),
    "-I",
    "-m",
    "mlx_one.cli",
    "serve",
    str(args.checkpoint.resolve()),
    "--offline",
    "--host",
    "127.0.0.1",
    "--port",
    str(port),
    "--alias",
    "sentiment",
    "--api-key",
    "test-key",
    "--no-warmup",
    "--timeout",
    "30",
]
environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
environment.pop("PYTHONPATH", None)
with args.log.open("x") as log:
    process = subprocess.Popen(
        command, cwd="/private/tmp", env=environment, stdout=log, stderr=subprocess.STDOUT
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False
        ) as client:
            deadline = time.monotonic() + 20
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("classification server failed to start; inspect log")
                time.sleep(0.1)
            body = {"model": "sentiment", "input": "I loved this film."}
            assert client.post("/v1/tasks/text-classification", json=body).status_code == 401
            client.headers["Authorization"] = "Bearer test-key"
            response = client.post("/v1/tasks/text-classification", json=body)
            assert response.status_code == 200, response.text
            assert response.json()["predictions"][0]["labels"][0]["label"] == "POSITIVE"
            repeated = client.post("/v1/tasks/text-classification", json=body)
            assert repeated.json() == response.json()
            result = {
                "command": command,
                "http_status": 200,
                "unauth_status": 401,
                "repeat_equal": True,
                "result": response.json(),
            }
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    result["server_exit_code"] = process.returncode
    if process.returncode != 0:
        raise RuntimeError(f"server shutdown failed: {process.returncode}")
with args.output.open("x") as output:
    json.dump(result, output, indent=2, allow_nan=False)
    output.write("\n")
