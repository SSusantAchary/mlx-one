#!/usr/bin/env python3
"""One real loopback HTTP inference plus strict negative identity probes, offline."""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import mlx.core as mx
import uvicorn

from mlx_one.engine.modes import configure_execution_dtype
from mlx_one.inference_qualification import write_integrity
from mlx_one.native_correctness import checkpoint_proof
from mlx_one.native_qualification import qualification_source, write
from mlx_one.server.app import create_app
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEngine
from mlx_one.server.model_manager import ModelManager
from mlx_one.server.scheduler import GenerationScheduler


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    paths = json.loads(os.environ["MLX_ONE_FUSED_PATHS"])
    keys = json.loads(os.environ["MLX_ONE_FUSED_KEYS"])
    report = {"source": qualification_source(), "status": "RUNNING", "cases": []}
    try:
        for key in keys:
            asset = checkpoint_proof(key, paths[key])
            row = {"checkpoint": asset, "status": "RUNNING", "probes": []}
            report["cases"].append(row)
            scheduler = GenerationScheduler(parallel=1, timeout=120)
            manager = ModelManager()
            engine = GenerationEngine(
                manager,
                inference_mode="reference_dense",
                execution_dtype="float16",
                collect_execution_metrics=True,
            )
            server = thread = listener = None
            dispatches = []

            class Observer:
                capture_math_dispatch = True
                profile_math = False

                def __init__(self, events):
                    self.events = events

                def __call__(self, event):
                    self.events.append(event)

            engine._qualification_math_observer = Observer(dispatches)
            try:
                bundle = scheduler.execute(
                    lambda manager=manager, asset=asset: manager.load(
                        asset["model_id"],
                        checkpoint_path=asset["path"],
                        expected_files=asset["files"],
                        revision=asset["revision"],
                        offline=True,
                    )
                )
                scheduler.execute(
                    lambda bundle=bundle: configure_execution_dtype(bundle, "float16")
                )
                app = create_app(
                    manager,
                    engine,
                    scheduler=scheduler,
                    config=ServerConfig(
                        warmup=False,
                        inference_mode="reference_dense",
                        execution_dtype="float16",
                        reasoning="off",
                        cache_prompt=False,
                        cache_idle_slots=False,
                    ),
                )
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                listener.listen(16)
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
                thread = threading.Thread(
                    target=lambda server=server, listener=listener: server.run(sockets=[listener]),
                    daemon=True,
                )
                thread.start()
                deadline = time.monotonic() + 30
                while not server.started:
                    if not thread.is_alive() or time.monotonic() >= deadline:
                        raise RuntimeError("HTTP server failed to start")
                    time.sleep(0.01)
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
                    for model, expected in (
                        (asset["model_id"] + "-wrong", 404),
                        (asset["path"], 404),
                        (asset["model_id"], 200),
                    ):
                        response = client.post(
                            "/v1/chat/completions",
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "user", "content": "Reply with one word: hello."}
                                ],
                                "max_tokens": 4,
                                "temperature": 0,
                                "stream": False,
                            },
                        )
                        row["probes"].append(
                            {
                                "requested_model": model,
                                "status_code": response.status_code,
                                "body": response.json(),
                            }
                        )
                        if response.status_code != expected:
                            raise AssertionError(
                                f"identity probe expected {expected}: {response.text}"
                            )
                    row["runtime"] = client.get("/v1/runtime").json()
                completed = row["probes"][-1]["body"]
                if completed.get("usage", {}).get("completion_tokens", 0) < 1:
                    raise AssertionError("HTTP 200 did not complete native inference")
                attention = [e for e in dispatches if e.get("operation") == "attention"]
                if not attention or any(e["selected"] != "reference" for e in attention):
                    raise AssertionError("reference HTTP request lacks actual reference dispatch")
                if checkpoint_proof(key, paths[key]) != asset:
                    raise AssertionError("checkpoint changed during HTTP inference")
                row.update(
                    status="PASS",
                    math_dispatch=dispatches,
                    peak_mlx_bytes=scheduler.execute(mx.get_peak_memory),
                )
            except Exception as exc:
                row.update(
                    status="FAIL", exception={"type": type(exc).__name__, "message": str(exc)}
                )
                raise
            finally:
                if server:
                    server.should_exit = True
                if thread:
                    thread.join(timeout=10)
                if listener:
                    listener.close()
                scheduler.close(
                    finalizer=lambda engine=engine, manager=manager: (
                        engine.close(),
                        manager.unload(),
                    )
                )
                row["cleanup"] = {"http_thread_alive": bool(thread and thread.is_alive())}
                if row["cleanup"]["http_thread_alive"]:
                    row["status"] = "FAIL"
        report["status"] = "PASS" if all(r["status"] == "PASS" for r in report["cases"]) else "FAIL"
    except Exception as exc:
        report.update(status="FAIL", exception={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "http-identity.json", report)
        write_integrity(output)


if __name__ == "__main__":
    main()
