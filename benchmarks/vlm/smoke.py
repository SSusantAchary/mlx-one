"""Exact-checkpoint Python/CLI/live HTTP image smoke; results are not accuracy scores."""

from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.vlm.metrics import EVIDENCE, read
from benchmarks.vlm.run import identify


def run(checkpoint, model, image_dir):
    import httpx
    import mlx.core as mx

    from mlx_one.text import TextGenerationOptions
    from mlx_one.vision import VisionLanguageEngine

    identify(checkpoint, model)
    image = image_dir / "VLM-HF-S1-1.png"
    car = image_dir / "VLM-HF-S3-2.png"
    record = {
        "checkpoint": model["checkpoint"],
        "revision": model["revision"],
        "weight_sha256": model["weight_sha256"],
        "interfaces": {},
        "semantic_review": "PENDING",
    }
    with VisionLanguageEngine(checkpoint, offline=True, revision=model["revision"]) as engine:
        first = engine.generate(
            "Describe this image.", [image], options=TextGenerationOptions(max_tokens=64)
        )
        changed = engine.generate(
            "Describe this image.", [car], options=TextGenerationOptions(max_tokens=48)
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": "How many cats are visible?"},
        ]
        followup = engine.generate(messages=messages, options=TextGenerationOptions(max_tokens=32))
        stream = engine.stream(
            "Describe this image.", [image], options=TextGenerationOptions(max_tokens=128)
        )
        try:
            for index, _ in enumerate(stream):
                if index == 1:
                    break
        finally:
            stream.close()
        memory = []
        for _ in range(3):
            engine.generate(
                "Describe this image.", [image], options=TextGenerationOptions(max_tokens=16)
            )
            mx.synchronize()
            mx.clear_cache()
            memory.append(mx.get_active_memory())
        record["python"] = {
            "image": first.to_dict(),
            "changed_image": changed.to_dict(),
            "multiturn": followup.to_dict(),
            "post_request_active_bytes": memory,
            "early_stream_close_and_next_request": True,
        }
        record["interfaces"]["python"] = "PASS"
    cli = str(Path(sys.executable).with_name("mlx-one"))
    completed = subprocess.run(
        [
            cli,
            "vlm",
            "--model",
            str(checkpoint),
            "--offline",
            "--image",
            str(image),
            "--prompt",
            "Describe this image.",
            "--max-tokens",
            "16",
            "--json-output",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    record["cli"] = json.loads(completed.stdout)
    record["interfaces"]["cli"] = "PASS"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_log = checkpoint.parent / (model["id"] + "-server.log")
    with server_log.open("w") as log:
        child = subprocess.Popen(
            [
                cli,
                "serve",
                str(checkpoint),
                "--offline",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--alias",
                "vlm-smoke",
                "--api-key",
                "local-vlm-test",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180) as client:
                ready = False
                for _ in range(1200):
                    if child.poll() is not None:
                        raise RuntimeError("server exited; inspect " + str(server_log))
                    try:
                        if client.get("/health").status_code == 200:
                            ready = True
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.1)
                if not ready:
                    raise RuntimeError("server startup timeout")
                assert client.get("/v1/models").status_code == 401
                headers = {"Authorization": "Bearer local-vlm-test"}
                discovery = client.get("/v1/models", headers=headers)
                discovery.raise_for_status()
                assert discovery.json()["data"][0]["id"] == "vlm-smoke"
                data = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
                payload = {
                    "model": "vlm-smoke",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data}},
                                {"type": "text", "text": "Describe this image."},
                            ],
                        }
                    ],
                }
                response = client.post("/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                record["http"] = response.json()
                payload["messages"] += [
                    {
                        "role": "assistant",
                        "content": response.json()["choices"][0]["message"]["content"],
                    },
                    {"role": "user", "content": "How many cats are visible?"},
                ]
                streamed = client.post(
                    "/v1/chat/completions", headers=headers, json={**payload, "stream": True}
                )
                streamed.raise_for_status()
                assert "data: [DONE]" in streamed.text
                record["sse"] = streamed.text
                with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    headers=headers,
                    json={**payload, "stream": True, "max_tokens": 128},
                ) as early:
                    early.raise_for_status()
                    for line in early.iter_lines():
                        if line.startswith("data: "):
                            break
                after = client.post("/v1/chat/completions", headers=headers, json=payload)
                after.raise_for_status()
                record["interfaces"].update(
                    http="PASS",
                    sse="PASS",
                    auth_alias="PASS",
                    disconnect_recovery="PASS",
                    startup_warmup="PASS",
                )
        finally:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    record["status"] = "CHECKPOINT_SMOKE_PASSED"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    model = next(
        m for m in read(EVIDENCE / "benchmark_plan.json")["checkpoints"] if m["id"] == args.id
    )
    result = run(args.checkpoint.resolve(), model, args.image_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
