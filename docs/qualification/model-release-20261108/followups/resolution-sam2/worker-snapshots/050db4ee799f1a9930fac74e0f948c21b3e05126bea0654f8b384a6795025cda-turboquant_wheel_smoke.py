"""Validate an installed native TQ3 wheel with fabricated weights, never a hub model."""

import argparse
import importlib.metadata
import itertools
import json
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not any("references_libs" in path for path in sys.path)
    for name in ("mlx-lm", "transformers", "turboquant-mlx-full"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError(f"forbidden runtime dependency: {name}")
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one import CacheConfig, Engine
    from mlx_one.cli import main as cli_main
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.model_manager import ModelManager
    from mlx_one.text import generation
    from mlx_one.text.loading import load_text_model

    bundle = load_text_model(args.checkpoint, offline=True)
    assert bundle.model.lm_head.format == "turboquant_tq3"
    script = [9, 6, 10, 5, 11, 8, 14, 9, 7, 10, 4, 15]

    def reset():
        tokens = itertools.cycle(script)
        generation.select_token = lambda *a, **kw: next(tokens)

    prompt = "<|start|>assistant"
    reset()
    with Engine(bundle, cache_config=CacheConfig(prefix_cache_enabled=False)) as engine:
        result = engine.generate(prompt, max_tokens=32)
        assert result.text.strip() == "Answer"
        reset()
        cancel = threading.Event()
        stream = engine.stream(prompt, max_tokens=32, cancel=cancel)
        next(stream)
        cancel.set()
        list(stream)
        stream.close()
        assert engine.runtime_stats()["cache_stats"]["active_handles"] == 0
        reset()
        assert engine.generate(prompt, max_tokens=32).text.strip() == "Answer"
    reset()
    result = CliRunner().invoke(
        cli_main,
        [
            "generate",
            str(args.checkpoint),
            prompt,
            "--offline",
            "--max-tokens",
            "32",
            "--json-output",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["text"].strip() == "Answer"
    manager = ModelManager()
    manager.load(args.checkpoint, offline=True)
    config = ServerConfig(
        warmup=False, cache_prompt=False, api_keys=("wheel-test",), inference_mode="reference_dense"
    )
    try:
        with TestClient(create_app(manager, config=config)) as client:
            assert client.get("/v1/models").status_code == 401
            client.headers["Authorization"] = "Bearer wheel-test"
            model_id = client.get("/v1/models").json()["data"][0]["id"]
            for route in ("completions", "chat/completions"):
                payload = dict(model=model_id, max_tokens=32, temperature=0)
                payload.update(
                    dict(prompt=prompt)
                    if route == "completions"
                    else dict(messages=[dict(role="user", content="hello")])
                )
                for streaming in (False, True):
                    reset()
                    response = client.post("/v1/" + route, json={**payload, "stream": streaming})
                    assert response.status_code == 200, response.text
                    assert "Answer" in response.text and "scratchpad-secret" not in response.text
                    if streaming:
                        assert "[DONE]" in response.text
    finally:
        manager.unload()
    assert not any(name in sys.modules for name in ("mlx_lm", "transformers", "turboquant_mlx"))
    report = dict(
        status="PASS",
        fixture="fabricated tiny GPT-OSS, scripted sampler with real packed forwards",
        python=True,
        cli=True,
        http=True,
        sse=True,
        authentication=True,
        cancellation=True,
        unload=True,
        forbidden_runtime_dependencies=False,
        installed_package=importlib.metadata.version("mlx-one"),
        full_checkpoint_status="PENDING_USER_DOWNLOAD",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
