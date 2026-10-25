"""Run outside the checkout with an isolated wheel installation and import guard."""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path


class ForbidReferences(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlx_lm", "transformers"}:
            raise AssertionError("production imported " + fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--saved-profile-info", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    sys.meta_path.insert(0, ForbidReferences())
    for name in ["mlx-lm", "transformers"]:
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError(name + " is installed")
    import mlx.core as mx

    import mlx_one
    from mlx_one import CacheConfig, Engine
    from mlx_one.core.registry import registered_model_types

    assert mx.metal.is_available() and mx.default_device() == mx.gpu
    assert Path(mlx_one.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert len(registered_model_types()) == 36
    asset = next(r for r in json.loads(args.assets.read_text()) if r["id"] == args.model)
    if args.saved_profile_info:
        saved = json.loads(args.saved_profile_info.read_text())
        if saved["status"] != "PASS":
            raise ValueError("saved profile was not verified")
        asset = {
            **asset,
            "source_revision": asset["revision"],
            "path": saved["destination"],
            "revision": saved["revision"],
            "precision": saved["profile"],
        }
    report = {
        "timestamp": time.time(),
        "checkpoint": asset["repo"],
        "revision": asset["revision"],
        "installed_source": mlx_one.__file__,
        "version": mlx_one.__version__,
        "reference_packages_absent": True,
        "registrations": len(registered_model_types()),
        "source_revision": asset.get("source_revision", asset["revision"]),
        "profile": asset.get("precision", "stored"),
    }
    if not args.http:
        with Engine(
            asset["path"],
            offline=True,
            inference_mode="reference_dense",
            cache_config=CacheConfig(prefix_cache_enabled=False),
        ) as engine:
            assert engine._bundle.revision == asset["revision"]
            result = engine.generate("The city was", max_tokens=2)
            report.update(
                text=result.text,
                finish_reason=result.finish_reason,
                generated_tokens=result.generation_tokens,
                cleanup=engine.runtime_stats()["cache_stats"],
            )
            assert report["cleanup"]["active_handles"] == report["cleanup"]["reserved_bytes"] == 0
    else:
        import httpx
        import uvicorn

        from mlx_one.server.app import create_app
        from mlx_one.server.config import ServerConfig
        from mlx_one.server.generation_engine import GenerationEngine
        from mlx_one.server.model_manager import ModelManager

        manager = ModelManager()
        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary)
            for source in Path(asset["path"]).iterdir():
                if source.is_file() and source.suffix in {".json", ".txt", ".jinja", ".model"}:
                    shutil.copy2(source, malformed / source.name)
            (malformed / "model.safetensors.index.json").unlink(missing_ok=True)
            (malformed / "mlx-one-conversion.json").unlink(missing_ok=True)
            (malformed / "model.safetensors").write_bytes(b"invalid safetensors header")
            try:
                manager.load(malformed, offline=True)
            except Exception as exc:
                from mlx_one.text.loading import TextModelLoadError

                assert isinstance(exc, TextModelLoadError)
                report["malformed_artifact"] = "PASS"
            else:
                raise AssertionError("malformed model was accepted")
        bundle = manager.load(asset["path"], offline=True)
        engine = GenerationEngine(manager, inference_mode="reference_dense")
        app = create_app(
            manager,
            engine,
            config=ServerConfig(
                warmup=False,
                cache_prompt=False,
                inference_mode="reference_dense",
                api_keys=("qualification-example-key",),
                reasoning="off",
            ),
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 20
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
                assert client.get("/health").status_code == 200
                assert client.get("/v1/models").status_code == 401
                client.headers["Authorization"] = "Bearer qualification-example-key"
                models = client.get("/v1/models").json()["data"]
                assert len(models) == 1 and models[0]["id"] == bundle.model_id
                assert models[0]["mlx"]["release_support"]["alpha_supported"] is False
                runtime = client.get("/v1/runtime").json()
                assert "qualification-example-key" not in json.dumps(runtime)
                report["runtime"] = runtime
                payload = {
                    "model": bundle.model_id,
                    "prompt": "The city was",
                    "max_tokens": 4,
                    "temperature": 0,
                }
                completion = client.post("/v1/completions", json=payload)
                completion.raise_for_status()
                stream = client.post("/v1/completions", json={**payload, "stream": True})
                chunks = [
                    json.loads(s[6:])
                    for s in stream.text.splitlines()
                    if s.startswith("data: ") and s != "data: [DONE]"
                ]
                assert (
                    "".join(c["choices"][0]["text"] for c in chunks)
                    == completion.json()["choices"][0]["text"]
                )
                assert stream.text.count("data: [DONE]") == 1
                body = {
                    "model": bundle.model_id,
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "max_tokens": 4,
                    "temperature": 0,
                }
                chat = client.post("/v1/chat/completions", json=body)
                chat.raise_for_status()
                chat_stream = client.post("/v1/chat/completions", json={**body, "stream": True})
                assert chat_stream.status_code == 200 and "data: [DONE]" in chat_stream.text
                assert (
                    client.post("/v1/completions", json={**payload, "max_tokens": -1}).status_code
                    == 422
                )
                assert (
                    client.post("/v1/completions", json={**payload, "model": "missing"}).status_code
                    == 404
                )
                assert engine.cache_stats()["active_handles"] == 0
                report.update(
                    completion=completion.json(),
                    chat=chat.json(),
                    completion_sse="PASS",
                    chat_sse="PASS",
                    authentication="PASS",
                    errors="PASS",
                    cleanup=engine.cache_stats(),
                )
        finally:
            server.should_exit = True
            thread.join(timeout=20)
            listener.close()
            assert not thread.is_alive()
        report["server_shutdown"] = True
    assert not any(n.split(".")[0] in {"mlx_lm", "transformers"} for n in sys.modules)
    report["status"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.model, "http" if args.http else "native", "PASS")


if __name__ == "__main__":
    main()
