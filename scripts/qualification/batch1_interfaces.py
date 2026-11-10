"""Real checkpoint interfaces and installed-package identity; no reference runtime."""

import base64
import hashlib
import importlib.metadata
import json
import sys
import threading
from pathlib import Path

import numpy as np


def package_identity():
    import mlx_one

    root = Path(mlx_one.__file__).resolve().parent
    forbidden = (
        "torch",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "kokoro",
        "mlx-lm",
        "mlx-vlm",
    )
    present = []
    for name in forbidden:
        try:
            importlib.metadata.distribution(name)
            present.append(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(
        installed_in_environment=root.is_relative_to(Path(sys.prefix)),
        forbidden_distributions=present,
        files_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.py")
        },
        python=sys.version.split()[0],
    )


def check_interfaces(bundle, kind, *, images=None, output_dir=None):
    """Exercise real Python/CLI/actor/HTTP paths while the exact assets still exist."""
    from click.testing import CliRunner
    from fastapi.testclient import TestClient

    from mlx_one.cli import main
    from mlx_one.engine.batching import BatchingConfig
    from mlx_one.engine.service import InferenceService
    from mlx_one.server.app import create_app
    from mlx_one.server.config import ServerConfig
    from mlx_one.server.model_manager import ModelManager

    checks = {}
    query, documents = (
        "What is the capital of France?",
        ["The capital of France is Paris.", "Bananas are yellow fruit."],
    )
    root = str(bundle.path)
    if kind == "embedding":
        from mlx_one.retrieval import embed

        payload = dict(texts=documents)
        expected = embed(bundle, documents)
        cli = ["embed", root, *documents, "--offline", "--json-output"]
        route, body = "/v1/embeddings", dict(input=documents)

        def equal(value):
            return np.allclose(value, expected.embeddings, atol=1e-4, rtol=1e-4)
    elif kind == "reranking":
        from mlx_one.retrieval import rerank

        payload = dict(query=query, documents=documents)
        expected = rerank(bundle, query, documents)
        cli = ["rerank", root, query, *documents, "--offline", "--json-output"]
        route, body = "/v1/rerank", dict(query=query, documents=documents)

        def equal(value):
            return [row["index"] for row in value] == [
                row.index for row in expected.items
            ] and np.allclose(
                [row["score"] for row in value],
                [row.score for row in expected.items],
                atol=1e-4,
                rtol=1e-4,
            )
    elif kind == "text-to-text-generation":
        from mlx_one.seq2seq import generate

        text = "translate English to German: The house is wonderful."
        payload = dict(texts=text, max_new_tokens=24)
        expected = generate(bundle, **payload)
        cli = ["seq2seq", root, text, "--offline", "--max-new-tokens", "24", "--json-output"]
        route, body = "/v1/tasks/text-to-text-generation", dict(input=text, max_new_tokens=24)

        def equal(value):
            return tuple(value["texts"]) == expected.texts
    elif kind == "image-classification":
        from mlx_one.classification.image import classify_image

        payload = dict(images=images, top_k=5)
        expected = classify_image(bundle, **payload)
        cli = ["classify-image", root, *map(str, images), "--offline", "--json-output"]
        route, body = (
            "/v1/tasks/image-classification",
            dict(
                images=[
                    "data:image/jpeg;base64," + base64.b64encode(Path(image).read_bytes()).decode()
                    for image in images
                ]
            ),
        )

        def equal(value):
            actual = [[row["index"] for row in item["labels"]] for item in value["predictions"]]
            wanted = [[row["index"] for row in item["labels"]] for item in expected.predictions]
            return actual == wanted
    elif kind == "speech-generation":
        from mlx_one.audio.kokoro import synthesize

        payload = dict(text="Hello world. This is native speech.")
        expected = synthesize(bundle, **payload)
        cli = [
            "tts",
            root,
            payload["text"],
            "--output",
            str(output_dir / "cli.wav"),
            "--json-output",
        ]
        route, body = "/v1/tasks/speech-generation", dict(input=payload["text"])

        def equal(value):
            return value["sample_rate"] == 24000 and value["samples"] > 0
    else:
        raise ValueError("unsupported interface qualification task")

    def parsed(value):
        if kind == "embedding":
            return value["embeddings"]
        if kind == "reranking":
            return value["items"]
        return value

    invoked = CliRunner().invoke(main, cli)
    checks["cli"] = dict(
        status="PASS"
        if invoked.exit_code == 0 and equal(parsed(json.loads(invoked.output)))
        else "FAIL",
        exit_code=invoked.exit_code,
        error=str(invoked.exception) if invoked.exception else None,
    )
    service = InferenceService(
        batching_config=BatchingConfig(
            max_sequences=8, max_scheduled_tokens=8192, task_transient_budget_bytes=4 * 1024**3
        )
    )
    try:
        service.register_task("candidate", bundle)
        try:
            actual = service.run_task("candidate", kind, payload).output
            checks["engine"] = dict(status="PASS" if equal(parsed(actual.to_dict())) else "FAIL")
        except Exception as exc:
            checks["engine"] = dict(status="FAIL", exception=type(exc).__name__, reason=str(exc))
        canceled = threading.Event()
        canceled.set()
        try:
            service.run_task("candidate", kind, payload, cancel=canceled)
            checks["canceled_request"] = dict(
                status="FAIL", reason="already canceled request executed"
            )
        except Exception as exc:
            # A canceled request must terminate without returning a task output.
            checks["canceled_request"] = dict(status="PASS", exception=type(exc).__name__)
    finally:
        service.close()
    try:
        service.run_task("candidate", kind, payload)
        checks["closed_service"] = dict(status="FAIL")
    except RuntimeError:
        checks["closed_service"] = dict(status="PASS")

    manager = ModelManager(alias="batch1")
    manager.load(
        root,
        offline=True,
        **({"retrieval_task": kind} if kind in {"embedding", "reranking"} else {}),
    )
    with TestClient(
        create_app(
            config=ServerConfig(
                warmup=False, cache_prompt=False, timeout=300, api_keys=("batch1-key",)
            ),
            model_manager=manager,
        )
    ) as client:
        body["model"] = "batch1"
        unauthorized = client.post(route, json=body)
        headers = {"Authorization": "Bearer batch1-key"}
        response = client.post(route, json=body, headers=headers)
        ok = response.status_code == 200
        if ok:
            if kind == "speech-generation":
                import io
                import wave

                with wave.open(io.BytesIO(response.content)) as wav:
                    ok = wav.getframerate() == 24000 and wav.getnframes() > 0
            else:
                value = response.json()
                value = (
                    [item["embedding"] for item in value["data"]]
                    if kind == "embedding"
                    else value["results"]
                    if kind == "reranking"
                    else value
                )
                ok = equal(value)
        checks["http"] = dict(
            status="PASS" if ok else "FAIL",
            status_code=response.status_code,
            error=response.text[:500] if response.status_code != 200 else None,
        )
        invalid = client.post(route, json={**body, "model": "missing"}, headers=headers)
        malformed = client.post(route, json={"model": "batch1"}, headers=headers)
        checks["http_validation"] = dict(
            status="PASS"
            if unauthorized.status_code == 401
            and invalid.status_code == 404
            and malformed.status_code == 422
            else "FAIL",
            unauthorized=unauthorized.status_code,
            wrong_model=invalid.status_code,
            malformed=malformed.status_code,
        )
    try:
        manager.current_model()
        checks["http_shutdown"] = dict(status="FAIL")
    except RuntimeError:
        checks["http_shutdown"] = dict(status="PASS")
    return checks
