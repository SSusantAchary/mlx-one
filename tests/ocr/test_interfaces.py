import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 for native Metal OCR tests", allow_module_level=True)

import json
import threading
from concurrent.futures import CancelledError

import pytest
from click.testing import CliRunner
from conftest import write_tiny_checkpoint
from fastapi.testclient import TestClient
from PIL import Image

from mlx_one.cli import main
from mlx_one.ocr import OCREngine
from mlx_one.server.app import create_app
from mlx_one.server.config import ServerConfig
from mlx_one.server.model_manager import ModelManager


@pytest.mark.parametrize(
    "family", ["trocr", "got_ocr2", "florence2", "paddleocr_vl", "lightonocr", "deepseek_ocr"]
)
def test_cli_and_authenticated_http_native(tmp_path, family):
    root = write_tiny_checkpoint(tmp_path / family, family)
    path = tmp_path / "line.png"
    with Image.new("RGB", (16, 12), "white") as image:
        image.save(path)
    cli = CliRunner().invoke(
        main,
        [
            "ocr",
            "--model",
            str(root),
            "--image",
            str(path),
            "--offline",
            "--max-tokens",
            "2",
            "--json",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["pages"][0]["metadata"]["visual_tokens"] > 0
    manager = ModelManager(alias="local-ocr")
    manager.load(root, offline=True)
    app = create_app(
        model_manager=manager, config=ServerConfig(api_keys=("secret",), warmup=False, timeout=0)
    )
    with TestClient(app) as client:
        assert client.post("/v1/ocr").status_code == 401
        auth = {"Authorization": "Bearer secret"}
        assert client.get("/v1/models", headers=auth).json()["data"][0]["id"] == "local-ocr"
        response = client.post(
            "/v1/ocr",
            headers=auth,
            files={"file": ("line.png", path.read_bytes(), "image/png")},
            data={"model": "local-ocr", "max_tokens": "2"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["pages"][0]["metadata"]["visual_tokens"] > 0
        assert response.json()["pages"][0]["elements"] == []
        invalid = client.post(
            "/v1/ocr",
            headers=auth,
            files={"file": ("x.txt", b"bad", "text/plain")},
            data={"model": "local-ocr"},
        )
        assert invalid.status_code == 415
        mismatch = client.post(
            "/v1/ocr",
            headers=auth,
            files={"file": ("line.png", path.read_bytes())},
            data={"model": "wrong"},
        )
        assert mismatch.status_code == 404
        stream = client.post(
            "/v1/ocr",
            headers=auth,
            files={"file": ("line.png", path.read_bytes())},
            data={"model": "local-ocr", "stream": "true", "max_tokens": "2"},
        )
        assert stream.status_code == 200
        assert "event: page_started" in stream.text and "event: document_completed" in stream.text


def test_task_rejection_export_cancel_and_stream_close(tmp_path):
    root = write_tiny_checkpoint(tmp_path / "trocr", "trocr")
    with OCREngine(root) as engine, Image.new("RGB", (16, 12), "white") as image:
        for task in ["ocr.document", "ocr.detect", "ocr.handwriting", "ocr.formula"]:
            with pytest.raises(ValueError):
                engine.recognize(image, task=task, max_tokens=2)
        with pytest.raises(ValueError):
            engine.recognize(image, output_format="markdown")
        with pytest.raises(ValueError):
            engine.recognize(image, language="en")
        result = engine.recognize(image, max_tokens=2)
        result.save(tmp_path / "out.json")
        result.save(tmp_path / "out.txt")
        with pytest.raises(ValueError):
            result.save(tmp_path / "out.html")
        with pytest.raises(ValueError):
            engine.recognize(image, region=(0, 0, 99, 99))
        event = threading.Event()
        event.set()
        with pytest.raises(CancelledError):
            engine.recognize(image, cancel=event)
        stream = engine.stream(image, max_tokens=2)
        assert next(stream)["event"] == "page_started"
        stream.close()
        assert engine.recognize(image, max_tokens=2).pages
