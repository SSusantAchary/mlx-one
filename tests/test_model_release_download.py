"""Pinned ranged downloads must reject corrupt data and retain resumable chunks."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/qualification/model_release_download.py"
SPEC = importlib.util.spec_from_file_location("release_range_download", PATH)
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


def fixture(tmp_path, monkeypatch, *, corrupt=False):
    import huggingface_hub

    data = b"checkpoint"
    file = dict(path="model.safetensors", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    requests = []

    class Response:
        status_code = 206
        headers = {"content-range": f"bytes 0-{len(data) - 1}/{len(data)}"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def iter_bytes(self, **kwargs):
            yield b"corruption" if corrupt else data

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def stream(self, method, url, headers):
            requests.append(headers["Range"])
            return Response()

    monkeypatch.setattr(downloads.httpx, "Client", Client)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    return data, file, requests


def test_range_transfer_checks_digest_and_reuses_owned_completed_file(tmp_path, monkeypatch):
    data, file, requests = fixture(tmp_path, monkeypatch)
    path = downloads.download("test/model", "a" * 40, file, tmp_path)
    assert path.read_bytes() == data and requests == ["bytes=0-9"]
    assert downloads.download("test/model", "a" * 40, file, tmp_path) == path
    assert len(requests) == 1


def test_corrupt_assembled_checkpoint_is_not_installed(tmp_path, monkeypatch):
    _, file, _ = fixture(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="checksum"):
        downloads.download("test/model", "a" * 40, file, tmp_path)
    assert not (tmp_path / "model.safetensors").exists()
