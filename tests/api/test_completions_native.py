"""Opt-in native HTTP/Python parity and lifecycle, using a tiny GPT-2 model."""

import json
import os
from pathlib import Path

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
from fastapi.testclient import TestClient

from mlx_one.models.language.gpt2.config import GPT2Config
from mlx_one.models.language.gpt2.model import GPT2LMHeadModel
from mlx_one.server.app import create_app
from mlx_one.server.config import ServerConfig
from mlx_one.server.generation_engine import GenerationEngine
from mlx_one.server.model_manager import ModelManager
from mlx_one.text import LoadedTextModel, TextGenerationOptions, generate
from mlx_one.text.tokenizer import GPT2Tokenizer, bytes_to_unicode


def test_native_raw_completion_matches_python_and_releases_cache_and_model(monkeypatch, tmp_path):
    vocabulary = {character: index for index, character in enumerate(bytes_to_unicode().values())}
    vocabulary["<|endoftext|>"] = 256
    tokenizer = GPT2Tokenizer(vocabulary, ())
    config = GPT2Config(n_embd=8, n_layer=2, n_head=2, n_positions=32, n_ctx=32,
                        vocab_size=257, bos_token_id=256, eos_token_id=256,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    mx.random.seed(7)
    bundle = LoadedTextModel(GPT2LMHeadModel(config), tokenizer, Path("."), "tiny-gpt2", None,
                             context_length=32, eos_token_ids=(256,), chat_template=None)
    bundle.model.eval()
    options = TextGenerationOptions(max_tokens=4, temperature=0)
    expected = generate(bundle, "raw", options=options)
    monkeypatch.setattr("mlx_one.server.model_manager.load_text_model", lambda *a, **kw: bundle)
    manager = ModelManager()
    manager.load("tiny-gpt2", offline=True)
    engine = GenerationEngine(manager)
    app = create_app(manager, engine, ui_dir=tmp_path,
                     config=ServerConfig(warmup=False, cache_prompt=False))
    with TestClient(app) as client:
        payload = {"model": "tiny-gpt2", "prompt": "raw", "max_tokens": 4, "temperature": 0}
        result = client.post("/v1/completions", json=payload)
        assert result.status_code == 200, result.text
        assert result.json()["choices"][0]["text"] == expected.text
        assert result.json()["choices"][0]["finish_reason"] == expected.finish_reason
        assert result.json()["usage"]["completion_tokens"] == expected.generation_tokens
        assert engine.cache_stats()["active_handles"] == 0
        streamed = client.post("/v1/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200, streamed.text
        chunks = [json.loads(line[6:]) for line in streamed.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == expected.text
        assert engine.cache_stats()["active_handles"] == 0
        overflow = client.post("/v1/completions", json={**payload, "max_tokens": 32})
        assert overflow.status_code == 400
    with pytest.raises(RuntimeError, match="no model"):
        manager.current_model()
