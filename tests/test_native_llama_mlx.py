"""Opt-in tiny-MLX execution tests for native Llama-family models."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.llama.config import LlamaConfig
from mlx_one.models.language.llama.model import LlamaForCausalLM
from mlx_one.text.generation import stream_generate
from mlx_one.text.loading import LoadedTextModel
from mlx_one.text.schemas import TextGenerationOptions


def tiny_config() -> LlamaConfig:
    return LlamaConfig.from_dict(
        {
            "model_type": "llama",
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "max_position_embeddings": 32,
            "tie_word_embeddings": False,
            "eos_token_id": [1, 7],
        }
    )


def test_llama_shapes_explicit_projection_width_and_cache_equivalence() -> None:
    config = tiny_config()
    model = LlamaForCausalLM(config)
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (16, 12)
    assert model.model.layers[0].self_attn.o_proj.weight.shape == (12, 16)
    tokens = mx.array([[2, 3, 4]])
    full = model(tokens, output_hidden_states=True)
    caches = model.make_cache()
    cached = mx.concatenate(
        [model(tokens[:, index : index + 1], cache=caches).logits for index in range(3)],
        axis=1,
    )
    mx.eval(full.logits, cached)
    assert full.logits.shape == (1, 3, 32)
    assert full.last_hidden_state.shape == (1, 3, 12)
    assert len(full.hidden_states) == 3
    assert [cache.offset for cache in caches] == [3, 3]
    assert mx.max(mx.abs(full.logits - cached)).item() < 1e-4


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        return [2] if text else []

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token) for token in token_ids)


class _ScriptedModel:
    config = SimpleNamespace(max_position_embeddings=16)

    def __init__(self, token: int) -> None:
        self.token = token

    def make_cache(self) -> tuple[object, ...]:
        return ()

    def __call__(self, input_ids: object, *, cache: object) -> ModelOutput:
        del input_ids, cache
        logits = mx.where(mx.arange(8) == self.token, 10.0, -10.0)[None, None, :]
        return ModelOutput(logits, mx.zeros((1, 1, 1)))


@pytest.mark.parametrize("eos_token", [1, 7])
def test_generation_stops_on_every_configured_eos_token(eos_token: int) -> None:
    bundle = LoadedTextModel(
        _ScriptedModel(eos_token),
        _Tokenizer(),
        Path("."),
        "synthetic-llama",
        None,
        eos_token_ids=(1, 7),
    )
    chunks = tuple(
        stream_generate(bundle, "prompt", options=TextGenerationOptions(max_tokens=2))
    )
    assert len(chunks) == 1
    assert chunks[0].finish_reason == "stop"
