"""Opt-in tiny-MLX execution tests for native GPT-2."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx

from mlx_one.core.outputs import ModelOutput
from mlx_one.models.language.gpt2.config import GPT2Config
from mlx_one.models.language.gpt2.model import GPT2LMHeadModel, gelu_new
from mlx_one.models.language.gpt2.weights import weight_contract
from mlx_one.text.generation import _select_token, generate, stream_generate
from mlx_one.text.loading import LoadedTextModel, load_text_model
from mlx_one.text.schemas import TextGenerationOptions
from mlx_one.text.tokenizer import GPT2Tokenizer, bytes_to_unicode


def tiny_config(**updates: object) -> GPT2Config:
    values = {
        "model_type": "gpt2",
        "vocab_size": 257,
        "n_positions": 16,
        "n_ctx": 16,
        "n_embd": 8,
        "n_layer": 2,
        "n_head": 2,
        "bos_token_id": 256,
        "eos_token_id": 256,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }
    values.update(updates)
    return GPT2Config.from_dict(values)


def byte_tokenizer() -> GPT2Tokenizer:
    byte_map = bytes_to_unicode()
    vocabulary = {character: index for index, character in enumerate(byte_map.values())}
    vocabulary["<|endoftext|>"] = 256
    return GPT2Tokenizer(vocabulary, ())


def test_gpt2_shapes_fused_qkv_attentions_positions_and_tied_logits() -> None:
    config = tiny_config()
    model = GPT2LMHeadModel(config)
    tokens = mx.array([[1, 2, 3]])
    output = model(tokens, output_hidden_states=True, output_attentions=True)
    hidden, _, _ = model.transformer(tokens)
    tied = model.transformer.wte.as_linear(hidden)
    mx.eval(output.logits, tied)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.last_hidden_state.shape == (1, 3, config.n_embd)
    assert len(output.hidden_states) == config.n_layer + 1
    assert len(output.attentions) == config.n_layer
    assert output.attentions[0].shape == (1, config.n_head, 3, 3)
    assert model.transformer.h[0].attn.c_attn.weight.shape == (3 * config.n_embd, config.n_embd)
    assert mx.max(mx.abs(output.logits - tied)).item() == 0
    assert mx.max(output.attentions[0][..., 0, 1:]).item() == 0

    shifted = model(tokens, position_ids=mx.array([[1, 2, 3]])).logits
    mx.eval(shifted)
    assert mx.max(mx.abs(output.logits - shifted)).item() > 0


def test_gpt2_attention_mask_and_cache_equivalence() -> None:
    config = tiny_config()
    model = GPT2LMHeadModel(config)
    tokens = mx.array([[1, 2, 3, 4]])
    full = model(tokens).logits
    caches = model.make_cache()
    cached = mx.concatenate(
        [model(tokens[:, index : index + 1], cache=caches).logits for index in range(4)],
        axis=1,
    )
    masked = model(
        tokens,
        attention_mask=mx.array([[0, 1, 1, 1]]),
        output_attentions=True,
    )
    mx.eval(full, cached, masked.logits)
    assert [cache.offset for cache in caches] == [4, 4]
    assert mx.max(mx.abs(full - cached)).item() < 1e-4
    assert mx.max(masked.attentions[0][..., 1:, 0]).item() == 0
    assert not mx.any(mx.isnan(masked.logits)).item()
    with pytest.raises(ValueError, match="complete cached sequence"):
        model(tokens, attention_mask=mx.ones((1, 3)))


def test_gpt2_gelu_new_matches_formula() -> None:
    values = mx.array([-1.0, 0.0, 1.0])
    actual = gelu_new(values)
    expected = mx.array([-0.158808, 0.0, 0.841192])
    mx.eval(actual)
    assert mx.max(mx.abs(actual - expected)).item() < 1e-5


class ScriptedModel:
    def __init__(self, tokens: list[int]) -> None:
        self.config = SimpleNamespace(n_positions=32)
        self.tokens = tokens
        self.calls = 0

    def make_cache(self) -> tuple[object, ...]:
        return ()

    def __call__(self, input_ids, *, cache):
        del input_ids, cache
        token = self.tokens[min(self.calls, len(self.tokens) - 1)]
        self.calls += 1
        logits = mx.where(mx.arange(257) == token, 10.0, -10.0)[None, None, :]
        return ModelOutput(logits, mx.zeros((1, 1, 1)))


def test_gpt2_generation_greedy_stop_batch_stream_and_seeded_sampling() -> None:
    tokenizer = byte_tokenizer()
    ids = tokenizer.encode("ABCD")
    bundle = LoadedTextModel(ScriptedModel(ids), tokenizer, Path("."), "synthetic-gpt2", None)
    chunks = tuple(
        stream_generate(
            bundle,
            "x",
            options=TextGenerationOptions(max_tokens=4, stop=("BC",)),
        )
    )
    assert "".join(chunk.text for chunk in chunks) == "A"
    assert chunks[-1].finish_reason == "stop"

    batches = generate(
        LoadedTextModel(ScriptedModel([ids[0]]), tokenizer, Path("."), "synthetic", None),
        ["one", "two"],
        options=TextGenerationOptions(max_tokens=1),
    )
    assert len(batches) == 2
    assert all(result.text == "A" for result in batches)

    logits = mx.array([0.1, 0.2, 0.3, 4.0])
    options = TextGenerationOptions(temperature=0.8, top_k=2, top_p=0.9, seed=7)
    first = _select_token(logits, options, random.Random(7))
    second = _select_token(logits, options, random.Random(7))
    assert first == second
    assert first in {2, 3}


def _write_tokenizer(root: Path) -> None:
    byte_map = bytes_to_unicode()
    vocabulary = {character: index for index, character in enumerate(byte_map.values())}
    vocabulary["<|endoftext|>"] = 256
    (root / "vocab.json").write_text(json.dumps(vocabulary), encoding="utf-8")
    (root / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )


def _checkpoint_arrays(config: GPT2Config) -> dict[str, np.ndarray]:
    arrays = {}
    transposed_suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )
    for name, shape in weight_contract(config).expected.items():
        source_shape = tuple(reversed(shape)) if name.endswith(transposed_suffixes) else shape
        arrays[name] = np.zeros(source_shape, dtype=np.float32)
    arrays["lm_head.weight"] = arrays["transformer.wte.weight"]
    arrays["transformer.h.0.attn.bias"] = np.zeros((1, 1, 16, 16), dtype=np.bool_)
    arrays["transformer.h.0.attn.masked_bias"] = np.array(-1e4, dtype=np.float32)
    return arrays


@pytest.mark.parametrize("sharded", [False, True])
def test_gpt2_safe_loader_accepts_synthetic_unsharded_and_sharded(
    tmp_path: Path, sharded: bool
) -> None:
    config = tiny_config()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **config.extra,
                **{
                    field: getattr(config, field)
                    for field in (
                        "model_type",
                        "vocab_size",
                        "n_positions",
                        "n_ctx",
                        "n_embd",
                        "n_layer",
                        "n_head",
                        "bos_token_id",
                        "eos_token_id",
                        "resid_pdrop",
                        "embd_pdrop",
                        "attn_pdrop",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    _write_tokenizer(tmp_path)
    arrays = _checkpoint_arrays(config)
    if not sharded:
        save_file(arrays, tmp_path / "model.safetensors")
    else:
        names = sorted(arrays)
        midpoint = len(names) // 2
        parts = (names[:midpoint], names[midpoint:])
        weight_map = {}
        for index, selected in enumerate(parts, 1):
            filename = f"model-{index:05d}-of-00002.safetensors"
            save_file({name: arrays[name] for name in selected}, tmp_path / filename)
            weight_map.update({name: filename for name in selected})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}), encoding="utf-8"
        )
    loaded = load_text_model(tmp_path, offline=True)
    output = loaded.model(mx.array([[1, 2]])).logits
    mx.eval(output)
    assert output.shape == (1, 2, config.vocab_size)
