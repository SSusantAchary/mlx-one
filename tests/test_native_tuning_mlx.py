"""Opt-in MLX tests for native dense and quantized LoRA adapters."""

from __future__ import annotations

import json
import os

import pytest

if os.environ.get("MLX_ONE_RUN_MLX_TESTS") != "1":
    pytest.skip("set MLX_ONE_RUN_MLX_TESTS=1 on a Metal host", allow_module_level=True)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_one.models.language.qwen2.config import Qwen2Config
from mlx_one.models.language.qwen2.model import Qwen2ForCausalLM
from mlx_one.tuning import apply_lora, load_adapter, save_adapter


def _model(*, quantized: bool = False) -> Qwen2ForCausalLM:
    mx.random.seed(7)
    config = Qwen2Config.from_dict(
        {
            "model_type": "qwen2",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
        }
    )
    model = Qwen2ForCausalLM(config)
    if quantized:
        nn.quantize(model, group_size=8, bits=4, mode="affine")
    return model


@pytest.mark.parametrize("quantized", [False, True], ids=["lora", "qlora"])
def test_native_adapter_round_trip_and_trainable_boundary(tmp_path, quantized: bool) -> None:
    model = _model(quantized=quantized)
    targets = apply_lora(model, num_layers=1, rank=4, scale=2.0, dropout=0.0)
    assert set(targets) == {
        "layers.1.self_attn.q_proj",
        "layers.1.self_attn.k_proj",
        "layers.1.self_attn.v_proj",
        "layers.1.self_attn.o_proj",
    }
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable
    assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)
    output = model(mx.array([[1, 2, 3]])).logits
    mx.eval(output)
    adapter = save_adapter(
        model,
        tmp_path,
        {
            "schema_version": 1,
            "base_model_id": "synthetic/qwen2",
            "base_revision": "a" * 40,
            "num_layers": 1,
            "rank": 4,
            "scale": 2.0,
            "dropout": 0.0,
            "target_modules": [],
            "resolved_targets": list(targets),
        },
    )
    assert adapter.is_file()
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert len(config["weights_sha256"]) == 64
    reloaded = _model(quantized=quantized)
    load_adapter(
        reloaded,
        tmp_path,
        base_model_id="synthetic/qwen2",
        base_revision="a" * 40,
    )
    restored = reloaded(mx.array([[1, 2, 3]])).logits
    mx.eval(restored)
    assert mx.max(mx.abs(output - restored)).item() < 1e-5
