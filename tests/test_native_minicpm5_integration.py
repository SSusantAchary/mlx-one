"""Opt-in pinned real-checkpoint smoke tests for MiniCPM5 MLX models."""

from __future__ import annotations

import os

import pytest

if os.environ.get("MLX_ONE_RUN_MINICPM5_INTEGRATION") != "1":
    pytest.skip(
        "set MLX_ONE_RUN_MINICPM5_INTEGRATION=1 to run pinned MiniCPM5 checkpoints",
        allow_module_level=True,
    )

import mlx.core as mx

from mlx_one.text.generation import stream_chat
from mlx_one.text.loading import load_text_model
from mlx_one.text.schemas import TextGenerationOptions

MINICPM5_1B_REVISION = "9879b18bf2928355fcdf4287635388a3665a40cb"
MINICPM5_2B_REVISION = "8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3"


@pytest.mark.parametrize(
    ("model_id", "revision", "hidden", "layers"),
    [
        ("openbmb/MiniCPM5-1B-MLX", MINICPM5_1B_REVISION, 1536, 24),
        ("openbmb/MiniCPM5-2B-MLX", MINICPM5_2B_REVISION, 2048, 42),
    ],
    ids=["1b", "2b"],
)
def test_pinned_minicpm5_checkpoint_loads_renders_and_executes(
    model_id: str, revision: str, hidden: int, layers: int
) -> None:
    bundle = load_text_model(model_id, revision=revision)
    config = bundle.model.config
    assert bundle.architecture == "llama"
    assert bundle.revision == revision
    assert bundle.context_length == 131072
    assert bundle.quantization == {"bits": 4, "group_size": 64, "mode": "affine"}
    assert bundle.eos_token_ids == (1, 130073)
    assert (config.hidden_size, config.num_hidden_layers, config.head_dim) == (
        hidden,
        layers,
        128,
    )
    assert bundle.chat_template is not None
    messages = [
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Again."},
    ]
    thinking = bundle.chat_template.render(messages, enable_thinking=True)
    no_thinking = bundle.chat_template.render(messages, enable_thinking=False)
    assert thinking.endswith("<|im_start|>assistant\n<think>\n")
    assert no_thinking.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    output = bundle.model(mx.array([[bundle.tokenizer.bos_token_id]]))
    mx.eval(output.logits)
    assert output.logits.shape == (1, 1, config.vocab_size)
    chunks = tuple(
        stream_chat(
            bundle,
            [{"role": "user", "content": "Say hello."}],
            options=TextGenerationOptions(max_tokens=1),
            template_options={"enable_thinking": False},
        )
    )
    assert chunks[-1].finish_reason in {"stop", "length"}
