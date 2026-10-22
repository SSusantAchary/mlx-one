"""Always-run architecture/metadata tests; no full weights or MLX runtime required."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx_one.core.registry import get_registration
from mlx_one.models.language.gpt_oss.config import GptOssConfig
from mlx_one.models.language.lfm2_moe.config import Lfm2MoeConfig
from mlx_one.text.chat import ChatTemplate
from mlx_one.text.harmony import HarmonyFinalParser
from mlx_one.text.loading import _eos_token_ids, _quantization_config

METADATA = Path(__file__).resolve().parents[1] / "fixtures/moe_metadata"


def tiny_config(family, **updates):
    data = dict(
        model_type=family,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    if family == "lfm2_moe":
        data.update(
            layer_types=["conv", "full_attention", "conv"],
            moe_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            num_dense_layers=1,
        )
    else:
        data.update(
            head_dim=32,
            num_local_experts=4,
            num_experts_per_tok=2,
            sliding_window=4,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 32.0,
                "truncate": False,
                "original_max_position_embeddings": 16,
            },
        )
    data.update(updates)
    return get_registration(family).config_class().from_dict(data)


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
def test_exact_checkpoint_metadata_maps_are_complete(family):
    root = METADATA / family
    data = json.loads((root / "config.json").read_text())
    config = get_registration(family).config_class().from_dict(data)
    quant = _quantization_config(data)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    assert config.max_position_embeddings == 131072
    assert len(config.layer_types) == config.num_hidden_layers
    if family == "lfm2_moe":
        assert quant["model.embed_tokens"]["bits"] == 6
        assert config.rope_theta == 10000000
        assert config.moe_intermediate_size == 1536 and config.intermediate_size == 11776
        assert config.layer_types[2] == "full_attention" and config.num_dense_layers == 2
        assert "model.layers.2.feed_forward.gate.weight" in index
        assert "model.layers.2.feed_forward.gate.scales" not in index
    else:
        assert len(quant) == 125 and quant["mode"] == "mxfp4"
        assert config.head_dim == 64 != config.hidden_size // config.num_attention_heads
        assert quant["model.layers.23.mlp.router"] == {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
        }
        assert quant["lm_head"]["mode"] == "affine"
        assert config.rope_scaling["truncate"] is False
        assert "model.layers.23.mlp.experts.gate_proj.scales" in index
        assert "model.layers.23.mlp.experts.gate_proj.biases" not in index
    generation = json.loads((root / "generation_config.json").read_text())
    assert _eos_token_ids(generation) == (
        (124900,) if family == "lfm2_moe" else (200002, 199999, 200012)
    )


@pytest.mark.parametrize("family", ["lfm2_moe", "gpt_oss"])
@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "other"},
        {"layer_types": ["bad"]},
        {"num_experts_per_tok": 5},
        {"num_experts_per_tok": True},
        {"layer_types": ["full_attention"]},
        {"typo_architecture_setting": 2},
        {"quantization": {"bits": 4, "mode": "awq"}},
    ],
)
def test_invalid_architecture_is_rejected(family, changes):
    with pytest.raises(ValueError):
        tiny_config(family, **changes)


def test_family_specific_validation():
    assert isinstance(tiny_config("gpt_oss"), GptOssConfig)
    assert isinstance(tiny_config("lfm2_moe"), Lfm2MoeConfig)
    for changes in (
        {"head_dim": 7},
        {"experts_per_token": 1},
        {"rope_scaling": {"rope_type": "linear"}},
        {"rope_scaling": {"rope_type": "yarn", "factor": -1}},
    ):
        with pytest.raises(ValueError):
            tiny_config("gpt_oss", **changes)
    with pytest.raises(ValueError, match="match layer_types"):
        tiny_config("lfm2_moe", full_attn_idxs=[0])
    for family in ("lfm2_moe", "gpt_oss"):
        with pytest.raises(ValueError, match="missing.*architecture fields"):
            get_registration(family).config_class().from_dict({"model_type": family})


@pytest.mark.parametrize("old_aliases", [False, True])
def test_harmony_stream_hides_every_nonfinal_channel_at_all_character_splits(old_aliases):
    start, channel, message, end, finish = (
        ("start", "channel", "message", "end", "return")
        if old_aliases
        else ("im_start", "meta_sep", "im_sep", "im_end", "fim_suffix")
    )

    def token(name):
        return f"<|{name}|>"

    prompt = token(start) + "assistant"
    text = (
        token(channel)
        + "analysis"
        + token(message)
        + "private scratchpad"
        + token(end)
        + token(start)
        + "assistant"
        + token(channel)
        + "commentary"
        + token(message)
        + "nonfinal"
        + token(end)
        + token(start)
        + "assistant"
        + token(channel)
        + "final"
        + token(message)
        + "Hello world!"
        + token(finish)
    )
    for split in range(len(text) + 1):
        parser = HarmonyFinalParser(prompt)
        actual = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.flush()
        assert actual == "Hello world!"
    parser = HarmonyFinalParser(prompt)
    assert "".join(parser.feed(char) for char in text) + parser.flush() == "Hello world!"


def test_actual_chat_templates_render_without_transformers():
    tokenizer = type("Tokenizer", (), {"bos_token": "<s>", "eos_token": "</s>"})()
    for family in ("lfm2_moe", "gpt_oss"):
        template = ChatTemplate.from_directory(METADATA / family, family, tokenizer)
        rendered = template.render([{"role": "user", "content": "Hello"}])
        assert "Hello" in rendered and "assistant" in rendered
        if family == "gpt_oss":
            assert rendered.endswith("<|start|>assistant")
            rendered = template.render(
                [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hello"}]
            )
            assert "<|channel|>final<|message|>Hi" in rendered
    with pytest.raises(ValueError, match="chat template"):
        ChatTemplate("gpt_oss", None, tokenizer).render([{"role": "user", "content": "Hi"}])


def test_harmony_prefilled_json_header_and_tool_recipient():
    prompt = "<|im_start|>assistant<|meta_sep|>final<|meta_start|>json<|im_sep|>"
    parser = HarmonyFinalParser(prompt)
    assert parser.feed('{"answer":42}<|fim_suffix|>') + parser.flush() == '{"answer":42}'
    parser = HarmonyFinalParser("<|start|>assistant")
    tool = " to=functions.foo<|channel|>commentary<|message|>hidden<|call|>"
    assert parser.feed(tool) + parser.flush() == ""


def test_production_import_has_no_upstream_dependency():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'mlx_lm', 'transformers'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import mlx_one
from mlx_one.core.registry import get_registration
for family in ('lfm2_moe', 'gpt_oss'):
    get_registration(family).config_class()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
