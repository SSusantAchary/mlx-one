"""Strict publisher tensor mapping, without importing an inference backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlx_one.core.weights import WeightContract, WeightContractError
from mlx_one.models.audio.qwen3_asr.config import QwenASRConfig


def weight_contract(config: QwenASRConfig) -> WeightContract:
    a, t = config.audio_config, config.text_config
    shapes: dict[str, tuple[int, ...]] = {}

    def linear(name: str, out: int, inp: int, bias: bool = True) -> None:
        shapes[name + ".weight"] = (out, inp)
        if bias:
            shapes[name + ".bias"] = (out,)

    def norm(name: str, width: int, bias: bool = True) -> None:
        shapes[name + ".weight"] = (width,)
        if bias:
            shapes[name + ".bias"] = (width,)

    for i in range(1, 4):
        shapes[f"audio_tower.conv2d{i}.weight"] = (
            a.downsample_hidden_size,
            3,
            3,
            1 if i == 1 else a.downsample_hidden_size,
        )
        shapes[f"audio_tower.conv2d{i}.bias"] = (a.downsample_hidden_size,)
    linear("audio_tower.conv_out", a.d_model, a.downsample_hidden_size * 16, False)
    for i in range(a.encoder_layers):
        p = f"audio_tower.layers.{i}"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(p + ".self_attn." + name, a.d_model, a.d_model)
        norm(p + ".self_attn_layer_norm", a.d_model)
        norm(p + ".final_layer_norm", a.d_model)
        linear(p + ".fc1", a.encoder_ffn_dim, a.d_model)
        linear(p + ".fc2", a.d_model, a.encoder_ffn_dim)
    norm("audio_tower.ln_post", a.d_model)
    linear("audio_tower.proj1", a.d_model, a.d_model)
    linear("audio_tower.proj2", a.output_dim, a.d_model)
    shapes["model.embed_tokens.weight"] = (t.vocab_size, t.hidden_size)
    for i in range(t.num_hidden_layers):
        p = f"model.layers.{i}"
        for name, out in (
            ("q_proj", t.num_attention_heads * t.head_dim),
            ("k_proj", t.num_key_value_heads * t.head_dim),
            ("v_proj", t.num_key_value_heads * t.head_dim),
        ):
            linear(p + ".self_attn." + name, out, t.hidden_size, False)
        linear(p + ".self_attn.o_proj", t.hidden_size, t.num_attention_heads * t.head_dim, False)
        for name in ("q_norm", "k_norm"):
            norm(p + ".self_attn." + name, t.head_dim, False)
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm(p + "." + name, t.hidden_size, False)
        for name in ("gate_proj", "up_proj"):
            linear(p + ".mlp." + name, t.intermediate_size, t.hidden_size, False)
        linear(p + ".mlp.down_proj", t.hidden_size, t.intermediate_size, False)
    norm("model.norm", t.hidden_size, False)
    if config.model_type == "qwen3_forced_aligner":
        linear("lm_head", config.classify_num, t.hidden_size, False)
    return WeightContract(shapes)


def sanitize_weights(weights: Mapping[str, Any], config: QwenASRConfig) -> dict[str, Any]:
    if any(not k.startswith("thinker.") for k in weights):
        raise WeightContractError("Qwen audio requires the publisher thinker.* tensor layout")
    for name, value in weights.items():
        if hasattr(value, "dtype") and str(value.dtype).rsplit(".", 1)[-1] not in {
            "bfloat16",
            "float16",
            "float32",
        }:
            raise WeightContractError(f"unsupported Qwen tensor precision: {name}")
    result = {k.removeprefix("thinker."): v for k, v in weights.items()}
    if config.model_type == "qwen3_asr" and "lm_head.weight" in result:
        import mlx.core as mx

        head = result.pop("lm_head.weight")
        embed = result.get("model.embed_tokens.weight")
        if embed is None or head.shape != embed.shape or not mx.all(head == embed).item():
            raise WeightContractError("ASR output head is not tied to its embedding")
    for i in range(1, 4):
        name = f"audio_tower.conv2d{i}.weight"
        if name in result:
            if result[name].ndim != 4:
                raise WeightContractError("audio convolution weights must be OIHW")
            result[name] = result[name].transpose(0, 2, 3, 1)
    weight_contract(config).validate(result)
    return result
