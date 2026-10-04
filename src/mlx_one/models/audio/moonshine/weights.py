"""Strict original Hugging Face Moonshine tensor mapping."""

from typing import Any

from mlx_one.core.weights import WeightContract
from mlx_one.models.audio.moonshine.config import MoonshineConfig


def sanitize_weights(weights: dict[str, Any], config: MoonshineConfig) -> dict[str, Any]:
    result = {}
    for name, value in weights.items():
        # The loader validates an optional tied alias before dropping it.
        if name == "proj_out.weight":
            continue
        if name in {f"model.encoder.conv{i}.weight" for i in (1, 2, 3)}:
            value = value.transpose(0, 2, 1)
        result[name] = value
    return result


def weight_contract(config: MoonshineConfig) -> WeightContract:
    d, f = config.hidden_size, config.intermediate_size
    expected = {
        "model.encoder.conv1.weight": (d, 127, 1),
        "model.encoder.conv2.weight": (2 * d, 7, d),
        "model.encoder.conv2.bias": (2 * d,),
        "model.encoder.conv3.weight": (d, 3, 2 * d),
        "model.encoder.conv3.bias": (d,),
        "model.encoder.groupnorm.weight": (d,),
        "model.encoder.groupnorm.bias": (d,),
        "model.encoder.layer_norm.weight": (d,),
        "model.decoder.embed_tokens.weight": (config.vocab_size, d),
        "model.decoder.norm.weight": (d,),
    }
    for side, count in (
        ("encoder", config.encoder_num_hidden_layers),
        ("decoder", config.decoder_num_hidden_layers),
    ):
        for index in range(count):
            prefix = f"model.{side}.layers.{index}"
            attentions = ("self_attn", "encoder_attn") if side == "decoder" else ("self_attn",)
            for attention in attentions:
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    expected[f"{prefix}.{attention}.{projection}.weight"] = (d, d)
            norms = ("input_layernorm", "post_attention_layernorm")
            if side == "decoder":
                norms += ("final_layernorm",)
            for norm in norms:
                expected[f"{prefix}.{norm}.weight"] = (d,)
            first = 2 * f if side == "decoder" else f
            expected[f"{prefix}.mlp.fc1.weight"] = (first, d)
            expected[f"{prefix}.mlp.fc1.bias"] = (first,)
            expected[f"{prefix}.mlp.fc2.weight"] = (d, f)
            expected[f"{prefix}.mlp.fc2.bias"] = (d,)
    return WeightContract(expected)
