# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
"""GPT-OSS native MLX and interleaved MXFP4 weight mapping; see UPSTREAM_LICENSE."""

from mlx_one.core.weights import WeightContract


def sanitize_weights(weights, config):
    import mlx.core as mx

    result = {}
    for name, value in weights.items():
        if name.endswith("rotary_emb.inv_freq"):
            continue
        if "_blocks" in name:
            if value.dtype != mx.uint8 or value.shape[-1] % 4:
                raise ValueError(f"malformed MXFP4 blocks: {name}")
            value = value.view(mx.uint32).flatten(-2)
            name = name.replace("_blocks", ".weight")
        if "_scales" in name:
            name = name.replace("_scales", ".scales")
        if "gate_up_proj_bias" in name:
            result[name.replace("gate_up_proj_bias", "gate_proj.bias")] = mx.contiguous(
                value[..., ::2]
            )
            result[name.replace("gate_up_proj_bias", "up_proj.bias")] = mx.contiguous(
                value[..., 1::2]
            )
        elif "gate_up_proj" in name:
            if value.shape[-2] != 2 * config.intermediate_size:
                raise ValueError(f"malformed interleaved GPT-OSS experts: {name}")
            result[name.replace("gate_up_proj", "gate_proj")] = mx.contiguous(value[..., ::2, :])
            result[name.replace("gate_up_proj", "up_proj")] = mx.contiguous(value[..., 1::2, :])
        elif "down_proj_bias" in name:
            result[name.replace("down_proj_bias", "down_proj.bias")] = value
        else:
            if name in result:
                raise ValueError(f"duplicate sanitized GPT-OSS tensor: {name}")
            result[name] = value
    return result


def weight_contract(config):
    d, h, kv, head = (
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    e, f = config.num_local_experts, config.intermediate_size
    expected = {
        "model.embed_tokens.weight": (config.vocab_size, d),
        "model.norm.weight": (d,),
        "lm_head.weight": (config.vocab_size, d),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        for name in ("input_layernorm", "post_attention_layernorm"):
            expected[f"{prefix}.{name}.weight"] = (d,)
        expected[f"{prefix}.self_attn.sinks"] = (h,)
        for name, out, inputs in (
            ("q_proj", h * head, d),
            ("k_proj", kv * head, d),
            ("v_proj", kv * head, d),
            ("o_proj", d, h * head),
        ):
            expected[f"{prefix}.self_attn.{name}.weight"] = (out, inputs)
            expected[f"{prefix}.self_attn.{name}.bias"] = (out,)
        expected[f"{prefix}.mlp.router.weight"] = (e, d)
        expected[f"{prefix}.mlp.router.bias"] = (e,)
        for name, out, inputs in (("gate_proj", f, d), ("up_proj", f, d), ("down_proj", d, f)):
            expected[f"{prefix}.mlp.experts.{name}.weight"] = (e, out, inputs)
            expected[f"{prefix}.mlp.experts.{name}.bias"] = (e, out)
    return WeightContract(expected)
