"""Backend-free eligibility for the measured GPT-OSS prefill profile."""


def supports_prefill_sort(family, dtype, quantization, config, device, cache_types):
    """Restrict public opt-in to the measured model, storage and device envelope."""
    dimensions = {
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "swiglu_limit": 7,
    }
    if (
        family != "gpt_oss"
        or dtype != "bfloat16"
        or any(getattr(config, key, None) != value for key, value in dimensions.items())
        or (device or {}).get("device_name") != "Apple M4"
        or (device or {}).get("memory_size") != 32 * 1024**3
        or cache_types not in (("bf16", "bf16"), ("f16", "f16"))
        or any(
            quantization.get(key) != value
            for key, value in {"bits": 4, "group_size": 32, "mode": "mxfp4"}.items()
        )
    ):
        return False
    affine4 = {"bits": 4, "group_size": 64, "mode": "affine"}
    affine8 = {"bits": 8, "group_size": 64, "mode": "affine"}
    if any(quantization.get(path) != affine4 for path in ("model.embed_tokens", "lm_head")):
        return False
    for layer in range(24):
        if quantization.get(f"model.layers.{layer}.mlp.router") != affine8:
            return False
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if quantization.get(f"model.layers.{layer}.self_attn.{projection}") != affine4:
                return False
    return True
