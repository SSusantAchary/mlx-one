"""Strict original T5 parameter contract and declared tied/unused source aliases."""

from mlx_one.core.weights import WeightContract


def weight_contract(config):
    dim, inner, ff = config.d_model, config.num_heads * config.d_kv, config.d_ff
    expected = {"shared.weight": (config.vocab_size, dim)}
    for stack, count in (("encoder", config.num_layers), ("decoder", config.num_decoder_layers)):
        expected[f"{stack}.final_layer_norm.weight"] = (dim,)
        for i in range(count):
            base = f"{stack}.block.{i}.layer"
            attentions = [(0, "SelfAttention")]
            if stack == "decoder":
                attentions.append((1, "EncDecAttention"))
            for index, name in attentions:
                key = f"{base}.{index}"
                expected[f"{key}.layer_norm.weight"] = (dim,)
                for projection in ("q", "k", "v", "o"):
                    expected[f"{key}.{name}.{projection}.weight"] = (
                        (dim, inner) if projection == "o" else (inner, dim)
                    )
            if i == 0:
                expected[f"{base}.0.SelfAttention.relative_attention_bias.weight"] = (
                    config.relative_attention_num_buckets,
                    config.num_heads,
                )
            key = f"{base}.{2 if stack == 'decoder' else 1}"
            expected[f"{key}.layer_norm.weight"] = (dim,)
            expected[f"{key}.DenseReluDense.wi.weight"] = (ff, dim)
            expected[f"{key}.DenseReluDense.wo.weight"] = (dim, ff)
    return WeightContract(expected)


def sanitize_weights(weights, config):
    result = dict(weights)
    for name in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"):
        if name in result:
            import mlx.core as mx

            if (
                "shared.weight" not in result
                or not mx.array_equal(result[name], result["shared.weight"]).item()
            ):
                raise ValueError("T5 tied weight alias does not match shared.weight")
            del result[name]
    # This historical publisher tensor is unused by HF's cross-attention path.
    unused = "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"
    if unused in result:
        if result[unused].shape != (config.relative_attention_num_buckets, config.num_heads):
            raise ValueError("invalid historical unused cross-attention bias")
        del result[unused]
    return result
