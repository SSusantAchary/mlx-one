"""Phi-3 checkpoint contract: packed Q/K/V and gate/up remain packed."""

from mlx_one.core.weights import WeightContract


def sanitize_weights(weights, config):
    return {
        k: v
        for k, v in weights.items()
        if not k.endswith("rotary_emb.inv_freq")
        and not (config.tie_word_embeddings and k == "lm_head.weight")
    }


def weight_contract(config):
    d, inner = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    expected = {"model.embed_tokens.weight": (config.vocab_size, d), "model.norm.weight": (d,)}
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (config.vocab_size, d)
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        expected.update(
            {
                prefix + k: v
                for k, v in {
                    "input_layernorm.weight": (d,),
                    "post_attention_layernorm.weight": (d,),
                    "self_attn.qkv_proj.weight": (q + 2 * kv, d),
                    "self_attn.o_proj.weight": (d, q),
                    "mlp.gate_up_proj.weight": (2 * inner, d),
                    "mlp.down_proj.weight": (d, inner),
                }.items()
            }
        )
    return WeightContract(expected)
