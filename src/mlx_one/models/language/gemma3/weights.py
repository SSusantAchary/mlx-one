"""Gemma3 text tensor contract, retaining native offset-normalization weights."""

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
    q, kv = (
        config.num_attention_heads * config.head_dim,
        config.num_key_value_heads * config.head_dim,
    )
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
                    "pre_feedforward_layernorm.weight": (d,),
                    "post_feedforward_layernorm.weight": (d,),
                    "self_attn.q_proj.weight": (q, d),
                    "self_attn.k_proj.weight": (kv, d),
                    "self_attn.v_proj.weight": (kv, d),
                    "self_attn.o_proj.weight": (d, q),
                    "self_attn.q_norm.weight": (config.head_dim,),
                    "self_attn.k_norm.weight": (config.head_dim,),
                    "mlp.gate_proj.weight": (inner, d),
                    "mlp.up_proj.weight": (inner, d),
                    "mlp.down_proj.weight": (d, inner),
                }.items()
            }
        )
    return WeightContract(expected)
