"""Exact original Nomic fused-attention/SwiGLU checkpoint contract."""

from mlx_one.core.weights import WeightContract


def sanitize_weights(weights, config):
    return dict(weights)


def weight_contract(config):
    dim, ff = config.hidden_size, config.intermediate_size
    expected = {
        "embeddings.word_embeddings.weight": (config.vocab_size, dim),
        "embeddings.token_type_embeddings.weight": (config.type_vocab_size, dim),
        "emb_ln.weight": (dim,),
        "emb_ln.bias": (dim,),
    }
    for i in range(config.num_hidden_layers):
        key = f"encoder.layers.{i}"
        expected[f"{key}.attn.Wqkv.weight"] = (dim * 3, dim)
        expected[f"{key}.attn.out_proj.weight"] = (dim, dim)
        for name in ("fc11", "fc12"):
            expected[f"{key}.mlp.{name}.weight"] = (ff, dim)
        expected[f"{key}.mlp.fc2.weight"] = (dim, ff)
        for name in ("norm1", "norm2"):
            expected[f"{key}.{name}.weight"] = (dim,)
            expected[f"{key}.{name}.bias"] = (dim,)
    return WeightContract(expected)
