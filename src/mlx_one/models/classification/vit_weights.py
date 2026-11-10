"""ViT tensor names follow the publisher; convolution kernels use MLX layout."""

from mlx_one.core.weights import WeightContract


def sanitize_weights(weights, config):
    output = dict(weights)
    name = "vit.embeddings.patch_embeddings.projection.weight"
    if name in output:
        expected = (config.hidden_size, config.num_channels, config.patch_size, config.patch_size)
        canonical = (config.hidden_size, config.patch_size, config.patch_size, config.num_channels)
        if output[name].shape == expected:
            output[name] = output[name].transpose(0, 2, 3, 1)
        elif output[name].shape != canonical:
            raise ValueError("ViT patch projection has incompatible channels/kernel")
    return output


def weight_contract(config):
    dim, ff = config.hidden_size, config.intermediate_size
    expected = {
        "vit.embeddings.cls_token": (1, 1, dim),
        "vit.embeddings.position_embeddings": (
            1,
            (config.image_size // config.patch_size) ** 2 + 1,
            dim,
        ),
        "vit.embeddings.patch_embeddings.projection.weight": (
            dim,
            config.patch_size,
            config.patch_size,
            config.num_channels,
        ),
        "vit.embeddings.patch_embeddings.projection.bias": (dim,),
        "vit.layernorm.weight": (dim,),
        "vit.layernorm.bias": (dim,),
        "classifier.weight": (config.num_labels, dim),
        "classifier.bias": (config.num_labels,),
    }
    for i in range(config.num_hidden_layers):
        key = f"vit.encoder.layer.{i}"
        for name in ("query", "key", "value"):
            expected[f"{key}.attention.attention.{name}.weight"] = (dim, dim)
            expected[f"{key}.attention.attention.{name}.bias"] = (dim,)
        for name, shape in (
            ("attention.output.dense", (dim, dim)),
            ("intermediate.dense", (ff, dim)),
            ("output.dense", (dim, ff)),
        ):
            expected[f"{key}.{name}.weight"] = shape
            expected[f"{key}.{name}.bias"] = (shape[0],)
        for name in ("layernorm_before", "layernorm_after"):
            expected[f"{key}.{name}.weight"] = (dim,)
            expected[f"{key}.{name}.bias"] = (dim,)
    return WeightContract(expected)
