"""Native LFM2-VL / SigLIP2 publisher and MLX packed-weight contracts."""

from mlx_one.core.weights import WeightContract
from mlx_one.models.language.lfm2.weights import weight_contract as text_contract
from mlx_one.models.shared.vlm_weights import canonical_name


def sanitize_weights(weights, config):
    result = {}
    for original, value in weights.items():
        if original.endswith(("rotary_emb.inv_freq", "position_ids")):
            continue
        name = canonical_name(original, "lfm2_vl")
        if name.endswith("conv.conv.weight"):
            name = name.removesuffix(".weight")
            if value.ndim == 3:
                if value.shape[-1] == 1:
                    value = value.squeeze(-1)
                elif value.shape[1] == 1:
                    value = value.squeeze(1)
                else:
                    raise ValueError("unsupported LFM2 convolution weight layout")
        if name.startswith("lm_head.") and config.tie_word_embeddings:
            continue
        if name in result:
            raise ValueError(f"duplicate LFM2-VL tensor mapping: {name}")
        result[name] = value
    return result


def weight_contract(config):
    expected = {
        "language_model." + name.removeprefix("model."): shape
        for name, shape in text_contract(config.text_config).expected.items()
        if name.startswith("model.")
    }
    text, vision = config.text_config, config.vision_config
    hidden = vision.hidden_size
    patch_dim = vision.num_channels * vision.patch_size**2
    merged = hidden * vision.downsample_factor**2
    positions = vision.num_patches or (vision.image_size // vision.patch_size) ** 2
    expected.update(
        {
            "vision_model.patch_embedding.weight": (hidden, patch_dim),
            "vision_model.patch_embedding.bias": (hidden,),
            "vision_model.position_embedding.weight": (positions, hidden),
            "vision_model.post_layernorm.weight": (hidden,),
            "vision_model.post_layernorm.bias": (hidden,),
        }
    )
    if vision.projector_hidden_size is None:
        expected.update(
            {
                "vision_model.projector.weight": (vision.projection_dim, merged),
                "vision_model.projector.bias": (vision.projection_dim,),
            }
        )
    else:
        for name, out_dim, in_dim in (
            ("projector_1", vision.projector_hidden_size, merged),
            ("projector_2", vision.projection_dim, vision.projector_hidden_size),
        ):
            expected[f"vision_model.{name}.weight"] = (out_dim, in_dim)
            if vision.projector_bias:
                expected[f"vision_model.{name}.bias"] = (out_dim,)
    if vision.use_projection_layernorm:
        expected.update(
            {
                "vision_model.projector_norm.weight": (merged,),
                "vision_model.projector_norm.bias": (merged,),
            }
        )
    for index in range(vision.num_hidden_layers):
        prefix = f"vision_model.blocks.{index}."
        for norm in ("layer_norm1", "layer_norm2"):
            expected[prefix + norm + ".weight"] = (hidden,)
            expected[prefix + norm + ".bias"] = (hidden,)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            expected[prefix + "self_attn." + name + ".weight"] = (hidden, hidden)
            expected[prefix + "self_attn." + name + ".bias"] = (hidden,)
        expected.update(
            {
                prefix + "fc1.weight": (vision.intermediate_size, hidden),
                prefix + "fc1.bias": (vision.intermediate_size,),
                prefix + "fc2.weight": (hidden, vision.intermediate_size),
                prefix + "fc2.bias": (hidden,),
            }
        )
    if not config.tie_word_embeddings:
        expected["lm_head.weight"] = (text.vocab_size, text.hidden_size)
    return WeightContract(expected)
