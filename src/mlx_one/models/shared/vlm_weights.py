"""Publisher/MLX converter names mapped to the existing native VLM modules."""

from __future__ import annotations


def canonical_name(name: str, family: str) -> str:
    if family == "lfm2_vl":
        if name.startswith("model."):
            name = name.removeprefix("model.")
        name = name.replace("language_model.model.", "language_model.", 1)
        if name.startswith("vision_tower."):
            name = "vision_model." + name.removeprefix("vision_tower.")
        name = name.replace("vision_model.vision_encoder.", "vision_model.encoder.")
        name = name.replace("vision_model.vision_embeddings.", "vision_model.embeddings.")
        name = name.replace("vision_model.vision_post_layernorm.", "vision_model.post_layernorm.")
        name = name.replace("vision_model.embeddings.", "vision_model.")
        name = name.replace("vision_model.encoder.layers.", "vision_model.blocks.")
        name = name.replace(".mlp.fc1.", ".fc1.").replace(".mlp.fc2.", ".fc2.")
        for source, target in (
            ("layer_norm", "projector_norm"),
            ("linear_1", "projector_1"),
            ("linear_2", "projector_2"),
        ):
            name = name.replace("multi_modal_projector." + source, "vision_model." + target)
        return name
    if name.startswith("model.language_model.") or name.startswith("model.visual."):
        name = name.removeprefix("model.")
    elif name.startswith("model."):
        name = "language_model." + name.removeprefix("model.")
    name = name.replace("language_model.model.", "language_model.", 1)
    if name.startswith("language_model.lm_head."):
        name = name.removeprefix("language_model.")
    if name.startswith("vision_tower."):
        name = "visual." + name.removeprefix("vision_tower.")
    return name.replace(".merger.mlp.0.", ".merger.mlp_0.").replace(
        ".merger.mlp.2.", ".merger.mlp_2."
    )


def patch_weight(weight, channels):
    """Flatten HF OITHW or MLX OTHWI Conv3D into processor CTHW patch order."""
    if weight.ndim == 5:
        if weight.shape[-1] == channels:
            weight = weight.transpose(0, 4, 1, 2, 3)
        elif weight.shape[1] != channels:
            raise ValueError("unrecognized vision patch projection layout")
        return weight.reshape(weight.shape[0], -1)
    return weight


def quantization_names(quantization, family):
    """Keep module-specific formats attached to their renamed packed tensors."""
    return {
        canonical_name(k + ".weight", family).removesuffix(".weight")
        if isinstance(v, dict) or v is False
        else k: v
        for k, v in quantization.items()
    }
