"""Map the pinned MFLUX Q4 checkpoint layouts to native MLX-One modules."""

from __future__ import annotations

from .checkpoint import GROUP_SIZE, ImageCheckpoint, file_digest, header_digest, tensor_header


def _native_name(family: str, component: str, name: str) -> str:
    if family == "flux2-klein":
        if component == "transformer":
            if name.startswith("time_guidance_embed.linear_"):
                return name.replace(
                    "time_guidance_embed.", "time_guidance_embed.timestep_embedder."
                )
            if name.startswith("transformer_blocks."):
                return name.replace(".attn.to_out.", ".attn.to_out.0.")
        if component == "vae":
            return name.replace(
                "decoder.mid_block.attentions.0.to_out.", "decoder.mid_block.attentions.0.to_out.0."
            )
    if family == "z-image-turbo":
        if component == "transformer":
            for source, target in (
                ("t_embedder.linear1.", "t_embedder.mlp.0."),
                ("t_embedder.linear2.", "t_embedder.mlp.2."),
                (
                    "all_final_layer.2-1.adaLN_modulation.0.",
                    "all_final_layer.2-1.adaLN_modulation.1.",
                ),
            ):
                if name.startswith(source):
                    return target + name.removeprefix(source)
        if component == "vae":
            for source, target in (
                ("decoder.conv_in.conv.", "decoder.conv_in."),
                ("decoder.conv_norm_out.norm.", "decoder.conv_norm_out."),
                ("decoder.conv_out.conv.", "decoder.conv_out."),
            ):
                if name.startswith(source):
                    return target + name.removeprefix(source)
    return name


def _unused(component: str, name: str, module) -> bool:
    if component == "text_encoder":
        if name in {"rotary_emb.inv_freq", "norm.weight"}:
            return True
        parts = name.split(".")
        return (
            len(parts) > 2
            and parts[0] == "layers"
            and parts[1].isdigit()
            and int(parts[1]) >= len(module.layers)
        )
    if component == "vae":
        return name.startswith(("encoder.", "quant_conv."))
    return False


def prepare_published_weights(checkpoint: ImageCheckpoint, component: str, module):
    """Verify a pinned asset and attach only the affine layers present in its stored layout."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    index = {}
    mappings = {}
    for filename, identity in checkpoint.shards(component).items():
        path = checkpoint.root / filename
        header = tensor_header(path)
        if (
            path.stat().st_size != identity["size"]
            or header_digest(header) != identity["layout"]
            or file_digest(path) != identity["sha256"]
        ):
            raise ValueError(f"published Q4 shard differs from the pinned artifact: {filename}")
        mapping = {}
        for name, spec in header.items():
            native = _native_name(checkpoint.profile.family, component, name)
            if _unused(component, native, module):
                continue
            if native in index:
                raise ValueError(f"duplicate published Q4 tensor: {native}")
            mapping[name] = native
            index[native] = spec
        mappings[path] = mapping

    packed = {name.removesuffix(".weight") for name, spec in index.items() if spec[0] == "U32"}
    linear_or_embedding = {
        name: layer
        for name, layer in module.named_modules()
        if isinstance(layer, (nn.Linear, nn.Embedding))
    }
    for name in packed:
        if name not in linear_or_embedding:
            raise ValueError(f"invalid published Q4 layer: {name}")
        out_dim, in_dim = linear_or_embedding[name].weight.shape
        if (
            in_dim % GROUP_SIZE
            or index.get(f"{name}.weight") != ["U32", [out_dim, in_dim // 8]]
            or index.get(f"{name}.scales") != ["BF16", [out_dim, in_dim // GROUP_SIZE]]
            or index.get(f"{name}.biases") != ["BF16", [out_dim, in_dim // GROUP_SIZE]]
        ):
            raise ValueError(f"invalid published Q4 affine layout: {name}")

    nn.quantize(
        module,
        bits=4,
        group_size=GROUP_SIZE,
        mode="affine",
        class_predicate=lambda name, _: name in packed,
    )
    expected = dict(tree_flatten(module.parameters()))
    if index.keys() != expected.keys():
        raise ValueError(
            f"published Q4 {component} tensors differ from the native layout "
            f"(missing: {sorted(expected.keys() - index.keys())[:4]}, "
            f"unexpected: {sorted(index.keys() - expected.keys())[:4]})"
        )

    for name, tensor in expected.items():
        dtype = "U32" if tensor.dtype == mx.uint32 else "BF16"
        if index[name] != [dtype, list(tensor.shape)]:
            raise ValueError(f"published Q4 {component} tensor has incompatible layout: {name}")

    def tensors():
        for path, mapping in mappings.items():
            weights = mx.load(path)
            for source, target in mapping.items():
                yield target, weights[source]

    return tensors()
