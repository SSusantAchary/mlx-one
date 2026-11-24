# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, Inkling text weight namespaces and affine layout.
"""Read the installed Bare Speed pack without materializing or re-encoding experts."""

import json


def sanitize_weights(weights, config):
    if any(not k.startswith(("model.llm.", "model.visual.")) for k in weights):
        raise ValueError("unknown Inkling checkpoint tensor namespace")
    return {k.removeprefix("model.llm."): v for k, v in weights.items()
            if k.startswith("model.llm.")}


def load_parameters(root, config, tensors):
    from .model import InklingForCausalLM
    from .precision import AffineWeight

    for full_name, spec in config.precision.items():
        if not full_name.startswith("model.llm."):
            raise ValueError("unsupported Inkling packed matrix namespace")
        name = full_name.removeprefix("model.llm.")
        codes, scales, biases = (tensors.pop(name + suffix) for suffix in
                                  ("", ".mtplx_scales", ".mtplx_biases"))
        bits, group = spec["bits"], spec["group_size"]
        rows, columns = codes.shape
        if (bits != 4 or group != 64 or str(codes.dtype) != "mlx.core.uint32"
                or scales.shape != (rows, columns * 8 // group)
                or biases.shape != scales.shape or scales.dtype != biases.dtype):
            raise ValueError(f"invalid Inkling affine weight layout at {full_name}")
        tensors[name] = AffineWeight(codes, scales, biases, bits, group)
    quant = json.loads((root / "hf_quant_config.json").read_text())
    modes = {name: group.get("compute_dtype")
             for group in quant["model"]["parameter_groups"] for name in group["tensors"]}
    count = config.text["num_hidden_layers"]
    for i in range(count):
        for name in ("w13_weight", "w2_weight"):
            if modes.get(f"model.llm.layers.{i}.mlp.experts.{name}") != "nvfp4":
                raise ValueError("Inkling pack does not declare NVFP4 experts")
    model = InklingForCausalLM(config, tensors, ["nvfp4"] * count)
    sidecar = root / "mtp.safetensors"
    if sidecar.is_file():
        if sidecar.resolve().parent != root:
            raise ValueError("Inkling draft sidecar must be a local safetensors file")
        from mlx_one.text.loading import _load_safetensor_file

        from .draft import Drafter

        model.drafter = Drafter(_load_safetensor_file(sidecar), config.draft, model)
        model.drafter.fold(model.unembed)
    return model
