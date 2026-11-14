# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: MIT
# GPU assignment sorting follows MLX-LM SwitchGLU, pinned at
# 56230fa02b8ff8330daf26dc95c28480423ed5fe. See models/UPSTREAM_LICENSE.
"""Opt-in GPT-OSS prefill scheduling with unchanged packed expert weights.

Assignments are grouped by expert on the GPU, then restored to their original
token/expert order before the native weighted reduction. ``grouped=False``
retains ordinary gathered GEMV; ``grouped=True`` requests MLX's grouped kernel.
Neither route changes precision, selected experts, or retained KV history.
"""

import mlx.core as mx


def prefill_moe(experts, x, indices, scores, *, grouped=False):
    """Return the sorted-assignment MoE result, or None for unsupported layouts."""
    if (
        mx.default_device() != mx.gpu
        or x.ndim != 3
        or x.shape[0] != 1
        or not 16 <= x.shape[1] <= 1024
        or x.dtype != mx.bfloat16
        or indices.shape != (*x.shape[:2], 4)
        or indices.dtype not in {mx.uint32, mx.int32}
        or scores.shape != indices.shape
        or scores.dtype != x.dtype
        or experts.swiglu_limit != 7
    ):
        return None
    gate, up, down = experts.gate_proj, experts.up_proj, experts.down_proj
    hidden, intermediate = x.shape[-1], gate.weight.shape[1]
    for module, inputs, outputs in (
        (gate, hidden, intermediate),
        (up, hidden, intermediate),
        (down, intermediate, hidden),
    ):
        if (
            getattr(module, "mode", None) != "mxfp4"
            or module.group_size != 32
            or module.bits != 4
            or module.weight.dtype != mx.uint32
            or module.scales.dtype != mx.uint8
            or module.weight.shape != (gate.weight.shape[0], outputs, inputs // 8)
            or module.scales.shape != (gate.weight.shape[0], outputs, inputs // 32)
            or "bias" not in module
            or module.bias.shape != module.weight.shape[:2]
            or module.bias.dtype != x.dtype
            or inputs % 32
        ):
            return None

    flat_indices = indices.reshape(-1)
    order = mx.argsort(flat_indices)
    inverse = mx.argsort(order)
    selected = flat_indices[order]
    inputs = x.reshape(-1, hidden)[order // 4, None, :]

    def project(module, values):
        output = mx.gather_qmm(
            values,
            module.weight,
            module.scales,
            rhs_indices=selected,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=grouped,
        )
        return output + module.bias[selected][:, None, :]

    gate_values = mx.minimum(project(gate, inputs), 7)
    up_values = mx.clip(project(up, inputs), -7, 7)
    activation = (up_values + 1) * (gate_values * mx.sigmoid(1.702 * gate_values))
    values = project(down, activation)[inverse]
    values = values.reshape(*indices.shape, hidden)
    return (values * scores[..., None]).sum(axis=-2)
