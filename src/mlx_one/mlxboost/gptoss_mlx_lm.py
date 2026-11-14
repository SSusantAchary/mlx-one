# Copyright © 2023-2025 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from mlx-lm 56230fa02b8ff8330daf26dc95c28480423ed5fe:
# models/switch_layers.py and models/activations.py. See models/UPSTREAM_LICENSE.
"""GPT-OSS expert execution following the pinned MLX-LM operator sequence.

This deliberately includes MLX-LM's compiled activation and sorted gathered
matrix dispatch. It is separate from the eager-reference-preserving prefill
option; equivalence to that older arithmetic must not be assumed.
"""

from functools import partial

import mlx.core as mx


@partial(mx.compile, shapeless=True)
def swiglu_oai(gate, x, alpha=1.702, limit=7.0):
    gate = mx.minimum(gate, limit)
    x = mx.clip(x, -limit, limit)
    return (x + 1.0) * (gate * mx.sigmoid(alpha * gate))


def _gather_sort(x, indices):
    *_, count = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inverse = mx.argsort(order)
    return x.flatten(0, -3)[order // count], indices[order], inverse


def _project(module, x, indices, sorted_indices):
    # MXFP4 has no affine quantization bias or NVFP4 tensor global scale.
    # MLX 0.32.2 has no gather_qmm global_scale keyword. Omitting the upstream
    # None argument is the only backend-signature adaptation for this profile.
    x = mx.gather_qmm(
        x,
        module.weight,
        module.scales,
        module.get("biases"),
        rhs_indices=indices,
        transpose=True,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
        sorted_indices=sorted_indices,
    )
    return x + mx.expand_dims(module.bias[indices], -2)


def mlx_lm_moe(experts, x, indices, scores):
    """Return the upstream-sequence result, or None outside the scoped layout."""
    if (
        mx.default_device() != mx.gpu
        or x.ndim != 3
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= 2048
        or x.dtype != mx.bfloat16
        or indices.shape != (*x.shape[:2], 4)
        or indices.dtype not in {mx.int32, mx.uint32}
        or scores.shape != indices.shape
        or scores.dtype != x.dtype
        or experts.swiglu_limit != 7
    ):
        return None
    hidden = x.shape[-1]
    count, intermediate = experts.gate_proj.weight.shape[:2]
    for module, inputs, outputs in (
        (experts.gate_proj, hidden, intermediate),
        (experts.up_proj, hidden, intermediate),
        (experts.down_proj, intermediate, hidden),
    ):
        if (
            getattr(module, "mode", None) != "mxfp4"
            or module.group_size != 32
            or module.bits != 4
            or module.weight.dtype != mx.uint32
            or module.scales.dtype != mx.uint8
            or module.weight.shape != (count, outputs, inputs // 8)
            or module.scales.shape != (count, outputs, inputs // 32)
            or module.get("biases") is not None
            or "bias" not in module
            or module.bias.shape != (count, outputs)
            or module.bias.dtype != x.dtype
            or inputs % 32
        ):
            return None

    # Keep expansion, stop-gradient, threshold, projection order, activation,
    # inverse permutation and weighting in the same order as SwitchGLU/MLPBlock.
    x = mx.expand_dims(x, (-2, -3))
    indices = mx.stop_gradient(indices)
    do_sort = indices.size >= 64
    selected = indices
    if do_sort:
        x, selected, inverse = _gather_sort(x, indices)
    up = _project(experts.up_proj, x, selected, do_sort)
    gate = _project(experts.gate_proj, x, selected, do_sort)
    x = _project(experts.down_proj, swiglu_oai(gate, up), selected, do_sort)
    if do_sort:
        x = mx.unflatten(x[inverse], 0, indices.shape)
    x = x.squeeze(-2)
    return (x * mx.expand_dims(scores, axis=-1)).sum(axis=-2)
