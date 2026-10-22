# Copyright © 2023-2025 Apple Inc.
# SPDX-License-Identifier: MIT
"""Selected-expert MLX dispatch adapted from mlx-lm switch_layers.py.

See models/UPSTREAM_LICENSE. Both floating and packed experts execute only
the selected matrices, without materializing all experts or CPU token routing.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class ExpertLinear(nn.Module):
    def __init__(self, inputs, outputs, experts, *, bias=False):
        super().__init__()
        bound = math.sqrt(1 / inputs)
        self.weight = mx.random.uniform(-bound, bound, (experts, outputs, inputs))
        if bias:
            self.bias = mx.zeros((experts, outputs))

    def __call__(self, x, indices):
        y = mx.gather_mm(x, self.weight.swapaxes(-1, -2), rhs_indices=indices)
        if "bias" in self:
            y = y + self.bias[indices][..., None, :]
        return y

    def to_quantized(self, group_size=64, bits=4, mode="affine"):
        return QuantizedExpertLinear(self, group_size, bits, mode)


class QuantizedExpertLinear(nn.Module):
    def __init__(self, source, group_size, bits, mode):
        super().__init__()
        self.group_size, self.bits, self.mode = group_size, bits, mode
        self.weight, self.scales, *biases = mx.quantize(
            source.weight, group_size=group_size, bits=bits, mode=mode
        )
        if biases:
            self.biases = biases[0]
        if "bias" in source:
            self.bias = source.bias
        self.freeze()

    def __call__(self, x, indices):
        y = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.get("biases"),
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            y = y + self.bias[indices][..., None, :]
        return y


class SelectedExperts(nn.Module):
    def __init__(self, inputs, hidden, experts, *, bias=False, swiglu_limit=None):
        super().__init__()
        self.gate_proj = ExpertLinear(inputs, hidden, experts, bias=bias)
        self.up_proj = ExpertLinear(inputs, hidden, experts, bias=bias)
        self.down_proj = ExpertLinear(hidden, inputs, experts, bias=bias)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        gate, up = self.gate_proj(x, indices), self.up_proj(x, indices)
        if self.swiglu_limit is None:
            activation = nn.silu(gate) * up
        else:
            gate = mx.minimum(gate, self.swiglu_limit)
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            activation = (up + 1) * (gate * mx.sigmoid(1.702 * gate))
        return self.down_proj(activation, indices).squeeze(-2)
