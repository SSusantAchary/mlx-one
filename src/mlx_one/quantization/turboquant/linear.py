"""Packed projections: compressed parameters remain compressed for every call."""

import math

import mlx.core as mx
import mlx.nn as nn

from .dispatch import dense_kernel, record
from .kernels.tq3_qmm import tq3_qmm
from .kernels.tq3_qmv import tq3_qmv


def hadamard_block_size(dim):
    best = 1
    for m in (1, 12, 20, 28):
        n = m
        while n <= dim:
            if dim % n == 0:
                best = max(best, n)
            n *= 2
    return best


def rotate_input(x, signs):
    # Stored signs are authoritative. Do not regenerate from rotation_seed.
    dim = x.shape[-1]
    block = hadamard_block_size(dim)
    value = x * signs
    shape = value.shape
    return mx.hadamard_transform(
        value.reshape(*shape[:-1], dim // block, block), scale=1 / math.sqrt(block)
    ).reshape(shape)


class TQ3Linear(nn.Module):
    format = "turboquant_tq3"

    def __init__(self, inputs, outputs, tensors):
        super().__init__()
        self.input_dims, self.output_dims = inputs, outputs
        # Arrays are installed by reference, never repacked or expanded.
        for name, value in tensors.items():
            setattr(self, name, value)
        self.freeze()

    def __call__(self, x):
        x = rotate_input(x, self.signs)
        if x.dtype == mx.bfloat16:
            x = x.astype(mx.float32)
        shape = x.shape
        rows = math.prod(shape[:-1])
        kernel = dense_kernel(rows, self.output_dims)
        record(kernel)
        flat = x.reshape(rows, self.input_dims)
        if kernel == "qmv":
            y = tq3_qmv(self.weight, self.scales, self.codebook, flat, 3, 64)
        elif kernel == "qmv_rows":
            y = mx.stack(
                [
                    tq3_qmv(self.weight, self.scales, self.codebook, flat[i], 3, 64)
                    for i in range(rows)
                ]
            )
        else:
            y = tq3_qmm(self.weight, self.scales, self.codebook, flat, 3, 64)
        y = y.reshape(*shape[:-1], self.output_dims)
        return y + self.bias if "bias" in self else y
