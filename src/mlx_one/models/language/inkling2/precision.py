# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
"""Stored per-matrix affine weights with activations in their trained format."""

import mlx.core as mx
import mlx.nn as nn


class AffineWeight(nn.Module):
    def __init__(self, codes, scales, biases, bits, group_size):
        super().__init__()
        self.codes, self.scales, self.biases = codes, scales, biases
        self.bits, self.group_size = bits, group_size

    def product(self, x):
        return mx.quantized_matmul(
            x,
            self.codes,
            self.scales,
            self.biases,
            bits=self.bits,
            group_size=self.group_size,
        )

    def rows(self, start, end):
        return mx.dequantize(
            self.codes[start:end],
            self.scales[start:end],
            self.biases[start:end],
            bits=self.bits,
            group_size=self.group_size,
        )


class LogicalShape:
    """The shape and dtype of a packed matrix, for code that sizes buffers or casts from them.

    It holds no data: indexing it or multiplying by it fails, so a caller that needs the values
    has to read the packed weight it describes."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(int(n) for n in shape), dtype

    @property
    def ndim(self):
        return len(self.shape)


def packed_logical_shape(weight):
    """(rows, columns) and dtype of an ``AffineWeight`` as its dequantized matrix."""
    rows, words = weight.codes.shape[-2:]
    return LogicalShape((rows, words * 32 // weight.bits), weight.scales.dtype)
