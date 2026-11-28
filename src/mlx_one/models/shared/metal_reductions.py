# Author: Susant Achary
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Metal reductions with model-specific numerical contracts."""

import os
from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source


def _enabled():
    return (os.environ.get("MLX_ONE_METAL_REDUCTIONS") == "1"
            and mx.default_device() == mx.gpu and mx.metal.is_available())


@lru_cache(maxsize=1)
def _magnitude_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_split_spectrum_magnitude",
        input_names=["spectrum"], output_names=["output"],
        source=metal_source("split_spectrum_magnitude.metal"),
    )


def try_split_spectrum_magnitude(spectrum):
    if (not _enabled() or spectrum.ndim != 3 or not spectrum.size
            or spectrum.shape[2] % 2 or spectrum.dtype != mx.float32):
        return None
    return _magnitude_kernel()(
        inputs=[spectrum], grid=(spectrum.size // 2, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(*spectrum.shape[:2], spectrum.shape[2] // 2)],
        output_dtypes=[mx.float32],
    )[0]


@lru_cache(maxsize=1)
def _t5_norm_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_t5_rms_norm",
        input_names=["hidden", "weight", "eps"], output_names=["output"],
        source=metal_source("t5_rms_norm.metal"),
    )


def try_t5_norm(x, weight, eps):
    if (not _enabled() or x.ndim != 3 or not x.size or x.dtype != mx.float32
            or weight.dtype != x.dtype or weight.shape != (x.shape[-1],)
            or x.shape[-1] > 4096):
        return None
    rows = x.shape[0] * x.shape[1]
    return _t5_norm_kernel()(
        inputs=[x, weight, mx.array([eps], dtype=mx.float32)],
        grid=(32, rows, 1), threadgroup=(32, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]
