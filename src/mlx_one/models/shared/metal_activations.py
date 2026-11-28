# Author: Susant Achary
# SPDX-License-Identifier: Apache-2.0
"""Opt-in MLX-One Metal activation epilogues."""

import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from mlx_one.metal_source import metal_source


def _enabled(option):
    return (os.environ.get(option) == "1" and mx.default_device() == mx.gpu
            and mx.metal.is_available())


@lru_cache(maxsize=1)
def _gelu_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_gelu_f32",
        input_names=["input", "value"],
        output_names=["output"],
        source=metal_source("gelu.metal"),
    )


def try_gelu(x, *, form=0, value=None):
    if (not _enabled("MLX_ONE_METAL_GELU") or x.dtype != mx.float32 or not x.size
            or (value is not None and (value.dtype != x.dtype or value.shape != x.shape))):
        return None
    return _gelu_kernel()(
        inputs=[x, x if value is None else value],
        template=[("FORM", form), ("GATED", value is not None)],
        grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape], output_dtypes=[mx.float32],
    )[0]


def gelu(x):
    candidate = try_gelu(x)
    return nn.gelu(x) if candidate is None else candidate


@lru_cache(maxsize=1)
def _split_glu_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_split_glu_f32",
        input_names=["input", "valid"],
        output_names=["output"],
        source=metal_source("split_glu.metal"),
    )


def split_glu(x, valid=None):
    if (_enabled("MLX_ONE_METAL_SWIGLU") and x.ndim == 3 and x.dtype == mx.float32
            and x.size and x.shape[-1] % 2 == 0
            and (valid is None or (valid.shape == x.shape[:2] and valid.dtype == mx.bool_))):
        return _split_glu_kernel()(
            inputs=[x, x if valid is None else valid],
            template=[("MASKED", valid is not None)],
            grid=(x.size // 2, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(*x.shape[:2], x.shape[2] // 2)], output_dtypes=[mx.float32],
        )[0]
    value, gate = mx.split(x, 2, axis=-1)
    result = value * mx.sigmoid(gate)
    return result if valid is None else mx.where(valid[..., None], result, 0)


@lru_cache(maxsize=1)
def _swiglu_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_swiglu_f32",
        input_names=["gate", "value"],
        output_names=["output"],
        source=metal_source("swiglu.metal"),
    )


def swiglu(gate, value):
    """Fuse FP32 SiLU and multiplication at compatible model call sites."""
    candidate = try_swiglu(gate, value)
    return nn.silu(gate) * value if candidate is None else candidate


def try_swiglu(gate, value):
    if (not _enabled("MLX_ONE_METAL_SWIGLU")
            or gate.shape != value.shape or gate.size == 0
            or gate.dtype != mx.float32 or value.dtype != mx.float32):
        return None
    return _swiglu_kernel()(
        inputs=[gate, value], grid=(gate.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[gate.shape], output_dtypes=[mx.float32],
    )[0]
