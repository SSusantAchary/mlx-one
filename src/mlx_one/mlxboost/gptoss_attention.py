# Copyright © 2024 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from MLX v0.32.2 sdpa_vector.h, commit
# 1f8e74e3f12f31365464a6867c6579f0e9b29d85; see models/UPSTREAM_LICENSE.
"""Experimental sink-aware GQA decode with stock-MLX fallback at the caller.

The split pass reuses each K/V read across eight query heads. Exactly one
subchunk includes the per-head sink in its softmax denominator. No key/value
is associated with the sink. Only full-history B1, single-query D64 is eligible.
"""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source


@lru_cache(maxsize=1)
def _split_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_gqa_sinks_split",
        input_names=["queries", "keys", "values", "sinks"],
        output_names=["partials", "sums", "maxs"],
        ensure_row_contiguous=False,
        source=metal_source("gptoss_attention_split.metal"),
    )


@lru_cache(maxsize=1)
def _reduce_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_gqa_sinks_reduce",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=metal_source("gptoss_attention_reduce.metal"),
    )


def attention(queries, keys, values, sinks, *, blocks=64):
    """Full cached causal attention for a last-position query; None if unsupported."""
    if (
        mx.default_device() != mx.gpu
        or queries.shape != (1, 64, 1, 64)
        or keys.ndim != 4
        or keys.shape[:2] != (1, 8)
        or keys.shape[-1] != 64
        or keys.shape[2] < 1
        or values.shape != keys.shape
        or sinks.shape != (64,)
        or queries.dtype not in {mx.float32, mx.float16, mx.bfloat16}
        or any(value.dtype != queries.dtype for value in (keys, values, sinks))
        or blocks not in {32, 64, 128, 256}
    ):
        return None
    partials, sums, maxs = _split_kernel()(
        inputs=[queries, keys, values, sinks],
        template=[("T", queries.dtype), ("BLOCKS", blocks)],
        grid=(8 * 32, 8, blocks),
        threadgroup=(32, 8, 1),
        output_shapes=[(64, blocks, 64), (64, blocks), (64, blocks)],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )
    return _reduce_kernel()(
        inputs=[partials, sums, maxs],
        template=[("T", queries.dtype), ("BLOCKS", blocks)],
        grid=(64 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
