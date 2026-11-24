# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
"""Inkling's chronological attention over grouped KV, with one sink per query."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _grouped_mask_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_grouped_sink_mask",
        input_names=["profiles", "keys"],
        output_names=["mask"],
        ensure_row_contiguous=False,
        source="""
            const int col = thread_position_in_grid.x, row = thread_position_in_grid.y;
            const int head = thread_position_in_grid.z, rows = profiles_shape[1];
            const int count = keys_shape[2], extent = profiles_shape[3];
            if (col >= count) return;
            const int distance = count - rows + row - col;
            const bool visible = distance >= 0 && (!WINDOW || distance < WINDOW);
            T bias = T(0);
            if (!visible) bias = T(-INFINITY);
            else if (distance < extent) {
                const size_t offset = row * profiles_strides[1] + head * profiles_strides[2];
                bias = profiles[offset + distance * profiles_strides[3]];
            }
            mask[(head * rows + row) * size_t(count) + col] = bias;
        """,
    )


def grouped_sink_attention(query, key, value, profiles, sink, window):
    """One native SDPA sink per (head,row), sharing the original KV heads."""
    _, heads, rows, width = query.shape
    if window and key.shape[2] > window + rows - 1:
        start = key.shape[2] - (window + rows - 1)
        key, value = key[:, :, start:], value[:, :, start:]
    count = key.shape[2]
    shape = (1, heads * rows, 1, count)
    mask = _grouped_mask_kernel()(
        inputs=[profiles, key],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(count, rows, heads),
        threadgroup=(min(128, count), 1, 1),
        output_shapes=[shape],
        output_dtypes=[query.dtype],
    )[0]
    # Flatten head first so integer division by the expanded GQA group still
    # maps every query row to the original KV head. Each sink stays independent.
    result = mx.fast.scaled_dot_product_attention(
        query.reshape(1, heads * rows, 1, width),
        key, value, scale=width**-0.5, mask=mask, sinks=sink[0].T.reshape(-1),
    )
    return result.reshape(query.shape)
