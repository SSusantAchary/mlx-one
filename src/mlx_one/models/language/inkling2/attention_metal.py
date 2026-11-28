# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
"""Inkling's chronological attention over grouped KV, with one sink per query."""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source

from ._long_attention_metal import HEADER


@lru_cache(maxsize=1)
def _grouped_mask_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_grouped_sink_mask",
        input_names=["profiles", "keys"],
        output_names=["mask"],
        ensure_row_contiguous=False,
        source=metal_source("inkling_attention_grouped_mask.metal"),
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


@lru_cache(maxsize=1)
def _window_panel_kernel():
    # Follows MTPLX 2.13.0 attend.py's window-panel threadgroup layout and
    # MLX v0.32.2 sdpa_vector's single-pass reduction with an initial sink.
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_window_panel", input_names=["q", "k", "v", "p", "s"],
        output_names=["out"], ensure_row_contiguous=False, header=HEADER,
        source=metal_source("inkling_attention_window_panel.metal"),
    )


def window_panel_attention(query, key, value, profiles, sink, window):
    """Read the existing chronological window directly for each verification row."""
    rows = query.shape[2]
    count = key.shape[2]
    if (not 2 <= rows <= 4 or not rows <= count <= window + rows
            or not 0 < window <= 1024 or query.shape != (1, 24, rows, 128)
            or key.shape != (1, 8, count, 128) or value.shape != key.shape
            or profiles.shape != (1, rows, 24, 1024) or sink.shape != (1, rows, 24)
            or any(a.dtype != mx.bfloat16 for a in (query, key, value, profiles, sink))):
        return None
    return _window_panel_kernel()(
        inputs=[query, key, value, profiles, sink],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(1024 * 24, rows, 1), threadgroup=(1024, 1, 1),
        output_shapes=[query.shape], output_dtypes=[query.dtype],
    )[0]
