# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
"""Inkling relative-position bias, sliding masks, and query-dependent sinks."""

from functools import lru_cache

import mlx.core as mx

from .attention_metal import grouped_sink_attention


@lru_cache(maxsize=1)
def _bias_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_relative_bias",
        input_names=["profiles", "sink", "geometry"],
        output_names=["bias"],
        source="""
            const uint i = thread_position_in_grid.x;
            const int keys = geometry[0], first = geometry[1], window = geometry[2];
            const int heads = profiles_shape[2], rows = profiles_shape[1];
            const int extent = profiles_shape[3];
            const int col = i % (keys + 1);
            uint plane = i / (keys + 1);
            const int row = plane % rows;
            plane /= rows;
            const int head = plane % heads, batch = plane / heads;
            const uint p = (batch * rows + row) * heads + head;
            const int distance = first + row - col;
            if (col == keys) bias[i] = sink[p];
            else if (distance < 0 || (window && distance >= window)) bias[i] = T(-INFINITY);
            else if (distance >= extent) bias[i] = T(0);
            else bias[i] = profiles[p * extent + distance];
        """,
    )


def relative_bias(profiles, sink, keys, first, window, *, metal=True):
    batch, rows, heads, extent = profiles.shape
    shape = (batch, heads, rows, keys + 1)
    if metal and mx.default_device() == mx.gpu and mx.metal.is_available():
        return _bias_kernel()(
            inputs=[profiles, sink, mx.array([keys, first, window], mx.int32)],
            template=[("T", profiles.dtype)],
            grid=(batch * heads * rows * (keys + 1), 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[shape],
            output_dtypes=[profiles.dtype],
        )[0]
    distance = first + mx.arange(rows)[:, None] - mx.arange(keys)[None, :]
    values = mx.take_along_axis(
        profiles.transpose(0, 2, 1, 3),
        mx.clip(distance, 0, extent - 1)[None, None],
        axis=-1,
    )
    values = mx.where(distance >= extent, mx.zeros((), profiles.dtype), values)
    valid = distance >= 0 if not window else (distance >= 0) & (distance < window)
    return mx.concatenate(
        [
            mx.where(valid, values, mx.array(-float("inf"), profiles.dtype)),
            sink.transpose(0, 2, 1)[..., None],
        ],
        axis=-1,
    )


def query_tile_rows(batch, heads, key_columns):
    return max(1, (4 * 1024 * 1024) // (batch * heads * key_columns))


def stock_attention(query, key, value, profiles, sink, window, *, overlap_tiles=False):
    """Tile the query axis so a long prompt never builds its full quadratic bias tensor."""
    batch, heads, rows, width = query.shape
    keys = key.shape[2]
    zero = mx.zeros((batch, key.shape[1], 1, width), key.dtype)
    key = mx.concatenate([key, zero], axis=2)
    value = mx.concatenate([value, zero], axis=2)
    step = query_tile_rows(batch, heads, keys + 1)
    parts = []
    for start in range(0, rows, step):
        end = min(rows, start + step)
        bias = relative_bias(
            profiles[:, start:end],
            sink[:, start:end],
            keys,
            keys - rows + start,
            window,
        )
        part = mx.fast.scaled_dot_product_attention(
            query[:, :, start:end], key, value, scale=width**-0.5, mask=bias
        )
        if rows > step:
            if overlap_tiles:
                # Queue the next tile before waiting for the oldest pending tile.
                mx.async_eval(part)
                if len(parts) >= 2:
                    mx.eval(parts[-2])
            else:
                mx.eval(part)
        parts.append(part)
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def attend(query, key, value, profiles, sink, window, *, overlap_tiles=False):
    if query.shape[0] == 1 and mx.default_device() == mx.gpu and mx.metal.is_available():
        # Bound mask workspace to the stock tile size without copying the KV history.
        count = key.shape[2]
        tiles = min(8, query_tile_rows(1, query.shape[1], count))
        if query.shape[2] <= tiles:
            return grouped_sink_attention(query, key, value, profiles, sink, window)
    return stock_attention(query, key, value, profiles, sink, window, overlap_tiles=overlap_tiles)


def chronological_attention(query, key, value, profiles, sink, window):
    rows, count = query.shape[2], key.shape[2]
    parts = []
    for row in range(rows):
        end = count - rows + row + 1
        start = max(0, end - window) if window else 0
        parts.append(attend(query[:, :, row:row + 1], key[:, :, start:end],
                            value[:, :, start:end], profiles[:, row:row + 1],
                            sink[:, row:row + 1], window))
    return mx.concatenate(parts, axis=2)
