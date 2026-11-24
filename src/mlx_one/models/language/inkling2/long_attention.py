# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, models/inkling/fast/prefill_attend.py and attend.py.
# See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
"""Exact M4 vector attention over the native cache's full-history BF16 banks."""

import os
import struct
from functools import lru_cache
from importlib.metadata import version

import mlx.core as mx

from . import _long_attention_metal as metal
from .attention import attend, chronological_attention, query_tile_rows

_SCALE = struct.unpack("<I", struct.pack("<f", 128**-0.5))[0]
_LAUNCH_ROWS = 16
# On a failed first-use check, keep the native reference for this reduction plan.
_qualified: dict[tuple[bool, tuple[int, ...], int], bool] = {}


@lru_cache(maxsize=1)
def _supported_gpu():
    return (version("mlx") == "0.32.2" and mx.metal.is_available()
            and mx.device_info().get("architecture") == "applegpu_g16g")


@lru_cache(maxsize=1)
def _kernels():
    partials = mx.fast.metal_kernel(
        name="mlx_one_inkling_long_partials",
        input_names=["q", "kc", "vc", "prof", "sink", "geometry"],
        output_names=["po", "pm", "ps"], source=metal.PARTIALS, header=metal.HEADER,
        ensure_row_contiguous=False,
    )
    merge = mx.fast.metal_kernel(
        name="mlx_one_inkling_long_merge", input_names=["po", "pm", "ps"],
        output_names=["hg"], source=metal.MERGE, ensure_row_contiguous=False,
    )
    return partials, merge


def _plan(rows, keys):
    # On the base M4, MLX 0.32.2 uses 32 blocks for three simdgroups
    # (one query row) and 64 for six or more. Preserve the stock tail tile.
    step = query_tile_rows(1, 24, keys + 1)
    if step > 8 or (1 < rows <= min(8, query_tile_rows(1, 24, keys))):
        return ()
    if rows == 1 or step == 1:
        return ((0, rows, 32),)
    if rows % step == 1:
        return ((0, rows - 1, 64), (rows - 1, rows, 32))
    return ((0, rows, 64),)


def _launch(query, keys, values, profiles, sink, count, plan, sink_first):
    partials, merge = _kernels()
    rows = query.shape[2]
    q = mx.contiguous(query.transpose(0, 2, 1, 3))
    profiles = mx.contiguous(profiles)
    # These are the allocated banks (normally already contiguous), never sliced
    # to the live prefix. Each launch reads only through its causal position.
    keys, values = mx.contiguous(keys), mx.contiguous(values)
    parts = []
    for start, end, blocks in plan:
        for first in range(start, end, _LAUNCH_ROWS):
            last = min(first + _LAUNCH_ROWS, end)
            n = last - first
            chains = n * 24 * blocks
            geometry = mx.array([count - rows + first, count], mx.int32)
            common = [("T", mx.bfloat16), ("BLOCKS", blocks)]
            state = partials(
                inputs=[q[:, first:last], keys, values, profiles[:, first:last],
                        sink[:, first:last], geometry],
                template=[*common, ("ROWS", 8), ("SCALE", _SCALE), ("SINK_FIRST", sink_first)],
                grid=(256 * blocks, 8, (n + 7) // 8), threadgroup=(256, 1, 1),
                output_shapes=[(chains * 128,), (chains,), (chains,)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
            result = merge(
                inputs=state, template=common,
                grid=(1024 * 24, n, 1), threadgroup=(1024, 1, 1),
                output_shapes=[(1, n, 24, 128)], output_dtypes=[mx.bfloat16],
            )[0]
            parts.append(result)
            if rows > _LAUNCH_ROWS:
                mx.async_eval(result)
                if len(parts) > 4:
                    mx.eval(parts[-5])
    out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
    return out.transpose(0, 2, 1, 3)


def try_global_attention(query, keys, values, profiles, sink, count, *, verify=False):
    """Return None to use the established attention, including on failed bit checks."""
    batch, heads, rows, width = query.shape
    if ((batch, heads, width) != (1, 24, 128) or not 1 <= rows <= 128
            or not rows <= count <= keys.shape[2] or values.shape != keys.shape
            or keys.shape != (1, 8, keys.shape[2], 128)
            or profiles.shape != (1, rows, 24, 1024) or sink.shape != (1, rows, 24)
            or any(x.dtype != mx.bfloat16 for x in (query, keys, values, profiles, sink))):
        return None
    plan = (((0, rows, 32),) if count - rows + 1 >= 4096 else ()) if verify else _plan(rows, count)
    if (not plan or mx.default_device() != mx.gpu or not _supported_gpu()
            or os.environ.get("MLX_SDPA_BLOCKS")):
        return None
    sink_first = verify or rows == 1
    signature = (sink_first, tuple(blocks for _, _, blocks in plan), rows if verify else 0)
    proven = _qualified.get(signature)
    if proven is False:
        return None
    try:
        result = _launch(query, keys, values, profiles, sink, count, plan, sink_first)
        if proven is None:
            reference_fn = chronological_attention if verify else attend
            reference = reference_fn(query, keys[:, :, :count], values[:, :, :count],
                                     profiles, sink, 0)
            equal = mx.array_equal(result.view(mx.uint16), reference.view(mx.uint16)).item()
            _qualified[signature] = bool(equal)
            if not equal:
                return reference
        return result
    except (RuntimeError, ValueError):
        _qualified[signature] = False
        return None
