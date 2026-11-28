# Copyright 2026 Youssof Altoukhi. SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, models/inkling/fast/round_program.py.
# See MTPLX_LICENSE.txt and MTPLX_NOTICE.txt.
"""Read a global drafter history bank and its proposed rows in separate Metal bindings."""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source

_ATTEND = metal_source("inkling_draft_partials.metal")

_MERGE = metal_source("inkling_draft_merge.metal")
@lru_cache(maxsize=1)
def _attend_kernels():
    partials = mx.fast.metal_kernel(
        name="mlx_one_replay_draft_attend",
        input_names=["q", "hk", "hv", "ok", "ov", "length"],
        output_names=["pm", "pl", "pacc"],
        source=_ATTEND,
    )
    merge = mx.fast.metal_kernel(
        name="mlx_one_replay_draft_merge",
        input_names=["pm", "pl", "pacc"],
        output_names=["out"],
        source=_MERGE,
    )
    return partials, merge



def draft_attention(q, keys, values, own_k, own_v, position):
    _, heads, rows, dim = q.shape
    groups, capacity = keys.shape[1:3]
    ratio = heads // groups
    if (rows, dim, ratio) != (8, 128, 3) or q.dtype != mx.bfloat16:
        raise ValueError("unsupported replay draft attention geometry")
    chunk = max(128, 1 << max(0, (capacity - 1).bit_length() - 8))
    chunks = (capacity + chunk - 1) // chunk + 1
    template = [("T", q.dtype), ("G", ratio), ("R", rows), ("D", dim),
                ("S", capacity), ("CHUNK", chunk), ("TK", 32), ("QPS", ratio * rows // 8)]
    partials, merge = _attend_kernels()
    chains = groups * chunks * ratio * rows
    state = partials(
        inputs=[q, keys, values, own_k, own_v, position], template=template,
        grid=(256 * chunks, groups, 1), threadgroup=(256, 1, 1),
        output_shapes=[(chains,), (chains,), (chains * dim,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merge(
        inputs=state, template=template, grid=(ratio * rows * dim, groups, 1),
        threadgroup=(dim, 1, 1), output_shapes=[q.shape], output_dtypes=[q.dtype],
    )[0]
