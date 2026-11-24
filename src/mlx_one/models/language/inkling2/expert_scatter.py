# SPDX-License-Identifier: Apache-2.0
"""Restore Inkling's expert slot order in one Metal dispatch.

Inspired by OpenAI's gpt-oss Metal scatter, also distributed in the Hugging Face
kernel catalog at revision 9e86c1c0b9935387e73b452b6d97f8c3e9251ee0.
This version reverses a one-to-one permutation of BF16 expert outputs.
"""

from functools import lru_cache

import mlx.core as mx

_qualified = {}


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_expert_output_scatter",
        input_names=["expert_rows", "order"],
        output_names=["token_rows"],
        source="""
            const uint column = thread_position_in_grid.x;
            const uint slot = thread_position_in_grid.y;
            const uint width = expert_rows_shape[1] / 8;
            if (column >= width || slot >= expert_rows_shape[0]) return;
            const device uint4* input = reinterpret_cast<const device uint4*>(expert_rows);
            device uint4* output = reinterpret_cast<device uint4*>(token_rows);
            output[size_t(order[slot]) * width + column] = input[size_t(slot) * width + column];
        """,
    )


def restore(rows, order):
    """Scatter each packed row to its original slot. Order must come from argsort."""
    if (rows.ndim != 2 or rows.dtype != mx.bfloat16 or not rows.size or rows.shape[1] % 8
            or order.shape != (rows.shape[0],) or order.dtype != mx.uint32
            or mx.default_device() != mx.gpu or not mx.metal.is_available()):
        return None
    passed = _qualified.get(rows.shape)
    if passed is False:
        return None
    try:
        result = _launch(rows, order)
        if passed is None:
            reference = rows[mx.argsort(order)]
            _qualified[rows.shape] = bool(
                mx.array_equal(result.view(mx.uint16), reference.view(mx.uint16)).item()
            )
            return result if _qualified[rows.shape] else reference
        return result
    except (RuntimeError, ValueError):
        _qualified[rows.shape] = False
        return None


def _launch(rows, order):
    return _kernel()(
        inputs=[rows, order],
        grid=(rows.shape[1] // 8, rows.shape[0], 1),
        threadgroup=(128, 1, 1),
        output_shapes=[rows.shape],
        output_dtypes=[rows.dtype],
    )[0]
