# Author: Susant Achary
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Metal packing for Inkling's multirow controller input."""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source

from .verify import same_bits


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_controller_pack",
        input_names=["x", "message", "eps"],
        output_names=["packed"],
        ensure_row_contiguous=False,
        source=metal_source("inkling_controller_pack.metal"),
    )


def _launch(x, message, eps):
    _, rows, width = x.shape
    threads = width // 4
    return _kernel()(
        inputs=[x, message, mx.array([eps], dtype=mx.float32)],
        template=[("WIDTH", width)],
        grid=(threads, rows, 2),
        threadgroup=(threads, 1, 1),
        output_shapes=[(1, rows, 2 * width)],
        output_dtypes=[mx.bfloat16],
    )[0]


def pack(x, message, eps, dtype, reference, proofs):
    """Qualify the actual operands on first use of each prefill shape."""
    if (x.ndim != 3 or x.shape != message.shape or x.shape[0] != 1
            or not 5 <= x.shape[1] <= 128 or x.shape[2] != 2304
            or x.dtype != mx.float32 or message.dtype != mx.float32
            or dtype != mx.bfloat16 or mx.default_device() != mx.gpu):
        return reference()

    signature = (x.shape, eps)
    passed = proofs.get(signature)
    if passed is False:
        return reference()
    if passed is True:
        return _launch(x, message, eps)

    expected = reference()
    try:
        actual = _launch(x, message, eps)
        proofs[signature] = bool(same_bits(actual, expected).item())
    except (RuntimeError, ValueError):
        proofs[signature] = False
    return actual if proofs[signature] else expected
