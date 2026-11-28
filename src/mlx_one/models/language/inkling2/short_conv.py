# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
# Source: mtplx/models/inkling/short_conv.py. See MTPLX_NOTICE.txt.
"""Opt-in Inkling convolution and residual fusion with committed-tail ownership."""

import math
import struct
from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source


def reference(x, prev, weight, change, add, out_dtype):
    size = weight.shape[0]
    xf = x.astype(mx.float32)
    if prev is None:
        prev = mx.zeros((x.shape[0], size - 1, x.shape[-1]), mx.float32)
    history = mx.concatenate([prev, xf], axis=1)
    tail = (
        mx.take(xf, mx.arange(x.shape[1] - size + 1, x.shape[1]), axis=1)
        if x.shape[1] > size else history[:, -(size - 1):]
    )
    if change is None:
        weighted = mx.conv1d(history, weight.T[:, :, None], groups=x.shape[-1])
        divisor = math.sqrt(size + 1)
    else:
        heads = change.shape[-1] // size
        change = change.astype(mx.float32).reshape(*x.shape[:2], heads, size)
        weighted = mx.zeros((*x.shape[:2], heads, x.shape[-1] // heads), mx.float32)
        for lag in range(size):
            start = size - 1 - lag
            current = history[:, start:start + x.shape[1]].reshape(weighted.shape)
            coefficients = weight[lag].reshape(heads, -1) + change[..., lag, None]
            weighted = weighted + current * coefficients
        weighted = weighted.reshape(x.shape)
        divisor = math.sqrt(2 * size + 1)
    out = ((xf + weighted) / divisor).astype(x.dtype)
    if out_dtype is not None:
        out = out.astype(mx.float32)
        if add is not None:
            out = out + add.astype(mx.float32)
        out = out.astype(out_dtype)
    return out, tail


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_short_conv_b1",
        input_names=["x", "prev", "weight", "change", "add"],
        output_names=["out", "tail"],
        ensure_row_contiguous=False,
        source=metal_source("inkling_conv_short.metal"),
    )


_DIVISORS = {
    dynamic: struct.unpack("<I", struct.pack("<f", math.sqrt(9 if dynamic else 5)))[0]
    for dynamic in (False, True)
}


def _launch(x, prev, weight, change, add, out_dtype):
    width = x.shape[-1]
    return _kernel()(
        inputs=[x, prev, weight, x if change is None else change, x if add is None else add],
        template=[("T", x.dtype), ("U", out_dtype or x.dtype), ("DYNAMIC", change is not None),
                  ("DIVISOR", _DIVISORS[change is not None]), ("ADD", add is not None)],
        grid=(width, x.shape[1], 1), threadgroup=(256, 1, 1),
        output_shapes=[x.shape, prev.shape],
        output_dtypes=[out_dtype or x.dtype, mx.float32],
    )


@lru_cache(maxsize=12)
def _matches_reference(width, change_dtype, add_dtype, out_dtype):
    # Keep separate proofs for each residual/scale layout and selected GPU dtype.
    keys = mx.random.split(mx.random.key(20261122 + width), 5)
    x = mx.random.normal((1, 1, width), key=keys[0]).astype(mx.bfloat16)
    prev = mx.random.normal((1, 3, width), key=keys[1])
    weight = mx.random.normal((width, 4), key=keys[2]).T
    change = (None if change_dtype is None else
              mx.random.normal((1, 1, 32), key=keys[3]).astype(change_dtype))
    add = None if add_dtype is None else mx.random.normal(x.shape, key=keys[4]).astype(add_dtype)
    try:
        expected = reference(x, prev, weight, change, add, out_dtype)
        actual = _launch(x, prev, weight, change, add, out_dtype)
        bits = (mx.uint32 if actual[0].dtype == mx.float32 else mx.uint16, mx.uint32)
        equal = [mx.array_equal(a.view(t), b.view(t))
                 for a, b, t in zip(actual, expected, bits, strict=True)]
        return bool(mx.all(mx.stack(equal)).item())
    except (RuntimeError, ValueError):
        return False


def _supported(x, prev, weight, change, add, out_dtype):
    batch, rows, width = x.shape
    return (batch == 1 and width in (1024, 2304) and x.dtype == mx.bfloat16
            and (prev is None or (prev.shape == (1, 3, width) and prev.dtype == mx.float32))
            and weight.shape == (4, width) and weight.dtype == mx.float32
            and (out_dtype is None or out_dtype in (mx.bfloat16, mx.float32))
            and (change is None or (change.shape == (1, rows, 32)
                                    and change.dtype == mx.bfloat16))
            and (add is None or (out_dtype is not None and add.shape == x.shape
                                 and add.dtype in (mx.bfloat16, mx.float32)))
            and mx.default_device() == mx.gpu and mx.metal.is_available())


def fused(x, prev, weight, change, add, out_dtype):
    """Return the optimized output/tail, or None to use the unmodified MLX operations."""
    if (prev is None or x.ndim != 3 or x.shape[1] != 1
            or not _supported(x, prev, weight, change, add, out_dtype)):
        return None
    if not _matches_reference(x.shape[-1], getattr(change, "dtype", None),
                              getattr(add, "dtype", None), out_dtype):
        return None
    return _launch(x, prev, weight, change, add, out_dtype)


def fused_panel(x, prev, weight, change, add, out_dtype):
    """Use the one-row tap order for each row of a bounded verification panel."""
    if (prev is None or x.ndim != 3 or not 2 <= x.shape[1] <= 4
            or not _supported(x, prev, weight, change, add, out_dtype)):
        return None
    return _launch(x, prev, weight, change, add, out_dtype)


def fused_prefill(x, prev, weight, change, add, out_dtype, *, proofs):
    """Compare each prefill form with the native multirow convolution on first use."""
    if (x.ndim != 3 or not 2 <= x.shape[1] <= 128
            or not _supported(x, prev, weight, change, add, out_dtype)):
        return None
    from .verify import same_bits

    signature = (x.shape, getattr(change, "dtype", None), getattr(add, "dtype", None),
                 out_dtype, prev is None)
    passed = proofs.get(signature)
    if passed is False:
        return None
    tail = prev if prev is not None else mx.zeros((1, 3, x.shape[-1]), mx.float32)
    if passed:
        return _launch(x, tail, weight, change, add, out_dtype)
    expected = reference(x, prev, weight, change, add, out_dtype)
    try:
        result = _launch(x, tail, weight, change, add, out_dtype)
        matches = [same_bits(a, b) for a, b in zip(result, expected, strict=True)]
        proofs[signature] = bool(mx.all(mx.stack(matches)).item())
        return result if proofs[signature] else expected
    except (RuntimeError, ValueError):
        proofs[signature] = False
        return expected


def verify_panel(x, prev, weight, change, add, out_dtype, *, fused_decode):
    from .verify import qualified

    signature = ("conv", x.shape, x.dtype, getattr(change, "dtype", None),
                 getattr(add, "dtype", None), out_dtype, prev is None, fused_decode)

    def chronological():
        tail = prev
        parts = []
        for row in range(x.shape[1]):
            sl = slice(row, row + 1)
            args = (
                x[:, sl], tail, weight,
                None if change is None else change[:, sl],
                None if add is None else add[:, sl], out_dtype,
            )
            result = fused(*args) if fused_decode else None
            part, tail = result if result is not None else reference(*args)
            parts.append(part)
        return mx.concatenate(parts, axis=1), tail

    return qualified(signature, lambda: fused_panel(x, prev, weight, change, add, out_dtype),
                     chronological)
