"""Model-owned state for checking several DSpark proposals in one forward."""

from contextvars import ContextVar
from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source

from ._verify_affine_metal import HEADER

active = ContextVar("inkling_verify_panel", default=None)


@lru_cache(maxsize=1)
def _affine_kernel():
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_verify_affine",
        input_names=["x", "codes", "scales", "biases"],
        output_names=["out"],
        header=HEADER,
        source=metal_source("inkling_verify_affine_panel.metal"),
    )


def affine_panel(x, weight):
    """Use the stock one-row dispatch's order for every row of a packed product."""
    n, words = weight.codes.shape
    bits = weight.bits
    k = words * 32 // bits
    rows = x.shape[1]
    if bits not in (4, 8):
        return rowwise(weight.product, x)
    if x.dtype == weight.scales.dtype == weight.biases.dtype == mx.bfloat16:
        pack = 32 // bits
        vpt = pack * (2 if n % 8 == 0 and k % (pack * 64) == 0 else 1)
        if n % 4 == 0 and k % (32 * vpt) == 0:
            def launch():
                return _affine_kernel()(
                    inputs=[x, weight.codes, weight.scales, weight.biases],
                    template=[("T", x.dtype), ("W", rows), ("K", k), ("N", n),
                              ("BITS", bits), ("GROUP", weight.group_size), ("VPT", vpt)],
                    grid=(n * 32, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(1, rows, n)], output_dtypes=[x.dtype],
                )[0]

            return qualified(("affine", rows, n, k, bits, weight.group_size), launch,
                             lambda: rowwise(weight.product, x))
    return rowwise(weight.product, x)


def rowwise(product, x):
    return mx.concatenate([product(row) for row in mx.split(x, x.shape[1], axis=1)], axis=1)


def qualified(signature, candidate, reference):
    """Use a shared panel stage only after it matches native one-row arithmetic."""
    decisions = active.get()
    decision = decisions.get(signature)
    if decision is False:
        return reference()
    if decision is True:
        return candidate()
    try:
        actual = candidate()
        if actual is not None:
            wanted = reference()
            if isinstance(actual, (list, tuple)):
                checks = [same_bits(a, b) for a, b in zip(actual, wanted, strict=True)]
                equal = mx.all(mx.stack(checks))
            else:
                equal = same_bits(actual, wanted)
            decisions[signature] = bool(equal.item())
            return actual if decisions[signature] else wanted
    except (RuntimeError, ValueError):
        pass
    decisions[signature] = False
    return reference()


def same_bits(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        return mx.array(False)
    return mx.array_equal(actual.view(mx.uint32 if actual.dtype.size == 4 else mx.uint16),
                          reference.view(mx.uint32 if reference.dtype.size == 4 else mx.uint16))


class VisitTape:
    """Read-only views of the appended window rows and FP32 convolution operands."""

    def __init__(self, rows):
        self.rows = rows
        self.window = None
        self.conv_histories = [None] * 4

    def record_conv(self, slot, old, incoming, taps):
        if old is None:
            old = mx.zeros((1, taps, incoming.shape[-1]), mx.float32)
        self.conv_histories[slot] = mx.concatenate([old, incoming.astype(mx.float32)], axis=1)

    def boundary(self, offset, kept, window):
        state = None
        if self.window is not None:
            end = self.window[0].shape[2] - self.rows + kept
            state = tuple(x[:, :, max(0, end - window):end] for x in self.window)
        conv = [history[:, kept:kept + history.shape[1] - self.rows]
                for history in self.conv_histories]
        return offset + kept, state, conv
