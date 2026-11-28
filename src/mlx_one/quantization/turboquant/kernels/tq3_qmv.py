"""Coalesced TQ3 GEMV: eight SIMD groups per threadgroup, no dense writes."""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="tq3_words_qmv",
        input_names=["packed", "scales", "codebook", "x"],
        output_names=["out"],
        source=metal_source("turboquant_qmv.metal"),
    )


def tq3_qmv(packed_weight, scales, codebook, x, bits=3, group_size=64):
    if bits != 3 or group_size != 64 or x.size != x.shape[-1]:
        raise ValueError("TQ3 GEMV requires 3-bit/g64 and one input vector")
    shape = (packed_weight.shape[0],) if x.ndim == 1 else (1, packed_weight.shape[0])
    return _kernel()(
        inputs=[packed_weight, scales, codebook, x.reshape(-1)],
        template=[("T", x.dtype)],
        grid=(((packed_weight.shape[0] + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
    )[0]
