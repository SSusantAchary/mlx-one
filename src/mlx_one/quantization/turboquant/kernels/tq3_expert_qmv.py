"""Coalesced packed sparse GEMV, with GPU indices and shared/per-route inputs."""

from functools import lru_cache

import mlx.core as mx

from mlx_one.metal_source import metal_source


@lru_cache(maxsize=2)
def _kernel(shared):
    row = "route / uint(repeats[0])" if shared else "route"
    source = metal_source("turboquant_expert_qmv.metal").replace("XROW", row)
    return mx.fast.metal_kernel(
        name=f"tq3_expert_words_shared{int(shared)}",
        input_names=["packed", "scales", "codebook", "x", "indices", "repeats"],
        output_names=["out"],
        source=source,
    )


def expert_qmv(packed, scales, codebook, x, indices, repeats=1):
    routes, outputs = indices.size, packed.shape[1]
    return _kernel(repeats > 1)(
        inputs=[
            packed,
            scales,
            codebook,
            x,
            indices.astype(mx.uint32),
            mx.array([repeats], mx.uint32),
        ],
        template=[("T", x.dtype)],
        grid=(((outputs + 7) // 8) * 256, routes, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(routes, outputs)],
        output_dtypes=[x.dtype],
    )[0]
