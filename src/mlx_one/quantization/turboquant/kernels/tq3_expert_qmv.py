"""Coalesced packed sparse GEMV, with GPU indices and shared/per-route inputs."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=2)
def _kernel(shared):
    source = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32u;
    uint row = threadgroup_position_in_grid.x * 8u + tid/32u;
    uint route = threadgroup_position_in_grid.y;
    uint O = packed_shape[1], P = packed_shape[2];
    uint K = x_shape[1], G = scales_shape[2];
    if (row >= O) return;
    uint expert = indices[route];
    uint xb = XROW * K;
    float acc = 0;
    for (uint wi = lane; wi < P; wi += 32u) {
        uint word = packed[(expert * O + row) * P + wi];
        #pragma unroll
        for (uint j = 0; j < 10u; ++j) {
            uint col = wi * 10u + j;
            if (col < K) {
                float w = float(codebook[(word >> (3u*j)) & 7u])
                    * float(scales[(expert * O + row)*G + col/64u]);
                acc = fma(w, float(x[xb + col]), acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) out[route * O + row] = T(acc);
    """.replace("XROW", "route / uint(repeats[0])" if shared else "route")
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
