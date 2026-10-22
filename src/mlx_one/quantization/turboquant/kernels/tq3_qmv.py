"""Coalesced TQ3 GEMV: eight SIMD groups per threadgroup, no dense writes."""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="tq3_words_qmv",
        input_names=["packed", "scales", "codebook", "x"],
        output_names=["out"],
        source="""
        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32u;
        uint row = threadgroup_position_in_grid.x * 8u + tid / 32u;
        uint O = packed_shape[0], P = packed_shape[1];
        uint K = x_shape[0], G = scales_shape[1];
        if (row >= O) return;
        float cb[8];
        #pragma unroll
        for (uint j = 0; j < 8u; ++j) cb[j] = float(codebook[j]);
        float acc = 0;
        for (uint wi = lane; wi < P; wi += 32u) {
            uint word = packed[row * P + wi];
            #pragma unroll
            for (uint j = 0; j < 10u; ++j) {
                uint col = wi * 10u + j;
                if (col < K) {
                    float w = cb[(word >> (3u*j)) & 7u]
                        * float(scales[row*G + col/64u]);
                    acc = fma(w, float(x[col]), acc);
                }
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) out[row] = T(acc);
        """,
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
