# Copyright © 2026 Neywa Labs Inc, Abdur Rahim
# SPDX-License-Identifier: MIT
"""Adapt Neywa's grouped expert QMV to MLX-One's FP32 NVFP4 operands.

Source: mlx-vlm, mlx_vlm/models/inkling2/grouped_experts.py at
8e80d434fe59f4d4e00bc91a418e0a16ad951326. See NEYWA_LICENSE.txt.
MLX-One keeps the interleaved gate/up matrix and applies the outer scales
after the raw product, at the same precision as its native gather path.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

from .long_attention import _supported_gpu


_TILE = r"""
inline float decode_nvfp4(uchar bits) {
    return float(*reinterpret_cast<thread fp4_e2m1*>(&bits));
}

inline float decode_nvscale(uchar bits) {
    return float(*reinterpret_cast<thread fp8_e4m3*>(&bits));
}

template <int V>
inline void expert_tile(
    const device float* x, const device uint* weights, const device uchar* scales,
    device float* out, const thread uint* src, const thread uint* dst,
    uint expert, uint row, uint lane, uint K, uint N) {
    constexpr int P = 8;  // MLX's non-fast gather_qmv reads one packed word per lane.
    const size_t channel = size_t(expert) * N + row;
    const uint weight_stride = K / 2, scale_stride = K / 16;
    const device uchar* packed = reinterpret_cast<const device uchar*>(weights)
        + channel * weight_stride;
    const device uchar* scale = scales + channel * scale_stride;
    float sums[V][4] = {};
    for (uint block = 0; block < K; block += 32 * P) {
        const uint col = block + lane * P;
        // The supported K values are multiples of P, including the final partial block.
        if (col >= K) break;
        float activations[V][P];
        #pragma unroll
        for (int v = 0; v < V; ++v) {
            #pragma unroll
            for (int i = 0; i < P; ++i) activations[v][i] = x[size_t(src[v]) * K + col + i];
        }
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const device uchar* w = packed + r * weight_stride + col / 2;
            const float s = decode_nvscale(scale[r * scale_stride + col / 16]);
            float decoded[P];
            #pragma unroll
            for (int i = 0; i < P; ++i) decoded[i] = decode_nvfp4(w[i / 2] >> (4 * (i % 2)));
            #pragma unroll
            for (int v = 0; v < V; ++v) {
                float dot = 0;
                // Match MLX gather_qmv's four-product sums and per-block scaling.
                #pragma unroll
                for (int i = 0; i < P; i += 4) {
                    dot += (activations[v][i] * decoded[i]
                            + activations[v][i + 1] * decoded[i + 1]
                            + activations[v][i + 2] * decoded[i + 2]
                            + activations[v][i + 3] * decoded[i + 3]);
                }
                sums[v][r] += s * dot;
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < V; ++v) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float result = simd_sum(sums[v][r]);
            if (lane == 0) out[size_t(dst[v]) * N + row + r] = result;
        }
    }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    headers = Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels"
    # Use the installed MLX decoders for its exact NVFP4 and E4M3 representation.
    fp_headers = "\n".join((headers / name).read_text().replace("#pragma once", "")
                           for name in ("fp4.h", "fp8.h"))
    cases = "\n".join(
        f"case {width}: expert_tile<{width}>(x, w, s, out, src, dst, expert, row, lane, K, N); break;"
        for width in range(1, 5)
    )
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_grouped_nvfp4",
        input_names=["x", "w", "s", "experts", "source", "destination"],
        output_names=["out"],
        header=fp_headers + "\n" + _TILE,
        compile_options={"math_mode": "safe"},
        source=r"""
            const uint first = threadgroup_position_in_grid.z;
            const uint expert = experts[first];
            if (first > 0 && experts[first - 1] == expert) return;
            uint end = first + 1;
            while (end < experts_shape[0] && experts[end] == expert) ++end;
            const uint K = x_shape[1], N = w_shape[1];
            const uint row = threadgroup_position_in_grid.y * 8
                + simdgroup_index_in_threadgroup * 4;
            const uint lane = thread_index_in_simdgroup;
            for (uint begin = first; begin < end; begin += 4) {
                const uint count = min(4u, end - begin);
                uint src[4], dst[4];
                for (uint i = 0; i < count; ++i) {
                    src[i] = source[begin + i];
                    dst[i] = destination[begin + i];
                }
                switch (count) {
        """ + cases + "}}",
    )


def raw_product(inputs, codes, scales, experts, source, destination):
    """Read sorted expert runs and write FP32 products in the original slot order."""
    if (inputs.ndim != 3 or inputs.shape[1] != 1 or inputs.dtype != mx.float32
            or codes.ndim != 3 or codes.dtype != mx.uint32 or scales.dtype != mx.uint8
            or experts.ndim != 1 or not 16 <= experts.size <= 32
            or any(x.shape != experts.shape or x.dtype != mx.uint32
                   for x in (experts, source, destination))
            or mx.default_device() != mx.gpu or not _supported_gpu()):
        return None
    width = inputs.shape[-1]
    banks, outputs, words = codes.shape
    if (width not in (1152, 2304) or (banks, outputs) != (128, 2304)
            or words != width // 8 or scales.shape != (banks, outputs, width // 16)):
        return None
    try:
        kernel = _kernel()
    except OSError:
        return None
    return kernel(
        inputs=[inputs.reshape(-1, width), codes, scales, experts, source, destination],
        grid=(32, outputs // 4, experts.size),
        threadgroup=(32, 2, 1),
        output_shapes=[(experts.size, 1, outputs)],
        output_dtypes=[mx.float32],
    )[0]
