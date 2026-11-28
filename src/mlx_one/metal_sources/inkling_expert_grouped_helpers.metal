// Copyright © 2026 Neywa Labs Inc, Abdur Rahim
// SPDX-License-Identifier: MIT
// MLX-One host module: models/language/inkling2/grouped_experts.py
// Begin shader source

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
