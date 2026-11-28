// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    #pragma clang fp contract(off)
    const uint row = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint width = weight_shape[0];
    const uint base = row * width;
    float sum = 0.0f;
    for (uint col = lane; col < width; col += 32) {
        const float x = hidden[base + col];
        sum += x * x;
    }
    const float inv = metal::precise::rsqrt(simd_sum(sum) / float(width) + eps[0]);
    for (uint col = lane; col < width; col += 32) {
        output[base + col] = (hidden[base + col] * inv) * weight[col];
    }
