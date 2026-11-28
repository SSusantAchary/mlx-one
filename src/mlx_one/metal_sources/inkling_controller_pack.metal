// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    #pragma clang fp contract(off)
    const uint row = threadgroup_position_in_grid.y;
    const bool message_half = threadgroup_position_in_grid.z != 0;
    const uint col = 4 * thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint group = simdgroup_index_in_threadgroup;
    const uint groups = threads_per_threadgroup.x / 32;
    threadgroup float partial[32];
    threadgroup float inverse;

    float values[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = message_half
            ? message[row * message_strides[1] + (col + i) * message_strides[2]]
            : x[row * x_strides[1] + (col + i) * x_strides[2]];
        sum += values[i] * values[i];
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (group == 0) {
        const float total = simd_sum(lane < groups ? partial[lane] : 0.0f);
        if (lane == 0) inverse = metal::precise::rsqrt(total / float(WIDTH) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start = (2 * row + uint(message_half)) * WIDTH + col;
    for (uint i = 0; i < 4; ++i) {
        packed[start + i] = static_cast<bfloat>(values[i] * inverse);
    }
