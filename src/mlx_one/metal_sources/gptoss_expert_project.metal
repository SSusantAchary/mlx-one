// Copyright © 2023 Apple Inc. (FP4/E8M0 representation reference, MIT)
// New MLX-One projection/fusion implementation: Apache-2.0.
// See models/UPSTREAM_LICENSE and docs/acceleration/gptoss20b/upstream_references.md.
// Native Metal implementation by MLX-One
// Begin shader source

    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    uint choice = thread_position_in_grid.z;
    uint N = weight_shape[1];
    uint K = x_shape[1];
    if (row >= N) return;
    long expert_value = long(indices[choice]);
    if (expert_value < 0 || expert_value >= long(weight_shape[0])) {
        if (lane == 0) {
    INVALID_WRITE
        }
        return;
    }
    uint expert = uint(expert_value);
    uint input_row = x_shape[0] == 1 ? 0 : choice;
    uint base = (expert * N + row) * (K / 8);
    uint sb = (expert * N + row) * (K / 32);
    float acc = 0.0f;
    PAIR_ACC
    for (uint p0 = lane * WORDS; p0 < K / 8; p0 += 32 * WORDS) {
        #pragma unroll
        for (uint w = 0; w < WORDS; ++w) {
            uint p = p0 + w;
            if (p >= K / 8) break;
            uint packed = weight[base + p];
            float scale = gptoss_scale(scales[sb + p / 4]);
            PAIR_LOAD
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                float value = float(x[input_row * K + p * 8 + j]);
                float weight_value = gptoss_fp4((packed >> (j * 4)) & 15u) * scale;
                acc += value * weight_value;
                PAIR_DOT
            }
        }
    }
    T gate_value = T(simd_sum(acc));
    gate_value = T(float(gate_value) + float(bias[expert * N + row]));
    PAIR_REDUCE
    if (lane == 0) {
        RESULT_WRITE
    }
