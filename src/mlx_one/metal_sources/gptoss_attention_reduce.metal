// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Adapted from MLX v0.32.2 sdpa_vector.h, commit
// 1f8e74e3f12f31365464a6867c6579f0e9b29d85; see models/UPSTREAM_LICENSE.
// MLX-One host module: mlxboost/gptoss_attention.py
// Begin shader source

        uint lane = thread_position_in_threadgroup.x;
        uint head = threadgroup_position_in_grid.x;
        float largest = -3.402823466e+38f;
        for (int b = 0; b < BLOCKS; b += 32)
            largest = max(largest, maxs[head * BLOCKS + b + lane]);
        largest = simd_max(largest);
        float denominator = 0, acc[2] = {0, 0};
        for (int b = 0; b < BLOCKS; b += 32) {
            float factor = fast::exp(maxs[head * BLOCKS + b + lane] - largest);
            denominator += simd_sum(sums[head * BLOCKS + b + lane] * factor);
            for (int s = 0; s < 32; ++s) {
                float weight = simd_shuffle(factor, ushort(s));
                for (int i = 0; i < 2; ++i)
                    acc[i] += weight * float(partials[
                        (head * BLOCKS + b + s) * 64 + lane * 2 + i]);
            }
        }
        for (int i = 0; i < 2; ++i)
            out[head * 64 + lane * 2 + i] = T(denominator > 0 ? acc[i] / denominator : 0);
