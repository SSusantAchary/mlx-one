// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Adapted from MLX v0.32.2 sdpa_vector.h, commit
// 1f8e74e3f12f31365464a6867c6579f0e9b29d85; see models/UPSTREAM_LICENSE.
// Native Metal implementation by MLX-One
// Begin shader source

        constexpr int G = 8;
        constexpr int HPT = G;
        constexpr int D = 64;
        constexpr float lowest = -3.402823466e+38f;
        uint lane = thread_position_in_threadgroup.x;
        uint g = thread_position_in_threadgroup.y;
        uint kv_head = threadgroup_position_in_grid.x;
        uint block = threadgroup_position_in_grid.z;
        int N = keys_shape[2];
        int chunk = (N + BLOCKS - 1) / BLOCKS;
        int start = int(block) * chunk;
        int end = min(N, start + chunk);
        int sub = (chunk + SUBGROUPS - 1) / SUBGROUPS;
        int s0 = start + int(g) * sub;
        int s1 = min(end, s0 + sub);
        uint base_head = kv_head * G;

        float q[HPT][2];
        float largest[HPT];
        float denominator[HPT];
        float accumulator[HPT][2];
        for (int h = 0; h < HPT; ++h) {
            for (int i = 0; i < 2; ++i) {
                size_t qi = (base_head + h) * queries_strides[1]
                    + (lane * 2 + i) * queries_strides[3];
                q[h][i] = 0.125f * float(queries[qi]);
                accumulator[h][i] = 0;
            }
            // Include the sink once across all splits/subchunks, for every head.
            bool include_sink = block == 0 && g == 0;
            largest[h] = include_sink
                ? float(sinks[(base_head + h) * sinks_strides[0]]) : lowest;
            denominator[h] = include_sink ? 1.0f : 0.0f;
        }
        for (int t = s0; t < s1; ++t) {
            float kr[2], vr[2];
            for (int i = 0; i < 2; ++i) {
                kr[i] = float(keys[kv_head * keys_strides[1] + t * keys_strides[2]
                    + (lane * 2 + i) * keys_strides[3]]);
                vr[i] = float(values[kv_head * values_strides[1] + t * values_strides[2]
                    + (lane * 2 + i) * values_strides[3]]);
            }
            for (int h = 0; h < HPT; ++h) {
                float score = q[h][0] * kr[0] + q[h][1] * kr[1];
                score = simd_sum(score);
                float next_max = max(largest[h], score);
                float correction = fast::exp(largest[h] - next_max);
                float weight = fast::exp(score - next_max);
                largest[h] = next_max;
                denominator[h] = denominator[h] * correction + weight;
                for (int i = 0; i < 2; ++i)
                    accumulator[h][i] = accumulator[h][i] * correction + weight * vr[i];
            }
        }
        if (SUBGROUPS == 1) {
            for (int h = 0; h < HPT; ++h) {
                uint index = (base_head + h) * BLOCKS + block;
                for (int i = 0; i < 2; ++i)
                    partials[index * D + lane * 2 + i] = T(accumulator[h][i]);
                if (lane == 0) {
                    sums[index] = denominator[h];
                    maxs[index] = largest[h];
                }
            }
            return;
        }
        threadgroup float local_output[G * SUBGROUPS * D];
        threadgroup float local_sums[G * SUBGROUPS];
        threadgroup float local_maxs[G * SUBGROUPS];
        for (int h = 0; h < HPT; ++h) {
            uint slot = h * SUBGROUPS + g;
            float inv = denominator[h] > 0 ? 1.0f / denominator[h] : 0;
            for (int i = 0; i < 2; ++i)
                local_output[slot * D + lane * 2 + i] = accumulator[h][i] * inv;
            if (lane == 0) {
                local_sums[slot] = denominator[h];
                local_maxs[slot] = largest[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint h = g; h < HPT; h += SUBGROUPS) {
            uint slot = h * SUBGROUPS;
            float largest_value = lowest;
            for (int s = 0; s < SUBGROUPS; ++s)
                largest_value = max(largest_value, local_maxs[slot + s]);
            float sum = 0, acc[2] = {0, 0};
            for (int s = 0; s < SUBGROUPS; ++s) {
                float factor = local_sums[slot + s]
                    * fast::exp(local_maxs[slot + s] - largest_value);
                sum += factor;
                for (int i = 0; i < 2; ++i)
                    acc[i] += factor * local_output[(slot + s) * D + lane * 2 + i];
            }
            uint index = (base_head + h) * BLOCKS + block;
            for (int i = 0; i < 2; ++i)
                partials[index * D + lane * 2 + i] = T(acc[i]);
            if (lane == 0) {
                sums[index] = sum;
                maxs[index] = largest_value;
            }
        }
