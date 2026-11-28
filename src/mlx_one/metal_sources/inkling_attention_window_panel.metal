// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

            const int head = int(threadgroup_position_in_grid.x);
            const int row = int(threadgroup_position_in_grid.y);
            const int chain = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);
            const int rows = q_shape[2], count = k_shape[2];
            const int end = count - rows + row + 1;
            const int first = max(0, end - WINDOW);
            const int kvhead = head / 3;
            const float scale = 0.08838834764831845f;
            const device T* qp = q + head * q_strides[1] + row * q_strides[2];
            const device T* bias = p + row * p_strides[1] + head * p_strides[2];
            float query[4];
            for (int d = 0; d < 4; ++d) {
                query[d] = scale * float(qp[(4 * lane + d) * q_strides[3]]);
            }
            Chain state;
            chain_open(state);
            if (chain == 0) {
                state.peak = float(s[row * s_strides[1] + head * s_strides[2]]);
                state.mass = 1.0f;
            }
            for (int pos = first + chain; pos < end; pos += 32) {
                const device T* kp = k + kvhead * k_strides[1] + pos * k_strides[2];
                const device T* vp = v + kvhead * v_strides[1] + pos * v_strides[2];
                float keys[4], values[4];
                for (int d = 0; d < 4; ++d) {
                    keys[d] = float(kp[(4 * lane + d) * k_strides[3]]);
                    values[d] = float(vp[(4 * lane + d) * v_strides[3]]);
                }
                const float b = float(bias[(end - 1 - pos) * p_strides[3]]);
                if (b >= -3.3895313892515355e38f) {
                    chain_take(state, key_logit(query, keys, b), values);
                }
            }
            threadgroup float peaks[32], masses[32], transpose[32 * 32];
            if (lane == 0) {
                peaks[chain] = state.peak;
                masses[chain] = state.mass;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float peak = peaks[lane];
            const float rescale = metal::fast::exp(peak - simd_max(peak));
            const float mass = simd_sum(masses[lane] * rescale);
            for (int d = 0; d < 4; ++d) {
                transpose[lane * 32 + chain] = state.acc[d];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float total = simd_sum(transpose[chain * 32 + lane] * rescale);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    out[((head * rows + row) * 128) + 4 * chain + d] =
                        T(mass == 0 ? total : total / mass);
                }
            }
