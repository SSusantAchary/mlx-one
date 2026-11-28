// Copyright © 2026 Neywa Labs Inc, Abdur Rahim
// SPDX-License-Identifier: MIT
// Native Metal implementation by MLX-One
// Begin shader source

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
