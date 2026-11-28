// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

            uint i = thread_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            float x = float(input[i]);
            if (FP4) {
                float peak = peaks[i / WIDTH];
                float outer = peak / 1536.0f;
                uint first = lane & 16u;
                float top = abs(x);
                for (uint d = 1; d < 16; d *= 2)
                    top = max(top, simd_shuffle_xor(top, d));
                float encoder = peak > 0 ? 1536.0f / peak : 0;
                float s6 = inkling_e4m3(top * (encoder / 6.0f));
                float s4 = inkling_e4m3(top * (encoder / 4.0f));
                float a = s6 == 0 ? 0 : inkling_e2m1(x / (s6 * outer)) * s6;
                float b = s4 == 0 ? 0 : inkling_e2m1(x / (s4 * outer)) * s4;
                float ae = abs(a * outer - x), be = abs(b * outer - x);
                for (uint d = 1; d < 16; d *= 2) {
                    ae += simd_shuffle_xor(ae, d);
                    be += simd_shuffle_xor(be, d);
                }
                values[i] = simd_shuffle(be, first) < simd_shuffle(ae, first) ? b : a;
            } else {
                float top = simd_max(abs(x));
                float scale = top > 0 ? exp2(ceil(log2(top / 448.0f))) : 1.0f;
                values[i] = inkling_e4m3(x / scale) * scale;
            }
