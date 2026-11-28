// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
// MLX-One host module: models/language/inkling2/activations.py
// Begin shader source

            inline float inkling_e4m3(float x) {
                float a = min(abs(x), 448.0f);
                float unit = ldexp(1.0f, max(-9, ilogb(max(a, 0.015625f)) - 3));
                return copysign(rint(a / unit) * unit, x);
            }
            inline float inkling_e2m1(float x) {
                float a = abs(x);
                float y = a > 5 ? 6 : a >= 3.5f ? 4 : a > 2.5f ? 3 : a >= 1.75f ? 2 :
                          a > 1.25f ? 1.5f : a >= .75f ? 1 : a > .25f ? .5f : 0;
                return copysign(y, x);
            }
