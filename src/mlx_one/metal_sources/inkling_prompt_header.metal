// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
// See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
// Source: mtplx/models/inkling/fast/prompt_flash.py.
// Reduction order follows MLX v0.32.2 Steel SDPA. See mlxboost/native/NOTICE.txt.
// MLX-One host module: models/language/inkling2/_prompt_attention_metal.py
// Begin shader source

#define INK_EVERY _Pragma("clang loop unroll_count(8)")

template <typename E>
inline vec<E, 2> ink_two(const device E* p, uint step) {
    if (step == 0u) {
        return *reinterpret_cast<const device vec<E, 2>*>(p);
    }
    return vec<E, 2>(p[0], p[step]);
}

template <typename E>
inline void ink_put_two(device E* p, vec<E, 2> x) {
    *reinterpret_cast<device vec<E, 2>*>(p) = x;
}
