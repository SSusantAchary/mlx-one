// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad,
// models/inkling/fast/{attend,prefill_attend}.py. See MTPLX_NOTICE.txt.
// MLX-One host module: models/language/inkling2/_long_attention_metal.py
// Begin shader source

    const int h = int(threadgroup_position_in_grid.x);
    const int r = int(threadgroup_position_in_grid.y);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const size_t base = (size_t(r) * 24 + size_t(h)) * BLOCKS;
    // Merging one row's chains for one head. Lane l reads chain l (and l + 32, ... on two-pass
    // rows) for the overall peak and mass; simdgroup s sums the value parts of its chains over
    // its lanes' four dimensions; one transposition through threadgroup memory then lets
    // simdgroup s total dimension 4 s + d across the 32 chain groups, in lane order.
    threadgroup float xpose[4 * 32 * 32];      // [d][chain group][simdgroup]
    float part[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float mass;
    float peak = -FLT_MAX;
    for (int b = 0; b < BLOCKS / 32; ++b) {
        peak = max(peak, pm[base + lane + 32 * b]);
    }
    peak = simd_max(peak);
    mass = 0.0f;
    for (int b = 0; b < BLOCKS / 32; ++b) {
        mass += metal::fast::exp(pm[base + lane + 32 * b] - peak) * ps[base + lane + 32 * b];
    }
    mass = simd_sum(mass);
    for (int b = 0; b < BLOCKS / 32; ++b) {
        const size_t c = base + sg + 32 * b;
        const float keep = metal::fast::exp(pm[c] - peak);
        for (int d = 0; d < 4; ++d) {
            part[d] += keep * po[c * 128 + 4 * lane + d];
        }
    }
    for (int d = 0; d < 4; ++d) {
        xpose[(d * 32 + lane) * 32 + sg] = part[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = 0; d < 4; ++d) {
        const float mine = xpose[(d * 32 + sg) * 32 + lane];
        const float total = simd_sum(mine);
        part[d] = mass == 0 ? total : (total / mass);
    }
    if (lane == 0) {
        for (int i = 0; i < 4; ++i) {
            const size_t at = size_t(r) * 3072 + size_t(h) * 128 + size_t(4 * sg + i);
            hg[at] = static_cast<T>(part[i]);
        }
    }
