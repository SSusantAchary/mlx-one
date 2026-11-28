// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    const uint lane = thread_index_in_simdgroup;
    const uint head = threadgroup_position_in_grid.y;
    const uint heads = partial_shape[1];
    const uint pages = partial_shape[0];
    float maximum = -INFINITY;
    for (uint page = lane; page < pages; page += 32) {
        const uint base = (page * heads + head) * (D + 2);
        maximum = max(maximum, partial[base + D]);
    }
    maximum = simd_max(maximum);
    const float center = maximum == -INFINITY ? 0.0f : maximum;
    float normalizer = 0.0f;
    float accumulated[D / 32] = {};
    for (uint page = 0; page < pages; ++page) {
        const uint base = (page * heads + head) * (D + 2);
        const float factor = exp(partial[base + D] - center);
        normalizer += partial[base + D + 1] * factor;
        for (uint j = 0; j < D / 32; ++j)
            accumulated[j] += partial[base + lane + j * 32] * factor;
    }
    for (uint j = 0; j < D / 32; ++j)
        out[head * D + lane + j * 32] = T(accumulated[j] / max(normalizer, 1e-20f));
