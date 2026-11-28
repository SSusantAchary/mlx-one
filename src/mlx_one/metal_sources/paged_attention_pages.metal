// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    // One SIMD group per head and logical page for batch-one decode.
    uint lane = thread_index_in_simdgroup;
    uint head = threadgroup_position_in_grid.y;
    uint page = threadgroup_position_in_grid.z;
    uint row = 0, qi = 0;
    uint kh = head / (H / HK);
    int sid = table0[page * 4], slot = table0[page * 4 + 1];
    int start = table0[page * 4 + 2], count = table0[page * 4 + 3];
    float query[D / 32], acc[D / 32];
    for (uint j = 0; j < D / 32; ++j) {
        query[j] = float(q[head * D + lane + j * 32]);
        acc[j] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
${segments}
    uint base = (page * H + head) * (D + 2);
    for (uint j = 0; j < D / 32; ++j)
        out[base + lane + j * 32] = acc[j];
    if (lane == 0) {
        out[base + D] = m;
        out[base + D + 1] = l;
    }
