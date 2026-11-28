// Copyright 2026 Youssof Altoukhi. SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, models/inkling/fast/round_program.py.
// See MTPLX_LICENSE.txt and MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

    uint c = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32, lane = tid % 32;
    const uint QV = G * R;
    const uint NHIST = (S + CHUNK - 1) / CHUNK;
    const uint C = NHIST + 1;
    threadgroup float qs[QV * D];
    threadgroup T kt[TK * D];
    threadgroup T vt[TK * D];
    const float scale = metal::rsqrt(float(D));
    for (uint i = tid; i < QV * D; i += 256) {
        uint qv = i / D, d = i % D;
        qs[i] = float(q[((h * G + qv / R) * R + qv % R) * D + d]) * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = length[0];
    const bool own = c == NHIST;
    const uint start = own ? 0 : c * CHUNK;
    // a bank's rows past the count are not history: a chunk wholly past it reads nothing
    const uint end = own ? R : metal::min(metal::min(start + CHUNK, (uint)S),
                                           metal::max(count, start));
    float m[QPS], l[QPS], acc[QPS][D / 32];
    for (uint a = 0; a < QPS; ++a) {
        m[a] = -INFINITY; l[a] = 0.0f;
        for (uint e = 0; e < D / 32; ++e) acc[a][e] = 0.0f;
    }
    for (uint base = start; base < end; base += TK) {
        uint n = metal::min((uint)TK, end - base);
        for (uint i = tid; i < n * D; i += 256) {
            uint j = base + i / D, d = i % D;
            kt[i] = own ? ok[(h * R + j) * D + d] : hk[(h * S + j) * D + d];
            vt[i] = own ? ov[(h * R + j) * D + d] : hv[(h * S + j) * D + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < n; ++t) {

            float kd[D / 32], vd[D / 32];
            for (uint e = 0; e < D / 32; ++e) {
                kd[e] = float(kt[t * D + lane * (D / 32) + e]);
                vd[e] = float(vt[t * D + lane * (D / 32) + e]);
            }
            for (uint a = 0; a < QPS; ++a) {
                uint qv = sg * QPS + a;
                float part = 0.0f;
                for (uint e = 0; e < D / 32; ++e) part += qs[qv * D + lane * (D / 32) + e] * kd[e];
                float sc = simd_sum(part);
                float mn = metal::max(m[a], sc);
                float corr = metal::exp(m[a] - mn);
                float p = metal::exp(sc - mn);
                l[a] = l[a] * corr + p;
                for (uint e = 0; e < D / 32; ++e) acc[a][e] = acc[a][e] * corr + p * vd[e];
                m[a] = mn;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint a = 0; a < QPS; ++a) {
        uint qv = sg * QPS + a;
        uint pi = (h * C + c) * QV + qv;
        if (lane == 0) { pm[pi] = m[a]; pl[pi] = l[a]; }
        for (uint e = 0; e < D / 32; ++e) pacc[pi * D + lane * (D / 32) + e] = acc[a][e];
    }
