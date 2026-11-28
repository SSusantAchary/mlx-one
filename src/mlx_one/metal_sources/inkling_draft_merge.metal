// Copyright 2026 Youssof Altoukhi. SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, models/inkling/fast/round_program.py.
// See MTPLX_LICENSE.txt and MTPLX_NOTICE.txt.
// MLX-One host module: models/language/inkling2/_replay_draft_attention.py
// Begin shader source

    uint qv = threadgroup_position_in_grid.x;
    uint h = threadgroup_position_in_grid.y;
    uint d = thread_position_in_threadgroup.x;
    const uint QV = G * R;
    const uint C = (S + CHUNK - 1) / CHUNK + 1;
    float mx_ = -INFINITY;
    for (uint c = 0; c < C; ++c) mx_ = metal::max(mx_, pm[(h * C + c) * QV + qv]);
    float lsum = 0.0f, o = 0.0f;
    for (uint c = 0; c < C; ++c) {
        uint pi = (h * C + c) * QV + qv;
        float w = metal::exp(pm[pi] - mx_);
        lsum += pl[pi] * w;
        o += pacc[pi * D + d] * w;
    }
    out[((h * G + qv / R) * R + qv % R) * D + d] = static_cast<T>(o / lsum);
