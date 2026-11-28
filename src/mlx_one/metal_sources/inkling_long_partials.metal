// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad,
// models/inkling/fast/{attend,prefill_attend}.py. See MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

    const int rho = int(threadgroup_position_in_grid.x);
    const int g = int(threadgroup_position_in_grid.y);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int r = int(threadgroup_position_in_grid.z) * ROWS + sg;
    if (r >= int(q_shape[1])) return;
    const int lane = int(thread_index_in_simdgroup);
    const int pos0 = geometry[0], keys = geometry[1];
    const int P = pos0 + r, span = BLOCKS;
    const float scale = as_type<float>(uint(SCALE));
    threadgroup float qsh[ROWS * 3 * 128];
    threadgroup float esh[ROWS * 32 * 4];
    float qs[3][4];
    Chain ch[3];
    for (int h3 = 0; h3 < 3; ++h3) {
        for (int j = 0; j < 4; ++j) {
            qs[h3][j] = static_cast<float>(scale) * static_cast<float>(
                q[r * q_strides[1] + (3 * g + h3) * q_strides[2] + 4 * lane + j]);
        }
        chain_open(ch[h3]);
        // Native one-row SDPA starts block zero with the sink's mass.
        if (SINK_FIRST && rho == 0) {
            ch[h3].peak = float(sink[r * sink_strides[1] + (3 * g + h3) * sink_strides[2]]);
            ch[h3].mass = 1.0f;
        }
    }
    const device T* pr = prof + (size_t(r) * 24 + size_t(3 * g)) * 1024;
    const device T* kg = kc + size_t(g) * kc_strides[1];
    const device T* vg = vc + size_t(g) * vc_strides[1];

    int p_from = rho;
    if (p_from < pos0) {
        // Read keys in 16-element pieces and preserve MLX's lane reduction tree.
        // Threadgroup memory shares each key's weights across the three query heads.
        threadgroup float4* qr4 = (threadgroup float4*)(qsh + sg * 3 * 128);
        threadgroup float4* es = (threadgroup float4*)(esh + sg * 32 * 4);
        for (int h3 = 0; h3 < 3; ++h3) {
            qr4[h3 * 32 + lane] = float4(qs[h3][0], qs[h3][1], qs[h3][2], qs[h3][3]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int ncom = (pos0 - 1 - p_from) / span + 1;
        const int tiles = ncom >> 5;
        const size_t kst = size_t(kc_strides[2]);
        const size_t vst2 = size_t(vc_strides[2]) / 4;
        const size_t vstep2 = size_t(span) * vst2;
        for (int tix = 0; tix < tiles; ++tix) {
            const int base = p_from + tix * 32 * span;
            const int pt = base + lane * span;
            const device uint4* kp = (const device uint4*)(kg + size_t(pt) * kst);
            float t01[3], t2[3], sc[3];
            for (int cc = 0; cc < 4; ++cc) {
                float s8[3];
                for (int hh = 0; hh < 2; ++hh) {
                    const uint4 w0 = kp[4 * cc + 2 * hh];
                    const uint4 w1 = kp[4 * cc + 2 * hh + 1];
                    const float kf[16] = {
                        as_type<float>(w0.x << 16), as_type<float>(w0.x & 0xffff0000u),
                        as_type<float>(w0.y << 16), as_type<float>(w0.y & 0xffff0000u),
                        as_type<float>(w0.z << 16), as_type<float>(w0.z & 0xffff0000u),
                        as_type<float>(w0.w << 16), as_type<float>(w0.w & 0xffff0000u),
                        as_type<float>(w1.x << 16), as_type<float>(w1.x & 0xffff0000u),
                        as_type<float>(w1.y << 16), as_type<float>(w1.y & 0xffff0000u),
                        as_type<float>(w1.z << 16), as_type<float>(w1.z & 0xffff0000u),
                        as_type<float>(w1.w << 16), as_type<float>(w1.w & 0xffff0000u)};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        float part[4];
                        for (int m4 = 0; m4 < 4; ++m4) {
                            const float4 qq = qr4[h3 * 32 + 8 * cc + 4 * hh + m4];
                            float lanepart = 0;
                            lanepart += qq.x * kf[4 * m4 + 0];
                            lanepart += qq.y * kf[4 * m4 + 1];
                            lanepart += qq.z * kf[4 * m4 + 2];
                            lanepart += qq.w * kf[4 * m4 + 3];
                            part[m4] = lanepart;
                        }
                        const float s4 = (part[0] + part[1]) + (part[2] + part[3]);
                        s8[h3] = hh == 0 ? s4 : (s8[h3] + s4);
                    }
                }
                for (int h3 = 0; h3 < 3; ++h3) {
                    if (cc == 0) {
                        t01[h3] = s8[h3];
                    } else if (cc == 1) {
                        t01[h3] = t01[h3] + s8[h3];
                    } else if (cc == 2) {
                        t2[h3] = s8[h3];
                    } else {
                        sc[h3] = t01[h3] + (t2[h3] + s8[h3]);
                    }
                }
            }
            const int dist = P - pt;
            float wt[3], kp3[3];
            bool all_fast = true;
            for (int h3 = 0; h3 < 3; ++h3) {
                float logit = sc[h3];
                logit += dist < 1024 ? static_cast<float>(pr[h3 * 1024 + dist]) : 0.0f;
                const float tile_top = simd_max(logit);
                if (tile_top <= ch[h3].peak) {
                    kp3[h3] = 1.0f;
                    wt[h3] = metal::fast::exp(logit - ch[h3].peak);
                } else {
                    all_fast = false;
                    float run = logit;
                    for (ushort d2 = 1; d2 < 32; d2 <<= 1) {
                        const float other = simd_shuffle_up(run, d2);
                        if (lane >= int(d2)) run = max(run, other);
                    }
                    const float at_key = max(ch[h3].peak, run);
                    float before = simd_shuffle_up(run, ushort(1));
                    before = lane == 0 ? ch[h3].peak : max(ch[h3].peak, before);
                    kp3[h3] = metal::fast::exp(before - at_key);
                    wt[h3] = metal::fast::exp(logit - at_key);
                    ch[h3].peak = simd_shuffle(at_key, ushort(31));
                }
            }
            es[lane] = float4(wt[0], wt[1], wt[2], 0.0f);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const device uint2* vrow = (const device uint2*)(
                vg + size_t(base) * size_t(vc_strides[2]) + 4 * lane);
            if (all_fast) {
                for (int t0 = 0; t0 < 32; t0 += 4) {
                    uint2 vb[4];
                    for (int u = 0; u < 4; ++u) {
                        vb[u] = vrow[size_t(t0 + u) * vstep2];
                    }
                    for (int u = 0; u < 4; ++u) {
                        const float4 w4 = es[t0 + u];
                        const float vv[4] = {
                            as_type<float>(vb[u].x << 16), as_type<float>(vb[u].x & 0xffff0000u),
                            as_type<float>(vb[u].y << 16), as_type<float>(vb[u].y & 0xffff0000u)};
                        ch[0].mass = ch[0].mass + w4.x;
                        ch[1].mass = ch[1].mass + w4.y;
                        ch[2].mass = ch[2].mass + w4.z;
                        for (int j = 0; j < 4; j++) {
                            #pragma clang fp contract(off)
                            const float wv0 = w4.x * vv[j];
                            const float wv1 = w4.y * vv[j];
                            const float wv2 = w4.z * vv[j];
                            ch[0].acc[j] = ch[0].acc[j] + wv0;
                            ch[1].acc[j] = ch[1].acc[j] + wv1;
                            ch[2].acc[j] = ch[2].acc[j] + wv2;
                        }
                    }
                }
            } else {
                for (int t = 0; t < 32; ++t) {
                    const uint2 vb = vrow[size_t(t) * vstep2];
                    const float vv[4] = {
                        as_type<float>(vb.x << 16), as_type<float>(vb.x & 0xffff0000u),
                        as_type<float>(vb.y << 16), as_type<float>(vb.y & 0xffff0000u)};
                    const float4 w4 = es[t];
                    const float w3[3] = {w4.x, w4.y, w4.z};
                    for (int h3 = 0; h3 < 3; ++h3) {
                        const float keep = simd_shuffle(kp3[h3], ushort(t));
                        const float w = w3[h3];
                        for (int j = 0; j < 4; j++) {
                            ch[h3].acc[j] = ch[h3].acc[j] * keep + w * vv[j];
                        }
                        ch[h3].mass = ch[h3].mass * keep + w;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        p_from += tiles * 32 * span;
        for (int h3 = 0; h3 < 3; ++h3) {                      // q again for the per-key loop below
            const float4 qq = qr4[h3 * 32 + lane];
            qs[h3][0] = qq.x; qs[h3][1] = qq.y; qs[h3][2] = qq.z; qs[h3][3] = qq.w;
        }
    }

    // Consume the remainder in chronological order, stopping at this query.
    for (int p = p_from; p <= P; p += span) {
        const device T* kp = kg + size_t(p) * kc_strides[2];
        const device T* vp = vg + size_t(p) * vc_strides[2];
        // APPENDED_ROWS
        float kv[4], vv[4];
        for (int j = 0; j < 4; ++j) {
            kv[j] = static_cast<float>(kp[4 * lane + j]);
            vv[j] = static_cast<float>(vp[4 * lane + j]);
        }
        const int dist = P - p;
        for (int h3 = 0; h3 < 3; ++h3) {
            const float bias = dist < 1024 ? static_cast<float>(pr[h3 * 1024 + dist]) : 0.0f;
            chain_take(ch[h3], key_logit(qs[h3], kv, bias), vv);
        }
    }

    // Multirow stock attention appends the sink at the end of the entire bank.
    if (!SINK_FIRST && keys % BLOCKS == rho) {
        float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int h3 = 0; h3 < 3; ++h3) {
            const float bias = static_cast<float>(
                sink[r * sink_strides[1] + (3 * g + h3) * sink_strides[2]]);
            chain_take(ch[h3], key_logit(qs[h3], zero, bias), zero);
        }
    }
    for (int h3 = 0; h3 < 3; ++h3) {
        const size_t c = (size_t(r) * 24 + size_t(3 * g + h3)) * BLOCKS + size_t(rho);
        for (int j = 0; j < 4; ++j) {
            // Two-pass partials are stored in the query type, as MLX stores them.
            po[c * 128 + 4 * lane + j] = static_cast<T>(ch[h3].acc[j]);
        }
        if (lane == 0) {
            pm[c] = ch[h3].peak;
            ps[c] = ch[h3].mass;
        }
    }
