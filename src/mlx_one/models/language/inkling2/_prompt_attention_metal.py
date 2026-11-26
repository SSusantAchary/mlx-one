# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
# See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
# Source: mtplx/models/inkling/fast/prompt_flash.py.
# Reduction order follows MLX v0.32.2 Steel SDPA. See mlxboost/native/NOTICE.txt.
"""Simdgroup attention for prompt chunks on Apple M4."""

HEADER = r"""
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
"""

SOURCE = r"""
    using Frag = mlx::steel::BaseMMAFrag<float, 8, 8>;
    typedef vec<T, 8> tpiece;
    typedef vec<float, 8> fpiece;
    typedef vec<T, 2> tfrag;
    typedef vec<float, 2> ffrag;
    constexpr int D = 128;
    constexpr int DP = D / 16;                                  // pieces along a head
    constexpr int TG_ROWS = (TG_M / G) * 16;
    const float scale_log2 = as_type<float>(uint(SCALE)) * 1.4426950408889634f;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint tid = thread_position_in_threadgroup.x;
    const int sg = int(tid >> 5);
    const short2 here = Frag::get_coord(ushort(tid & 31u));
    const short fr = here.y;
    const short fc = here.x;
    const int kvh = int(threadgroup_position_in_grid.y);
    const int rows = int(params[0]);
    const int total = int(params[1]);
    const int hist = total - rows;
    const int tg_row0 = int(threadgroup_position_in_grid.x) * TG_ROWS;
    const int tg_row_last = min(tg_row0 + TG_ROWS, rows) - 1;
    const int blk_row0 = tg_row0 + (sg / G) * 16;
    const int head = kvh * G + sg % G;

    int qrow[2];
    int pos[2];
    bool live[2];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        qrow[rp] = blk_row0 + int(fr) + 8 * rp;
        live[rp] = qrow[rp] < rows;
        pos[rp] = hist + qrow[rp];
    }

    const uint q_dim = uint(q_strides[3]);
    const uint k_dim = uint(k_strides[3]);
    const uint v_dim = uint(v_strides[3]);
    const uint q_step =
        (q_dim == 1u && q_strides[2] % 2 == 0 && q_strides[1] % 2 == 0) ? 0u : q_dim;
    const uint v_step =
        (v_dim == 1u && v_strides[2] % 2 == 0 && v_strides[1] % 2 == 0) ? 0u : v_dim;

    // Fragment element 4 rp + 2 h + i covers dimension 16 p + 8 h + fc + i.
    tpiece qp[DP];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        const device T* src = q + size_t(head) * size_t(q_strides[1]) +
            size_t(live[rp] ? qrow[rp] : 0) * size_t(q_strides[2]) + size_t(fc) * q_dim;
        INK_EVERY for (short p = 0; p < DP; ++p) {
            INK_EVERY for (short h = 0; h < 2; ++h) {
                const tfrag x = ink_two(src + uint(16 * p + 8 * h) * q_dim, q_step);
                qp[p][rp * 4 + 2 * h] = live[rp] ? x[0] : T(0);
                qp[p][rp * 4 + 2 * h + 1] = live[rp] ? x[1] : T(0);
            }
        }
    }
    static_assert(DP == 8, "a head of 128 is eight pieces");
    const fpiece zero8 = fpiece(0.0f);
    fpiece acc[DP] = {zero8, zero8, zero8, zero8, zero8, zero8, zero8, zero8};

    float run_max[2];
    float run_sum[2] = {0.0f, 0.0f};
    float sink_score[2];
    const device T* prof_row[2];
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        const int r = live[rp] ? qrow[rp] : 0;
        run_max[rp] = -FLT_MAX;
        sink_score[rp] = float(sink[size_t(r) * size_t(sink_strides[1]) +
                                    size_t(head) * size_t(sink_strides[2])]) * LOG2E;
        prof_row[rp] = prof + size_t(r) * size_t(prof_strides[1]) +
            size_t(head) * size_t(prof_strides[2]);
    }
    const uint prof_step = uint(prof_strides[3]);

    // Keys are transposed for QK. Values read adjacent dimensions at fc.
    const device T* k_lane = k + (size_t(kvh) * size_t(k_strides[1]) + size_t(fr) * size_t(k_dim));
    const device T* v_lane = v + (size_t(kvh) * size_t(v_strides[1]) + size_t(fc) * size_t(v_dim));
    const uint k_row = uint(k_strides[2]);
    const uint v_row = uint(v_strides[2]);

    const int pos_lo = hist + tg_row0;
    const int pos_hi = hist + tg_row_last;
    const int t_end = pos_hi / 16 + 1;
    const int sink_tile = total / 16;
    int t_begin = 0;
    int t_near = 0;
    if (WINDOW > 0) {
        t_begin = max(0, pos_lo - (WINDOW - 1)) / 16;
        t_near = t_begin;
    } else {
        const int clear = pos_lo - EXTENT + 1;
        t_near = clear > 0 ? min(clear / 16, t_end) : 0;
    }

    // Steel accumulates 16 keys at a time and sees the zero-valued sink as
    // column total. Whole causal-mask tiles between live keys and the sink
    // have zero mass and can be skipped.
    const int stop_tile = t_end + int(t_end <= sink_tile);
    for (int i = t_begin; i < stop_tile; ++i) {
        const int t = i < t_end ? i : sink_tile;
        if (TGBAR) {
            threadgroup_barrier(mem_flags::mem_none);
        }
        const int base = t * 16;
        const bool near = t >= t_near;
        // Score fragment keys are base + 8 cb + fc + i. Value rows are base + 8 kb + fr.
        int kpos[2][2];
        int vpos[2];
        INK_EVERY for (short b = 0; b < 2; ++b) {
            INK_EVERY for (short i = 0; i < 2; ++i) {
                kpos[b][i] = base + 8 * b + int(fc) + i;
            }
            vpos[b] = base + 8 * b + int(fr);
        }

        fpiece s = fpiece(0.0f);
        INK_EVERY for (short p = 0; p < DP; ++p) {
            tfrag kf[2][2];  // [kb][cb]: dims 16 p + 8 kb + fr of keys 8 cb + fc + i
            INK_EVERY for (short kb = 0; kb < 2; ++kb) {
                const device T* src = k_lane + uint(16 * p + 8 * kb) * k_dim;
                INK_EVERY for (short cb = 0; cb < 2; ++cb) {
                    const int k0 = kpos[cb][0], k1 = kpos[cb][1];
                    kf[kb][cb] = tfrag(k0 < total ? src[uint(k0) * k_row] : T(0),
                                       k1 < total ? src[uint(k1) * k_row] : T(0));
                }
            }
            INK_EVERY for (short rb = 0; rb < 2; ++rb) {
                INK_EVERY for (short cb = 0; cb < 2; ++cb) {
                    ffrag d = ffrag(s[4 * rb + 2 * cb], s[4 * rb + 2 * cb + 1]);
                    INK_EVERY for (short kb = 0; kb < 2; ++kb) {
                        tfrag a = tfrag(qp[p][4 * rb + 2 * kb], qp[p][4 * rb + 2 * kb + 1]);
                        Frag::template mma<T, T, float>(d, a, kf[kb][cb], d);
                    }
                    s[4 * rb + 2 * cb] = d[0];
                    s[4 * rb + 2 * cb + 1] = d[1];
                }
            }
        }

        // Logits in base-2 units, masked and biased where a row is near.
        INK_EVERY for (short e = 0; e < 8; ++e) {
            const short rp = e >> 2;
            const short c = e & 3;
            float x = s[e] * scale_log2;
            if (near) {
                const int key = kpos[c >> 1][c & 1];
                const int d = pos[rp] - key;
                if (key == total) {
                    x = sink_score[rp];
                } else if (key > total || d < 0 || (WINDOW > 0 && d >= WINDOW)) {
                    x = -INFINITY;
                } else if (d < EXTENT) {
                    x = fma(float(prof_row[rp][uint(d) * prof_step]), LOG2E, x);
                }
            }
            s[e] = x;
        }

        // A row's fragment columns occupy lanes xor 1 and xor 8 apart.
        float corr[2];
        INK_EVERY for (short rp = 0; rp < 2; ++rp) {
            float m = -INFINITY;
            INK_EVERY for (short col = 0; col < 4; ++col) {
                m = max(m, s[rp * 4 + col]);
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            const float nm = max(run_max[rp], m);
            corr[rp] = fast::exp2(run_max[rp] - nm);
            float sum = 0.0f;
            INK_EVERY for (short col = 0; col < 4; ++col) {
                s[rp * 4 + col] = fast::exp2(s[rp * 4 + col] - nm);
            }
            INK_EVERY for (short col = 0; col < 4; col += 2) {
                float pair = s[rp * 4 + col] + s[rp * 4 + col + 1];
                pair += simd_shuffle_xor(pair, ushort(1));
                pair += simd_shuffle_xor(pair, ushort(8));
                sum += pair;
            }
            run_sum[rp] = run_sum[rp] * corr[rp] + sum;
            run_max[rp] = nm;
        }
        if (simd_any(corr[0] != 1.0f || corr[1] != 1.0f)) {
            INK_EVERY for (short p = 0; p < DP; ++p) {
                INK_EVERY for (short e = 0; e < 8; ++e) {
                    acc[p][e] *= corr[e >> 2];
                }
            }
        }

        INK_EVERY for (short g = 0; g < DP; ++g) {
            tfrag vf[2][2];
            INK_EVERY for (short kb = 0; kb < 2; ++kb) {
                INK_EVERY for (short cb = 0; cb < 2; ++cb) {
                    vf[kb][cb] = vpos[kb] < total
                        ? ink_two(v_lane + uint(vpos[kb]) * v_row +
                                  uint(16 * g + 8 * cb) * v_dim, v_step)
                        : tfrag(T(0));
                }
            }
            INK_EVERY for (short rb = 0; rb < 2; ++rb) {
                INK_EVERY for (short cb = 0; cb < 2; ++cb) {
                    ffrag d = ffrag(acc[g][4 * rb + 2 * cb], acc[g][4 * rb + 2 * cb + 1]);
                    INK_EVERY for (short kb = 0; kb < 2; ++kb) {
                        ffrag a = ffrag(s[4 * rb + 2 * kb], s[4 * rb + 2 * kb + 1]);
                        Frag::template mma<float, T, float>(d, a, vf[kb][cb], d);
                    }
                    acc[g][4 * rb + 2 * cb] = d[0];
                    acc[g][4 * rb + 2 * cb + 1] = d[1];
                }
            }
        }
    }

    // Store: piece p, element 4 rp + 2 h + i is dimension 16 p + 8 h + fc + i of row fr + 8 rp.
    INK_EVERY for (short rp = 0; rp < 2; ++rp) {
        if (!live[rp]) {
            continue;
        }
        device T* o = out + (size_t(qrow[rp]) * size_t(HQ) + size_t(head)) * size_t(D) + size_t(fc);
        INK_EVERY for (short p = 0; p < DP; ++p) {
            INK_EVERY for (short h = 0; h < 2; ++h) {
                vec<T, 2> y;
                INK_EVERY for (short i = 0; i < 2; ++i) {
                    y[i] = T(acc[p][rp * 4 + 2 * h + i] / run_sum[rp]);
                }
                ink_put_two(o + 16 * p + 8 * h, y);
            }
        }
    }
"""
