# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
# See MTPLX_NOTICE.txt and MTPLX_LICENSE.txt.
# Sources: mtplx/models/inkling/fast/{affine_lane,qproduct}.py.
"""One-row affine arithmetic shared across a draft verification panel."""

HEADER = """
namespace affine_lane {

// 16^-j: the scale of the input that meets code j of a 16-bit half (exact powers of two).
constant float kInvPlace[4] = {1.0f, 0.0625f, 0.00390625f, 0.000244140625f};

// The four inputs that meet one 16-bit half of 4-bit codes: written scaled, so each meets its code
// where it sits; returned added together in their own type (bf16 for bf16 inputs).
template <typename P>
inline auto meet_half(P v, thread float* out) {
    out[0] = static_cast<float>(v[0]);
    out[1] = static_cast<float>(v[1]) * kInvPlace[1];
    out[2] = static_cast<float>(v[2]) * kInvPlace[2];
    out[3] = static_cast<float>(v[3]) * kInvPlace[3];
    return v[0] + v[1] + v[2] + v[3];
}

// The lane's SPAN inputs as the products take them (into out) and their float total (returned):
// at 4 bits a half's worth at a time, at 8 bits one input at a time. P: a device or thread pointer
// to bf16 or float inputs.
template <int SPAN, int BITS, typename P>
inline float take(P v, thread float* out) {
    float total = 0.0f;
    if (BITS == 4) {
        for (int h = 0; h < SPAN; h += 4) total = total + meet_half(v + h, out + h);
    } else {
        for (int j = 0; j < SPAN; ++j) {
            out[j] = static_cast<float>(v[j]);
            total = total + v[j];
        }
    }
    return total;
}

// The scaled inputs alone (the caller has their total) and the total alone.
template <int SPAN, int BITS, typename P>
inline void scaled(P v, thread float* out) {
    (void)take<SPAN, BITS>(v, out);
}

template <int SPAN, int BITS, typename P>
inline float total(P v) {
    float unused[SPAN];
    return take<SPAN, BITS>(v, unused);
}

// Code J of a 16-bit half left where it sits (so it carries 16^J), as the products take it.
template <int J>
inline float in_place(uint quad) {
    return static_cast<float>(int(quad & (0xFu << (4 * J))));
}

// One output row's SPAN codes under the lane's inputs, as the products take them: 4-bit codes a
// 16-bit half at a time (the low half of a word first), 8-bit codes as their bytes.
template <int SPAN, int BITS>
inline void unpack(const device uint32_t* words, thread float* out) {
    if (BITS == 4) {
        for (int h = 0; h < SPAN / 4; ++h) {
            const uint quad = (words[h >> 1] >> (16 * (h & 1))) & 0xFFFFu;
            out[4 * h] = in_place<0>(quad);
            out[4 * h + 1] = in_place<1>(quad);
            out[4 * h + 2] = in_place<2>(quad);
            out[4 * h + 3] = in_place<3>(quad);
        }
    } else {
        for (int j = 0; j < SPAN; ++j) out[j] = static_cast<float>(uchar(words[j >> 2] >> (8 * (j & 3))));
    }
}

// The block's term for one output row from the scaled inputs, their total and the row's codes: at 4
// bits each half's four products as one sum, in order, then into the dot; at 8 bits each product
// straight into the dot; then the group's scale and bias.
template <int SPAN, int BITS>
inline float term(const thread float* in, const thread float* code, float scale, float bias, float tot) {
    float dot = 0.0f;
    if (BITS == 4) {
        for (int h = 0; h < SPAN; h += 4)
            dot = dot + (in[h] * code[h] + in[h + 1] * code[h + 1] + in[h + 2] * code[h + 2] +
                         in[h + 3] * code[h + 3]);
    } else {
        for (int j = 0; j < SPAN; ++j) dot = dot + in[j] * code[j];
    }
    return scale * dot + tot * bias;
}

}  // namespace affine_lane

// A lane's SPAN inputs widened to float, the values a float32 copy of the row holds (the product of a
// 16-bit row whose reference takes it as float32: the head).
template <int SPAN, typename P>
inline void widen(P v, thread float* out) {
    for (int j = 0; j < SPAN; ++j) out[j] = static_cast<float>(v[j]);
}

// MLX's one-row affine product for output rows [row0, row0 + NROW) and W panel rows (row r at
// panel + r * panel_stride; bf16, float16 or float, as the reference product takes it), every lane of
// the calling simdgroup together, through affine_lane; scales and biases bf16 or float16 (S).
// result[r][ro] ends in every lane.
template <int W, int NROW, int K, int BITS, int GROUP, int VPT, typename T, typename S>
inline void qmv_rows(const device uint32_t* codes, const device S* scales, const device S* biases,
                     const device T* panel, int panel_stride, int row0,
                     uint lane, thread float (&result)[W][NROW]) {
    constexpr int BLOCK = 32 * VPT, NB = K / BLOCK, WPR = K * BITS / 32, GPR = K / GROUP;
    for (int r = 0; r < W; ++r)
        for (int ro = 0; ro < NROW; ++ro) result[r][ro] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const int col = b * BLOCK + int(lane) * VPT;
        float wf[NROW][VPT], sc[NROW], bi[NROW];
        for (int ro = 0; ro < NROW; ++ro) {
            const size_t row = size_t(row0 + ro);
            sc[ro] = static_cast<float>(scales[row * GPR + col / GROUP]);
            bi[ro] = static_cast<float>(biases[row * GPR + col / GROUP]);
            affine_lane::unpack<VPT, BITS>(codes + row * WPR + col * BITS / 32, wf[ro]);
        }
        for (int r = 0; r < W; ++r) {
            const device T* xs = panel + size_t(r) * panel_stride + col;
            float xt[VPT];
            const float tot = affine_lane::take<VPT, BITS>(xs, xt);
            for (int ro = 0; ro < NROW; ++ro)
                result[r][ro] += affine_lane::term<VPT, BITS>(xt, wf[ro], sc[ro], bi[ro], tot);
        }
    }
    for (int r = 0; r < W; ++r)
        for (int ro = 0; ro < NROW; ++ro) result[r][ro] = simd_sum(result[r][ro]);
}
"""
