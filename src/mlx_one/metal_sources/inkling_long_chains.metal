// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad,
// models/inkling/fast/{attend,prefill_attend}.py. See MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

// A softmax chain of one query head, the part this lane holds: the largest logit taken so far
// (peak), the sum of exp(logit - peak) over the chain's keys (mass), and the matching weighted
// sum of value rows over this lane's four dimensions (acc). Every operation below is the one the
// bits require (MLX's vector attention rounds in exactly this order), so a chain ends with the
// same peak, mass and acc as MLX's.
struct Chain {
    float peak;
    float mass;
    float acc[4];
};

inline void chain_open(thread Chain& c) {
    c.peak = -FLT_MAX;
    c.mass = 0;
    for (int d = 0; d < 4; ++d) {
        c.acc[d] = 0;
    }
}

// One key's logit: this lane's four query-key products added in dimension order, the
// simdgroup's total over its lanes, then the relative bias.
inline float key_logit(thread const float (&qv)[4], thread const float (&kv)[4], float bias) {
    float lanepart = 0;
    for (int d = 0; d < 4; d++) {
        lanepart += qv[d] * kv[d];
    }
    return simd_sum(lanepart) + bias;
}

// Take one key into a chain: the key's weight at the new peak, and the share of the held sums
// that survives the move to it.
inline void chain_take(thread Chain& c, float logit, thread const float (&vv)[4]) {
    const float peak = max(c.peak, logit);
    const float w = metal::fast::exp(logit - peak);
    const float keep = metal::fast::exp(c.peak - peak);
    for (int d = 0; d < 4; d++) {
        c.acc[d] = c.acc[d] * keep + w * vv[d];
    }
    c.mass = c.mass * keep + w;
    c.peak = peak;
}
