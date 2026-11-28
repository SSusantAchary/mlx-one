// Copyright © 2023 Apple Inc. (FP4/E8M0 representation reference, MIT)
// New MLX-One projection/fusion implementation: Apache-2.0.
// See models/UPSTREAM_LICENSE and docs/acceleration/gptoss20b/upstream_references.md.
// Native Metal implementation by MLX-One
// Begin shader source

// FP4/E8M0 conversions follow the formats documented in Apple's MIT-licensed
// mlx/backend/metal/kernels/fp4.h and fp8.h. Preserve signed zero and specials.
inline float gptoss_fp4(uint bits) {
    float value = float(as_type<half>(ushort((bits & 7u) << 9))) * 16384.0f;
    return (bits & 8u) ? -value : value;
}
inline float gptoss_scale(uchar bits) {
    uint value = bits == 0 ? 0x00400000u : bits == 255 ? 0x7fc00000u : uint(bits) << 23;
    return as_type<float>(value);
}
template <typename T> inline T gptoss_activation(T gate, T up) {
    T g = T(min(float(gate), 7.0f));
    T u = T(clamp(float(up), -7.0f, 7.0f));
    T z = T(float(g) * float(T(1.702f)));
    auto y = 1 / (1 + metal::exp(metal::abs(z)));
    T sigmoid = (z < 0) ? y : 1 - y;
    T swish = T(float(g) * float(sigmoid));
    T shifted = T(float(u) + float(T(1.0f)));
    return T(float(shifted) * float(swish));
}
