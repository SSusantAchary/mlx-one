// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    #pragma clang fp contract(off)
    const uint index = thread_position_in_grid.x;
    const float x = input[index];
    float result;
    if (FORM == 0) {
        // Abramowitz and Stegun 7.1.26, absolute erf error below 1.5e-7
        // before FP32 rounding.
        const float a = metal::abs(x) * 0.7071067811865475f;
        const float t = 1.0f / (1.0f + 0.3275911f * a);
        const float p = (((((1.061405429f * t - 1.453152027f) * t
                          + 1.421413741f) * t - 0.284496736f) * t
                          + 0.254829592f) * t) * metal::exp(-a * a);
        result = x * (x < 0.0f ? 0.5f * p : 1.0f - 0.5f * p);
    } else if (FORM == 1) {
        result = 0.5f * x * (1.0f + metal::precise::tanh(
            0.7978845608028654f * (x + 0.044715f * ((x * x) * x))));
    } else {
        const float z = 1.702f * x;
        const float p = 1.0f / (1.0f + metal::exp(metal::abs(z)));
        result = x * (z < 0.0f ? p : 1.0f - p);
    }
    output[index] = GATED ? result * value[index] : result;
