// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    #pragma clang fp contract(off)
    const uint i = thread_position_in_grid.x;
    const uint bins = spectrum_shape[2] / 2;
    const uint real_index = (i / bins) * (2 * bins) + i % bins;
    const float re = spectrum[real_index];
    const float im = spectrum[real_index + bins];
    output[i] = metal::precise::sqrt(re * re + im * im);
