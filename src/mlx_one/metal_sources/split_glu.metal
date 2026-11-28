// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    const uint index = thread_position_in_grid.x;
    const uint width = input_shape[2] / 2;
    if (MASKED && !valid[index / width]) {
        output[index] = 0.0f;
        return;
    }
    const uint left = (index / width) * (2 * width) + index % width;
    const float gate = input[left + width];
    const float p = 1.0f / (1.0f + metal::exp(metal::abs(gate)));
    output[index] = input[left] * (gate < 0.0f ? p : 1.0f - p);
