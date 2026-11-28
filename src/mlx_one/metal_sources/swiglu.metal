// Author: Susant Achary
// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

    const uint index = thread_position_in_grid.x;
    const float gate_value = float(gate[index]);
    const float reciprocal = 1.0f / (1.0f + metal::exp(metal::abs(gate_value)));
    const float sigmoid = gate_value < 0.0f ? reciprocal : 1.0f - reciprocal;
    output[index] = (gate_value * sigmoid) * value[index];
