// SPDX-License-Identifier: Apache-2.0
// Native Metal implementation by MLX-One
// Begin shader source

            const uint column = thread_position_in_grid.x;
            const uint slot = thread_position_in_grid.y;
            const uint width = expert_rows_shape[1] / 8;
            if (column >= width || slot >= expert_rows_shape[0]) return;
            const device uint4* input = reinterpret_cast<const device uint4*>(expert_rows);
            device uint4* output = reinterpret_cast<device uint4*>(token_rows);
            output[size_t(order[slot]) * width + column] = input[size_t(slot) * width + column];
