// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
// Native Metal implementation by MLX-One
// Begin shader source

            const int col = thread_position_in_grid.x, row = thread_position_in_grid.y;
            const int head = thread_position_in_grid.z, rows = profiles_shape[1];
            const int count = keys_shape[2], extent = profiles_shape[3];
            if (col >= count) return;
            const int distance = count - rows + row - col;
            const bool visible = distance >= 0 && (!WINDOW || distance < WINDOW);
            T bias = T(0);
            if (!visible) bias = T(-INFINITY);
            else if (distance < extent) {
                const size_t offset = row * profiles_strides[1] + head * profiles_strides[2];
                bias = profiles[offset + distance * profiles_strides[3]];
            }
            mask[(head * rows + row) * size_t(count) + col] = bias;
