// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
// MLX-One host module: models/language/inkling2/attention.py
// Begin shader source

            const uint i = thread_position_in_grid.x;
            const int keys = geometry[0], first = geometry[1], window = geometry[2];
            const int heads = profiles_shape[2], rows = profiles_shape[1];
            const int extent = profiles_shape[3];
            const int col = i % (keys + 1);
            uint plane = i / (keys + 1);
            const int row = plane % rows;
            plane /= rows;
            const int head = plane % heads, batch = plane / heads;
            const uint p = (batch * rows + row) * heads + head;
            const int distance = first + row - col;
            if (col == keys) bias[i] = sink[p];
            else if (distance < 0 || (window && distance >= window)) bias[i] = T(-INFINITY);
            else if (distance >= extent) bias[i] = T(0);
            else bias[i] = profiles[p * extent + distance];
