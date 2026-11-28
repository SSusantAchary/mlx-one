// Copyright 2026 Youssof Altoukhi
// SPDX-License-Identifier: Apache-2.0
// Adapted from MTPLX v2.13.0, commit bebe21b5ba36af3688335ab66d365509fd042fad.
// Source: mtplx/models/inkling/short_conv.py. See MTPLX_NOTICE.txt.
// MLX-One host module: models/language/inkling2/short_conv.py
// Begin shader source

            const uint c = thread_position_in_grid.x;
            const uint width = uint(x_shape[2]);
            if (c >= width) return;
            const int row = int(thread_position_in_grid.y);

            float h[4];
            for (int j = 0; j < 4; ++j) {
                const int source = row + j - 3;
                h[j] = source < 0
                    ? prev[(source + 3) * prev_strides[1] + c * prev_strides[2]]
                    : float(x[source * x_strides[1] + c * x_strides[2]]);
            }
            float acc = 0.0f;
            // Match the static depthwise convolution and dynamic tap orders.
            if constexpr (DYNAMIC) {
                const uint head = c / (width / 8);
                for (uint lag = 0; lag < 4; ++lag) {
                    const float offset = float(change[row * change_strides[1]
                        + (head * 4 + lag) * change_strides[2]]);
                    const float coefficient = weight[lag * weight_strides[0]
                                                   + c * weight_strides[1]] + offset;
                    const float product = h[3 - lag] * coefficient;
                    acc = acc + product;
                }
            } else {
                for (uint j = 0; j < 4; ++j)
                    acc += h[j] * weight[j * weight_strides[0] + c * weight_strides[1]];
            }
            float rounded = float(T((h[3] + acc) / as_type<float>(uint(DIVISOR))));
            if constexpr (ADD)
                rounded += float(add[row * add_strides[1] + c * add_strides[2]]);
            out[row * width + c] = U(rounded);
            if (row + 1 == int(x_shape[1])) {
                for (uint j = 0; j < 3; ++j)
                    tail[j * width + c] = h[j + 1];
            }
