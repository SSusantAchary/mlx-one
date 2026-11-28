// Native Metal implementation by MLX-One
// Begin shader source

        uint tid = thread_position_in_threadgroup.x;
        uint lane = tid % 32u;
        uint row = threadgroup_position_in_grid.x * 8u + tid / 32u;
        uint O = packed_shape[0], P = packed_shape[1];
        uint K = x_shape[0], G = scales_shape[1];
        if (row >= O) return;
        float cb[8];
        #pragma unroll
        for (uint j = 0; j < 8u; ++j) cb[j] = float(codebook[j]);
        float acc = 0;
        for (uint wi = lane; wi < P; wi += 32u) {
            uint word = packed[row * P + wi];
            #pragma unroll
            for (uint j = 0; j < 10u; ++j) {
                uint col = wi * 10u + j;
                if (col < K) {
                    float w = cb[(word >> (3u*j)) & 7u]
                        * float(scales[row*G + col/64u]);
                    acc = fma(w, float(x[col]), acc);
                }
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) out[row] = T(acc);
