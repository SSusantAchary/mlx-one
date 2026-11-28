// MLX-One host module: models/language/inkling2/verify.py
// Begin shader source

            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
            float result[W][1];
            qmv_rows<W, 1, K, BITS, GROUP, VPT>(
                codes, scales, biases, x, K, row, lane, result);
            if (lane == 0) {
                for (int r = 0; r < W; ++r) out[r * N + row] = T(result[r][0]);
            }
