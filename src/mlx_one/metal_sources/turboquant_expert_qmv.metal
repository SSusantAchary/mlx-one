// MLX-One host module: quantization/turboquant/kernels/tq3_expert_qmv.py
// Begin shader source

    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32u;
    uint row = threadgroup_position_in_grid.x * 8u + tid/32u;
    uint route = threadgroup_position_in_grid.y;
    uint O = packed_shape[1], P = packed_shape[2];
    uint K = x_shape[1], G = scales_shape[2];
    if (row >= O) return;
    uint expert = indices[route];
    uint xb = XROW * K;
    float acc = 0;
    for (uint wi = lane; wi < P; wi += 32u) {
        uint word = packed[(expert * O + row) * P + wi];
        #pragma unroll
        for (uint j = 0; j < 10u; ++j) {
            uint col = wi * 10u + j;
            if (col < K) {
                float w = float(codebook[(word >> (3u*j)) & 7u])
                    * float(scales[(expert * O + row)*G + col/64u]);
                acc = fma(w, float(x[xb + col]), acc);
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0u) out[route * O + row] = T(acc);
