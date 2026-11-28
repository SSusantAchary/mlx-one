// Copyright © 2023 Apple Inc. (FP4/E8M0 representation reference, MIT)
// New MLX-One projection/fusion implementation: Apache-2.0.
// See models/UPSTREAM_LICENSE and docs/acceleration/gptoss20b/upstream_references.md.
// Native Metal implementation by MLX-One
// Begin shader source

        uint d = thread_position_in_grid.x;
        if (d >= D) return;
        T value = T(0);
        #pragma unroll
        for (uint e = 0; e < 4; ++e) {
            T product = T(float(values[e * D + d]) * float(scores[e]));
            value = T(float(value) + float(product));
        }
        out[d] = value;
