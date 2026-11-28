// Copyright © 2023 Apple Inc. (FP4/E8M0 representation reference, MIT)
// New MLX-One projection/fusion implementation: Apache-2.0.
// See models/UPSTREAM_LICENSE and docs/acceleration/gptoss20b/upstream_references.md.
// Native Metal implementation by MLX-One
// Begin shader source

        uint i = thread_position_in_grid.x;
        if (i < COUNT) out[i] = gptoss_activation(gate[i], up[i]);
