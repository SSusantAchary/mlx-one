"""Sparse TQ3 expert projections, without dense expert materialization."""

import mlx.core as mx

from .dispatch import expert_kernel, record
from .kernels.tq3_expert_qmv import expert_qmv
from .kernels.tq3_gather_qmm import tq3_gather_qmm
from .linear import TQ3Linear, rotate_input


class TQ3ExpertLinear(TQ3Linear):
    def __call__(self, x, indices, sorted_indices=False):
        if x.shape[-1] != self.input_dims or x.shape[-2] != 1:
            raise ValueError("TQ3 experts require one vector per shared token or routing")
        shape = list(indices.shape) + [1, self.output_dims]
        routes = indices.size
        flat = x.reshape(-1, self.input_dims)
        if routes % flat.shape[0]:
            raise ValueError("TQ3 input rows do not match expert indices")
        repeats = routes // flat.shape[0]
        flat = rotate_input(flat, self.signs)
        if flat.dtype == mx.bfloat16:
            flat = flat.astype(mx.float32)
        idx = indices.reshape(-1)
        kernel = expert_kernel(routes, self.output_dims)
        record(kernel + ("_shared" if repeats > 1 else "_multi"))
        if kernel == "gather_qmv":
            y = expert_qmv(self.weight, self.scales, self.codebook, flat, idx, repeats)
        else:
            # Only activations are expanded; no expert bank ever becomes dense.
            rows = mx.repeat(flat, repeats, axis=0) if repeats > 1 else flat
            if sorted_indices:
                y = tq3_gather_qmm(self.weight, self.scales, self.codebook, rows, idx, 3, 64)
            else:
                order = mx.argsort(idx)
                inverse = mx.argsort(order)
                y = tq3_gather_qmm(
                    self.weight, self.scales, self.codebook, rows[order], idx[order], 3, 64
                )[inverse]
        y = y.reshape(shape)
        return y + self.bias[indices][..., None, :] if "bias" in self else y
