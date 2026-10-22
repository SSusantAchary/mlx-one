# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.5 math adopted from qwen3_next.py and qwen3_5.py.

The MLX-One adapters retain raw offset norms, fixed convolution/decay precision,
multimodal positions, and the existing [B,H,K,V] state owner. Source recurrent
helpers consume [B,H,V,K]; transposes occur only at this ownership boundary.
See models/UPSTREAM_LICENSE and reference-source-provenance.md.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_one.models.shared.reference_math import precise_swiglu, swiglu


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


def gated_delta_forward(module, inputs, cache=None, mask=None):
    from mlx_one.models.vision_language.qwen3_5.reference_delta import (
        gated_delta_update,
        normalize_qk,
    )

    B, S, _ = inputs.shape
    qkv = module.in_proj_qkv(inputs)
    z = module.in_proj_z(inputs).reshape(B, S, module.num_v_heads, module.head_v_dim)
    b = module.in_proj_b(inputs)
    a = module.in_proj_a(inputs)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_out = nn.silu(module.conv1d(qkv, cache))
    q, k, v = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [module.key_dim, 2 * module.key_dim], -1),
            [module.num_k_heads, module.num_k_heads, module.num_v_heads],
            [module.head_k_dim, module.head_k_dim, module.head_v_dim],
            strict=True,
        )
    ]
    state = None if cache is None else cache.recurrent_state
    if state is not None:
        state = mx.contiguous(state.swapaxes(-1, -2))
    q, k = normalize_qk(q, k, inv_scale=module.head_k_dim**-0.5, eps=1e-6)
    out, state = gated_delta_update(
        q,
        k,
        v,
        a,
        b,
        module.A_log,
        module.dt_bias,
        state,
        mask,
        use_kernel=not module.training,
    )
    if cache is not None:
        cache.recurrent_state = mx.contiguous(state.swapaxes(-1, -2))
    out = module.norm(out, z)
    return module.out_proj(out.reshape(B, S, -1))
