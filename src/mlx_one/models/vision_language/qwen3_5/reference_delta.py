# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from the pinned local MLX-LM gated_delta.py; see models/UPSTREAM_LICENSE.

import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_one.metal_source import metal_source

# For the shapes it supports, the packed kernel is bitwise-identical by
# construction to an explicit-tree comparator kernel that the tests pin it
# against (see _make_gated_delta_packed_kernel). Every other shape (masks,
# vector gates, Dk != 128) and the MLX_GDN_PACKED=0 kill-switch use the
# original simd_sum kernels, unchanged.
_ENABLE_GDN_PACKED = os.environ.get("MLX_GDN_PACKED", "1") != "0"


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a.astype(mx.float32) + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_lower_bound_g(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound
        * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias))
    )


def normalize_qk(
    q: mx.array, k: mx.array, *, inv_scale: float, eps: float
) -> tuple[mx.array, mx.array]:
    """L2-normalize q/k and fold the ``head_dim**-0.5`` readout scale into q.

    ``eps`` is the reference l2norm eps, which is added to sum(x^2).
    ``mx.fast.rms_norm`` adds it to mean(x^2), so scale it by ``inv_scale**2``.
    """
    rms_eps = eps * inv_scale**2
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, rms_eps)
    k = inv_scale * mx.fast.rms_norm(k, None, rms_eps)
    return q, k


def _make_gated_delta_kernel(has_mask=False, vectorized=False):
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # Configure g indexing based on whether gating is vectorized
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = metal_source(
        "qwen35_delta_vector.metal",
        g_comment=g_comment,
        g_setup=g_setup,
        mask_source=mask_source,
        g_access=g_access,
        g_advance=g_advance,
    )
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step{suffix}",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_kernel_xtree():
    """Scalar-gate, unmasked kernel with an explicitly-written reduction.

    This is the unpacked comparator for the packed kernel: it replaces the
    two simd_sum calls with the ascending butterfly (shuffle_xor 1,2,4,8,16)
    written out in source, so the reduction order is a contract of this file
    rather than of the simd_sum lowering. On current Apple GPUs this is the
    same tree simd_sum lowers to, so it is bit-identical to the generic
    kernel there. Only shapes eligible for the packed kernel ever run it;
    masked, vector-gate and Dk != 128 paths keep the original kernels.
    """
    if not mx.metal.is_available():
        return None

    source = metal_source("qwen35_delta_tree.metal")
    return mx.fast.metal_kernel(
        name="gated_delta_step_xtree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


def _make_gated_delta_packed_kernel():
    """Make the scalar-gate Dk=128 prefill specialization.

    The generic kernel assigns one 32-lane SIMD-group to each value row. For
    Dk=128 that leaves every lane with only four state elements and performs
    two full-SIMD reductions per row and token. This kernel instead packs
    eight independent value rows into a SIMD-group: four lanes own each row
    and each lane keeps 32 contiguous state elements in registers.

    The reduction reproduces the unpacked comparator
    (_make_gated_delta_kernel_xtree) bitwise BY CONSTRUCTION: both use the
    same explicitly-written ascending butterfly (shuffle_xor 1,2,4,8,16)
    rather than relying on how simd_sum lowers. The butterfly's first three
    levels combine partials that live in a single packed lane (IEEE addition
    is commutative, so the local pairwise tree is bit-identical), and the
    last two levels map onto shuffle_xor(1) and shuffle_xor(2) within the
    four-lane row group. Each 4-element partial keeps the comparator's
    sequential order, so y and the state are bit-identical to it on any
    device. On current Apple GPUs the explicit tree is also bit-identical
    to the simd_sum-based generic kernel.
    """
    if not mx.metal.is_available():
        return None

    source = metal_source("qwen35_delta_packed.metal")
    return mx.fast.metal_kernel(
        name="gated_delta_step_packed_btree",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=source,
    )


_gated_delta_kernel = _make_gated_delta_kernel(has_mask=False, vectorized=False)
_gated_delta_kernel_masked = _make_gated_delta_kernel(has_mask=True, vectorized=False)
_gated_delta_kernel_vec = _make_gated_delta_kernel(has_mask=False, vectorized=True)
_gated_delta_kernel_vec_masked = _make_gated_delta_kernel(has_mask=True, vectorized=True)
_gated_delta_kernel_xtree = _make_gated_delta_kernel_xtree()
_gated_delta_kernel_packed = _make_gated_delta_packed_kernel()


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """

    # Decay
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # [B, H, Dv]
    delta = (v - kv_mem) * beta[..., None]  # [B, H, Dv]
    state = state + k[..., None, :] * delta[..., None]
    # Output projection along key dim with q
    y = (state * q[..., None, :]).sum(axis=-1)  # [B, H, Dv]

    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
    return y.astype(q.dtype), state


def _gated_delta_kernel_impl(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
    *,
    allow_packed: bool,
) -> tuple[mx.array, mx.array]:
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype

    # The packed kernel gives each lane Dk/4 state elements and packs 32/4
    # value rows into a SIMD-group, so it needs Dk == 128 and Dv divisible by
    # 8. It is otherwise generic in B, Hk, Hv and the input element type.
    # Vector gating and padding masks keep the original kernels untouched.
    packed_eligible = (
        mask is None
        and g.ndim == 3
        and Dk == 128
        and Dv % 8 == 0
        and g.dtype == mx.float32
        and state.dtype == mx.float32
    )

    if packed_eligible and allow_packed and _ENABLE_GDN_PACKED:
        kernel = _gated_delta_kernel_packed
        inputs = [q, k, v, g, beta, state, T]
        grid = (32, Dv // 8, B * Hv)
        threadgroup = (32, 2, 1)
    elif g.ndim == 4:
        kernel = _gated_delta_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_vec_masked
            inputs.append(mask)
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)
    else:
        kernel = _gated_delta_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_kernel_masked
            inputs.append(mask)
        grid = (32, Dv, B * Hv)
        threadgroup = (32, 4, 1)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_kernel_xtree(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Explicit-tree comparator for the packed kernel (test use).

    Runs the unpacked layout with the same explicitly-written reduction tree
    the packed kernel uses, defining its bitwise contract independently of
    the simd_sum lowering.
    """
    assert mask is None
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    return _gated_delta_kernel_xtree(
        inputs=[q, k, v, g, beta, state, T],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )


def gated_delta_kernel_unpacked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Run the original one-value-row-per-SIMD-group kernel."""
    return _gated_delta_kernel_impl(q, k, v, g, beta, state, mask, allow_packed=False)


def gated_delta_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    return _gated_delta_kernel_impl(q, k, v, g, beta, state, mask, allow_packed=True)


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    ys = []
    for t in range(T):
        y, state = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state,
            None if mask is None else mask[:, t],
        )
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return y, state


def gated_delta_update(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    *,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    allow_neg_eigval: bool = False,
) -> tuple[mx.array, mx.array]:
    """Gated delta rule recurrence.

    Contract: callers normalize q/k with ``normalize_qk``, which folds the
    ``Dk**-0.5`` readout scale into q. The helper applies no scale of its own.

    Set ``allow_neg_eigval`` to put beta in [0, 2] instead of [0, 1].
    """
    beta = mx.sigmoid(b)
    if allow_neg_eigval:
        beta = beta * 2.0
    if lower_bound is None:
        g = compute_g(A_log, a, dt_bias)
    else:
        g = compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)
