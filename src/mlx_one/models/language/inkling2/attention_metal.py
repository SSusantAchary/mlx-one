# Copyright 2026 Youssof Altoukhi
# SPDX-License-Identifier: Apache-2.0
# Adapted from MTPLX 2.13.0, mtplx/models/inkling. See MTPLX_NOTICE.txt.
"""Inkling's chronological attention over grouped KV, with one sink per query."""

from functools import lru_cache

import mlx.core as mx

from ._long_attention_metal import HEADER


@lru_cache(maxsize=1)
def _grouped_mask_kernel():
    return mx.fast.metal_kernel(
        name="mtplx_inkling_grouped_sink_mask",
        input_names=["profiles", "keys"],
        output_names=["mask"],
        ensure_row_contiguous=False,
        source="""
            const int col = thread_position_in_grid.x, row = thread_position_in_grid.y;
            const int head = thread_position_in_grid.z, rows = profiles_shape[1];
            const int count = keys_shape[2], extent = profiles_shape[3];
            if (col >= count) return;
            const int distance = count - rows + row - col;
            const bool visible = distance >= 0 && (!WINDOW || distance < WINDOW);
            T bias = T(0);
            if (!visible) bias = T(-INFINITY);
            else if (distance < extent) {
                const size_t offset = row * profiles_strides[1] + head * profiles_strides[2];
                bias = profiles[offset + distance * profiles_strides[3]];
            }
            mask[(head * rows + row) * size_t(count) + col] = bias;
        """,
    )


def grouped_sink_attention(query, key, value, profiles, sink, window):
    """One native SDPA sink per (head,row), sharing the original KV heads."""
    _, heads, rows, width = query.shape
    if window and key.shape[2] > window + rows - 1:
        start = key.shape[2] - (window + rows - 1)
        key, value = key[:, :, start:], value[:, :, start:]
    count = key.shape[2]
    shape = (1, heads * rows, 1, count)
    mask = _grouped_mask_kernel()(
        inputs=[profiles, key],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(count, rows, heads),
        threadgroup=(min(128, count), 1, 1),
        output_shapes=[shape],
        output_dtypes=[query.dtype],
    )[0]
    # Flatten head first so integer division by the expanded GQA group still
    # maps every query row to the original KV head. Each sink stays independent.
    result = mx.fast.scaled_dot_product_attention(
        query.reshape(1, heads * rows, 1, width),
        key, value, scale=width**-0.5, mask=mask, sinks=sink[0].T.reshape(-1),
    )
    return result.reshape(query.shape)


@lru_cache(maxsize=1)
def _window_panel_kernel():
    # Follows MTPLX 2.13.0 attend.py's window-panel threadgroup layout and
    # MLX v0.32.2 sdpa_vector's single-pass reduction with an initial sink.
    return mx.fast.metal_kernel(
        name="mlx_one_inkling_window_panel", input_names=["q", "k", "v", "p", "s"],
        output_names=["out"], ensure_row_contiguous=False, header=HEADER,
        source="""
            const int head = int(threadgroup_position_in_grid.x);
            const int row = int(threadgroup_position_in_grid.y);
            const int chain = int(simdgroup_index_in_threadgroup);
            const int lane = int(thread_index_in_simdgroup);
            const int rows = q_shape[2], count = k_shape[2];
            const int end = count - rows + row + 1;
            const int first = max(0, end - WINDOW);
            const int kvhead = head / 3;
            const float scale = 0.08838834764831845f;
            const device T* qp = q + head * q_strides[1] + row * q_strides[2];
            const device T* bias = p + row * p_strides[1] + head * p_strides[2];
            float query[4];
            for (int d = 0; d < 4; ++d) {
                query[d] = scale * float(qp[(4 * lane + d) * q_strides[3]]);
            }
            Chain state;
            chain_open(state);
            if (chain == 0) {
                state.peak = float(s[row * s_strides[1] + head * s_strides[2]]);
                state.mass = 1.0f;
            }
            for (int pos = first + chain; pos < end; pos += 32) {
                const device T* kp = k + kvhead * k_strides[1] + pos * k_strides[2];
                const device T* vp = v + kvhead * v_strides[1] + pos * v_strides[2];
                float keys[4], values[4];
                for (int d = 0; d < 4; ++d) {
                    keys[d] = float(kp[(4 * lane + d) * k_strides[3]]);
                    values[d] = float(vp[(4 * lane + d) * v_strides[3]]);
                }
                const float b = float(bias[(end - 1 - pos) * p_strides[3]]);
                if (b >= -3.3895313892515355e38f) {
                    chain_take(state, key_logit(query, keys, b), values);
                }
            }
            threadgroup float peaks[32], masses[32], transpose[32 * 32];
            if (lane == 0) {
                peaks[chain] = state.peak;
                masses[chain] = state.mass;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float peak = peaks[lane];
            const float rescale = metal::fast::exp(peak - simd_max(peak));
            const float mass = simd_sum(masses[lane] * rescale);
            for (int d = 0; d < 4; ++d) {
                transpose[lane * 32 + chain] = state.acc[d];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float total = simd_sum(transpose[chain * 32 + lane] * rescale);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0) {
                    out[((head * rows + row) * 128) + 4 * chain + d] =
                        T(mass == 0 ? total : total / mass);
                }
            }
        """,
    )


def window_panel_attention(query, key, value, profiles, sink, window):
    """Read the existing chronological window directly for each verification row."""
    rows = query.shape[2]
    count = key.shape[2]
    if (not 2 <= rows <= 4 or not rows <= count <= window + rows
            or not 0 < window <= 1024 or query.shape != (1, 24, rows, 128)
            or key.shape != (1, 8, count, 128) or value.shape != key.shape
            or profiles.shape != (1, rows, 24, 1024) or sink.shape != (1, rows, 24)
            or any(a.dtype != mx.bfloat16 for a in (query, key, value, profiles, sink))):
        return None
    return _window_panel_kernel()(
        inputs=[query, key, value, profiles, sink],
        template=[("T", query.dtype), ("WINDOW", window)],
        grid=(1024 * 24, rows, 1), threadgroup=(1024, 1, 1),
        output_shapes=[query.shape], output_dtypes=[query.dtype],
    )[0]
