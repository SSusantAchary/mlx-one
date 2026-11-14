# Copyright © 2024 Apple Inc.
# SPDX-License-Identifier: MIT
# Adapted from MLX v0.32.2 sdpa_vector.h, commit
# 1f8e74e3f12f31365464a6867c6579f0e9b29d85; see models/UPSTREAM_LICENSE.
"""Experimental sink-aware GQA decode with stock-MLX fallback at the caller.

The split pass reuses each K/V read across eight query heads. Exactly one
subchunk includes the per-head sink in its softmax denominator. No key/value
is associated with the sink. Only full-history B1, single-query D64 is eligible.
"""

from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _split_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_gqa_sinks_split",
        input_names=["queries", "keys", "values", "sinks"],
        output_names=["partials", "sums", "maxs"],
        ensure_row_contiguous=False,
        source=r"""
        constexpr int G = 8;
        constexpr int HPT = 8;
        constexpr int D = 64;
        constexpr float lowest = -3.402823466e+38f;
        uint lane = thread_position_in_threadgroup.x;
        uint g = thread_position_in_threadgroup.y;
        uint kv_head = threadgroup_position_in_grid.x;
        uint block = threadgroup_position_in_grid.z;
        int N = keys_shape[2];
        int chunk = (N + BLOCKS - 1) / BLOCKS;
        int start = int(block) * chunk;
        int end = min(N, start + chunk);
        int sub = (chunk + HPT - 1) / HPT;
        int s0 = start + int(g) * sub;
        int s1 = min(end, s0 + sub);
        uint base_head = kv_head * G;

        float q[HPT][2];
        float largest[HPT];
        float denominator[HPT];
        float accumulator[HPT][2];
        for (int h = 0; h < HPT; ++h) {
            for (int i = 0; i < 2; ++i) {
                size_t qi = (base_head + h) * queries_strides[1]
                    + (lane * 2 + i) * queries_strides[3];
                q[h][i] = 0.125f * float(queries[qi]);
                accumulator[h][i] = 0;
            }
            // Include the sink once across all splits/subchunks, for every head.
            bool include_sink = block == 0 && g == 0;
            largest[h] = include_sink
                ? float(sinks[(base_head + h) * sinks_strides[0]]) : lowest;
            denominator[h] = include_sink ? 1.0f : 0.0f;
        }
        for (int t = s0; t < s1; ++t) {
            float kr[2], vr[2];
            for (int i = 0; i < 2; ++i) {
                kr[i] = float(keys[kv_head * keys_strides[1] + t * keys_strides[2]
                    + (lane * 2 + i) * keys_strides[3]]);
                vr[i] = float(values[kv_head * values_strides[1] + t * values_strides[2]
                    + (lane * 2 + i) * values_strides[3]]);
            }
            for (int h = 0; h < HPT; ++h) {
                float score = q[h][0] * kr[0] + q[h][1] * kr[1];
                score = simd_sum(score);
                float next_max = max(largest[h], score);
                float correction = fast::exp(largest[h] - next_max);
                float weight = fast::exp(score - next_max);
                largest[h] = next_max;
                denominator[h] = denominator[h] * correction + weight;
                for (int i = 0; i < 2; ++i)
                    accumulator[h][i] = accumulator[h][i] * correction + weight * vr[i];
            }
        }
        threadgroup float local_output[G * HPT * D];
        threadgroup float local_sums[G * HPT];
        threadgroup float local_maxs[G * HPT];
        for (int h = 0; h < HPT; ++h) {
            uint slot = h * HPT + g;
            float inv = denominator[h] > 0 ? 1.0f / denominator[h] : 0;
            for (int i = 0; i < 2; ++i)
                local_output[slot * D + lane * 2 + i] = accumulator[h][i] * inv;
            if (lane == 0) {
                local_sums[slot] = denominator[h];
                local_maxs[slot] = largest[h];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float largest_value = lowest;
        for (int s = 0; s < HPT; ++s)
            largest_value = max(largest_value, local_maxs[g * HPT + s]);
        float sum = 0, acc[2] = {0, 0};
        for (int s = 0; s < HPT; ++s) {
            float factor = local_sums[g * HPT + s]
                * fast::exp(local_maxs[g * HPT + s] - largest_value);
            sum += factor;
            for (int i = 0; i < 2; ++i)
                acc[i] += factor * local_output[(g * HPT + s) * D + lane * 2 + i];
        }
        uint head = base_head + g;
        for (int i = 0; i < 2; ++i)
            partials[(head * BLOCKS + block) * D + lane * 2 + i] = T(acc[i]);
        if (lane == 0) {
            sums[head * BLOCKS + block] = sum;
            maxs[head * BLOCKS + block] = largest_value;
        }
        """,
    )


@lru_cache(maxsize=1)
def _reduce_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_gqa_sinks_reduce",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=r"""
        uint lane = thread_position_in_threadgroup.x;
        uint head = threadgroup_position_in_grid.x;
        float largest = -3.402823466e+38f;
        for (int b = 0; b < BLOCKS; b += 32)
            largest = max(largest, maxs[head * BLOCKS + b + lane]);
        largest = simd_max(largest);
        float denominator = 0, acc[2] = {0, 0};
        for (int b = 0; b < BLOCKS; b += 32) {
            float factor = fast::exp(maxs[head * BLOCKS + b + lane] - largest);
            denominator += simd_sum(sums[head * BLOCKS + b + lane] * factor);
            for (int s = 0; s < 32; ++s) {
                float weight = simd_shuffle(factor, ushort(s));
                for (int i = 0; i < 2; ++i)
                    acc[i] += weight * float(partials[
                        (head * BLOCKS + b + s) * 64 + lane * 2 + i]);
            }
        }
        for (int i = 0; i < 2; ++i)
            out[head * 64 + lane * 2 + i] = T(denominator > 0 ? acc[i] / denominator : 0);
        """,
    )


def attention(queries, keys, values, sinks, *, blocks=64):
    """Full cached causal attention for a last-position query; None if unsupported."""
    if (
        mx.default_device() != mx.gpu
        or queries.shape != (1, 64, 1, 64)
        or keys.ndim != 4
        or keys.shape[:2] != (1, 8)
        or keys.shape[-1] != 64
        or keys.shape[2] < 1
        or values.shape != keys.shape
        or sinks.shape != (64,)
        or queries.dtype not in {mx.float32, mx.float16, mx.bfloat16}
        or any(value.dtype != queries.dtype for value in (keys, values, sinks))
        or blocks not in {32, 64, 128, 256}
    ):
        return None
    partials, sums, maxs = _split_kernel()(
        inputs=[queries, keys, values, sinks],
        template=[("T", queries.dtype), ("BLOCKS", blocks)],
        grid=(8 * 32, 8, blocks),
        threadgroup=(32, 8, 1),
        output_shapes=[(64, blocks, 64), (64, blocks), (64, blocks)],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )
    return _reduce_kernel()(
        inputs=[partials, sums, maxs],
        template=[("T", queries.dtype), ("BLOCKS", blocks)],
        grid=(64 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
