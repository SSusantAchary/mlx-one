# Copyright © 2023 Apple Inc. (FP4/E8M0 representation reference, MIT)
# New MLX-One projection/fusion implementation: Apache-2.0.
# See models/UPSTREAM_LICENSE and docs/acceleration/gptoss20b/upstream_references.md.
"""Experimental B1 MXFP4 kernels. Callers retain stock-MLX fallbacks.

Packed weights stay uint32, with eight E2M1 nibbles/word and one E8M0 byte per
32 values. No experts are expanded to dense matrices. All routing stays on GPU.
Unsupported shapes/dtypes return None before dispatch; invalid expert IDs never
form an out-of-bounds address. This module is imported only by opt-in execution.
"""

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
// FP4/E8M0 conversions follow the formats documented in Apple's MIT-licensed
// mlx/backend/metal/kernels/fp4.h and fp8.h. Preserve signed zero and specials.
inline float gptoss_fp4(uint bits) {
    float value = float(as_type<half>(ushort((bits & 7u) << 9))) * 16384.0f;
    return (bits & 8u) ? -value : value;
}
inline float gptoss_scale(uchar bits) {
    uint value = bits == 0 ? 0x00400000u : bits == 255 ? 0x7fc00000u : uint(bits) << 23;
    return as_type<float>(value);
}
template <typename T> inline T gptoss_activation(T gate, T up) {
    T g = T(min(float(gate), 7.0f));
    T u = T(clamp(float(up), -7.0f, 7.0f));
    T z = T(float(g) * float(T(1.702f)));
    auto y = 1 / (1 + metal::exp(metal::abs(z)));
    T sigmoid = (z < 0) ? y : 1 - y;
    T swish = T(float(g) * float(sigmoid));
    T shifted = T(float(u) + float(T(1.0f)));
    return T(float(shifted) * float(swish));
}
"""


def supported(module, x, indices):
    """Metadata-only guard; never reads expert IDs back to the host."""
    return (
        mx.default_device() == mx.gpu
        and getattr(module, "mode", None) == "mxfp4"
        and module.bits == 4
        and module.group_size == 32
        and x.dtype in {mx.float32, mx.float16, mx.bfloat16}
        and module.weight.dtype == mx.uint32
        and module.scales.dtype == mx.uint8
        and module.weight.ndim == 3
        and module.scales.ndim == 3
        and module.weight.shape[-1] * 8 == x.shape[-1]
        and x.shape[-1] % 32 == 0
        and indices.size == 4
        and x.size // x.shape[-1] in {1, 4}
        and indices.dtype in {mx.uint32, mx.int32, mx.int64}
        and "bias" in module
        and module.bias.dtype == x.dtype
        and module.scales.shape == (*module.weight.shape[:2], x.shape[-1] // 32)
        and module.bias.shape == module.weight.shape[:2]
    )


@lru_cache(maxsize=16)
def _projection_kernel(pair=False, activate=False, words_per_lane=1):
    if words_per_lane not in {1, 4}:
        raise ValueError("words_per_lane must be 1 or 4")
    inputs = ["x", "weight", "scales", "bias", "indices"]
    if pair:
        inputs += ["up_weight", "up_scales", "up_bias"]
    outputs = ["out"] if not pair or activate else ["gate_out", "up_out"]
    # Four neighboring rows/threadgroup; one SIMD group/row. A paired projection
    # shares activation reads but retains independent FP32 accumulators.
    source = r"""
    uint lane = thread_position_in_threadgroup.x;
    uint row = thread_position_in_grid.y;
    uint choice = thread_position_in_grid.z;
    uint N = weight_shape[1];
    uint K = x_shape[1];
    if (row >= N) return;
    long expert_value = long(indices[choice]);
    if (expert_value < 0 || expert_value >= long(weight_shape[0])) {
        if (lane == 0) {
    INVALID_WRITE
        }
        return;
    }
    uint expert = uint(expert_value);
    uint input_row = x_shape[0] == 1 ? 0 : choice;
    uint base = (expert * N + row) * (K / 8);
    uint sb = (expert * N + row) * (K / 32);
    float acc = 0.0f;
    PAIR_ACC
    for (uint p0 = lane * WORDS; p0 < K / 8; p0 += 32 * WORDS) {
        #pragma unroll
        for (uint w = 0; w < WORDS; ++w) {
            uint p = p0 + w;
            if (p >= K / 8) break;
            uint packed = weight[base + p];
            float scale = gptoss_scale(scales[sb + p / 4]);
            PAIR_LOAD
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                float value = float(x[input_row * K + p * 8 + j]);
                float weight_value = gptoss_fp4((packed >> (j * 4)) & 15u) * scale;
                acc += value * weight_value;
                PAIR_DOT
            }
        }
    }
    T gate_value = T(simd_sum(acc));
    gate_value = T(float(gate_value) + float(bias[expert * N + row]));
    PAIR_REDUCE
    if (lane == 0) {
        RESULT_WRITE
    }
    """
    source = source.replace("WORDS", str(words_per_lane))
    if pair:
        source = source.replace("PAIR_ACC", "float up_acc = 0.0f;")
        source = source.replace(
            "PAIR_LOAD",
            "uint up_packed = up_weight[base + p]; float us = gptoss_scale(up_scales[sb + p / 4]);",
        )
        source = source.replace(
            "PAIR_DOT", "up_acc += value * (gptoss_fp4((up_packed >> (j * 4)) & 15u) * us);"
        )
        source = source.replace(
            "PAIR_REDUCE",
            "T up_value = T(simd_sum(up_acc)); "
            "up_value = T(float(up_value) + float(up_bias[expert * N + row]));",
        )
    else:
        for marker in ("PAIR_ACC", "PAIR_LOAD", "PAIR_DOT", "PAIR_REDUCE"):
            source = source.replace(marker, "")
    writes = (
        "out[choice * N + row] = gptoss_activation(gate_value, up_value);"
        if activate
        else "gate_out[choice * N + row] = gate_value; up_out[choice * N + row] = up_value;"
        if pair
        else "out[choice * N + row] = gate_value;"
    )
    source = source.replace("RESULT_WRITE", writes)
    invalid = " ".join(f"{name}[choice * N + row] = T(NAN);" for name in outputs)
    source = source.replace("INVALID_WRITE", invalid)
    return mx.fast.metal_kernel(
        name=f"gptoss_mxfp4_{int(pair)}_{int(activate)}_{words_per_lane}",
        input_names=inputs,
        output_names=outputs,
        header=_HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def project(module, x, indices, *, words_per_lane=1, rows_per_group=4):
    if not supported(module, x, indices):
        return None
    if rows_per_group not in {1, 2, 4, 8}:
        raise ValueError("Unsupported threadgroup configuration")
    n, k = module.weight.shape[1], x.shape[-1]
    result = _projection_kernel(words_per_lane=words_per_lane)(
        inputs=[x.reshape(-1, k), module.weight, module.scales, module.bias, indices.reshape(-1)],
        template=[("T", x.dtype)],
        grid=(32, ((n + rows_per_group - 1) // rows_per_group) * rows_per_group, 4),
        threadgroup=(32, rows_per_group, 1),
        output_shapes=[(4, n)],
        output_dtypes=[x.dtype],
    )[0]
    return result.reshape(*indices.shape, 1, n)


def gate_up(gate, up, x, indices, *, activate=True, words_per_lane=1, rows_per_group=4):
    if not supported(gate, x, indices) or not supported(up, x, indices):
        return None
    if gate.weight.shape != up.weight.shape or rows_per_group not in {1, 2, 4, 8}:
        return None
    n, k = gate.weight.shape[1], x.shape[-1]
    count = 1 if activate else 2
    result = _projection_kernel(True, activate, words_per_lane)(
        inputs=[
            x.reshape(-1, k),
            gate.weight,
            gate.scales,
            gate.bias,
            indices.reshape(-1),
            up.weight,
            up.scales,
            up.bias,
        ],
        template=[("T", x.dtype)],
        grid=(32, ((n + rows_per_group - 1) // rows_per_group) * rows_per_group, 4),
        threadgroup=(32, rows_per_group, 1),
        output_shapes=[(4, n)] * count,
        output_dtypes=[x.dtype] * count,
    )
    shaped = tuple(value.reshape(*indices.shape, 1, n) for value in result)
    return shaped[0] if activate else shaped


@lru_cache(maxsize=1)
def _activation_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_swiglu_b1",
        input_names=["gate", "up"],
        output_names=["out"],
        header=_HEADER,
        source=r"""
        uint i = thread_position_in_grid.x;
        if (i < COUNT) out[i] = gptoss_activation(gate[i], up[i]);
        """,
    )


def swiglu(gate, up):
    if (
        mx.default_device() != mx.gpu
        or gate.shape != up.shape
        or gate.dtype != up.dtype
        or gate.dtype not in {mx.float32, mx.float16, mx.bfloat16}
    ):
        return None
    return _activation_kernel()(
        inputs=[gate, up],
        template=[("T", gate.dtype), ("COUNT", gate.size)],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


@lru_cache(maxsize=1)
def _reduction_kernel():
    return mx.fast.metal_kernel(
        name="gptoss_weighted_top4",
        input_names=["values", "scores"],
        output_names=["out"],
        source=r"""
        uint d = thread_position_in_grid.x;
        if (d >= D) return;
        T value = T(0);
        #pragma unroll
        for (uint e = 0; e < 4; ++e) {
            T product = T(float(values[e * D + d]) * float(scores[e]));
            value = T(float(value) + float(product));
        }
        out[d] = value;
        """,
    )


def weighted_sum(values, scores):
    if (
        mx.default_device() != mx.gpu
        or values.dtype not in {mx.float32, mx.float16, mx.bfloat16}
        or values.size != scores.size * values.shape[-1]
        or scores.size != 4
        or values.dtype != scores.dtype
    ):
        return None
    d = values.shape[-1]
    return _reduction_kernel()(
        inputs=[values, scores],
        template=[("T", values.dtype), ("D", d)],
        grid=(d, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*scores.shape[:-1], d)],
        output_dtypes=[values.dtype],
    )[0]
