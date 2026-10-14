"""Shape-stable arithmetic for explicit FP16 affine INT8 decoder execution.

Use existing MLX primitives, with one matrix-vector row per real token. Packed
weights remain shared; no weight dequantization or per-token Python loop is used.
Omitted execution policy and other weight formats retain native module dispatch.
"""
from __future__ import annotations

from typing import Any


def attention_scratch_bound(*, batch: int, heads: int, head_dim: int, chunk: int,
                            history: int) -> int:
    """Additional conservative FP32 cast/softmax and gather-index overlap bound.

    Dense cache packing and ordinary activations are reserved separately. This
    bound intentionally includes softmax overlap even if the allocator reuses it.
    """
    if min(batch, heads, head_dim, chunk) < 1 or history < 0:
        raise ValueError("invalid attention scratch dimensions")
    total = history + chunk
    casts = batch * heads * head_dim * (2 * total + chunk) * 4
    softmax_overlap = batch * heads * chunk * total * 4
    indices = batch * chunk * (heads + 1) * 8
    return casts + softmax_overlap + indices


def project(module: Any, value: Any, *, execution_dtype: str | None = None,
            as_linear: bool = False) -> Any:
    if (execution_dtype != "float16" or getattr(module, "bits", None) != 8
            or getattr(module, "mode", None) != "affine"):
        return module.as_linear(value) if as_linear else module(value)
    import mlx.core as mx

    # Ordinary qmm selects different matrix/vector reductions as M changes.
    # Gathered rows keep the reduction shape fixed across full/split/batched input.
    rows = value.reshape(-1, 1, value.shape[-1])
    output = mx.gather_qmm(rows, module.weight, scales=module.scales, biases=module.biases,
        lhs_indices=mx.arange(rows.shape[0]),
        rhs_indices=mx.zeros((rows.shape[0],), dtype=mx.int32), transpose=True,
        group_size=module.group_size, bits=module.bits, mode=module.mode)
    output = output.reshape(*value.shape[:-1], output.shape[-1])
    if not as_linear and "bias" in module:
        output = output + module.bias
    return output


def attention_matmul(left: Any, right: Any) -> Any:
    """FP32 per-query reduction, independent of prefill/decode row count."""
    import mlx.core as mx

    if (left.ndim != 4 or right.ndim != 4 or left.shape[:2] != right.shape[:2]
            or left.shape[-1] != right.shape[-2]):
        raise ValueError("attention matmul requires aligned batch/head dimensions")
    batch, heads, queries, width = left.shape
    rows = batch * heads * queries
    output = mx.gather_mm(left.astype(mx.float32).reshape(rows, 1, width),
        right.astype(mx.float32).reshape(batch * heads, right.shape[-2], right.shape[-1]),
        lhs_indices=mx.arange(rows), rhs_indices=mx.repeat(mx.arange(batch * heads), queries))
    return output.reshape(batch, heads, queries, right.shape[-1])
