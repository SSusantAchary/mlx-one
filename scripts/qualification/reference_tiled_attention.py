"""Diagnostic stock-MLX attention preserving the stable reference's row reductions.

This candidate is deliberately separate from production dispatch and qualification.
It replaces physical GQA head repeats with RHS gather indices and bounds each
score/probability tensor by a query tile. Hardware qualification must establish
actual peak memory and FP16 cast-boundary equivalence before any integration.
"""
from __future__ import annotations

import math
from numbers import Real


def _dtype_name(value):
    return str(value.dtype).rsplit(".", 1)[-1]


def _validate(q, k, v, scale, mask, query_tile):
    if type(query_tile) is not int or query_tile < 1:
        raise ValueError("query_tile must be a positive integer")
    if not isinstance(scale, Real) or isinstance(scale, bool) or not math.isfinite(scale):
        raise ValueError("scale must be a finite real number")
    if any(len(t.shape) != 4 or min(t.shape) < 1 for t in (q, k, v)):
        raise ValueError("attention requires positive [B,H,T,D] tensors")
    batch, heads, queries, width = q.shape
    kb, kv_heads, keys, kw = k.shape
    if batch != kb or v.shape[:3] != k.shape[:3] or width != kw or heads % kv_heads:
        raise ValueError("attention batch/head/GQA/width layout mismatch")
    dtypes = tuple(_dtype_name(t) for t in (q, k, v))
    if len(set(dtypes)) != 1 or dtypes[0] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("candidate requires matching FP16/BF16/FP32 Q/K/V")
    if isinstance(mask, str):
        if mask != "causal" or queries > keys:
            raise ValueError("unsupported attention mask or causal alignment")
    elif mask is not None:
        if not hasattr(mask, "ndim") or mask.ndim > 4:
            raise ValueError("mask must be an array with at most four dimensions")
        expected = (batch, heads, queries, keys)
        aligned = (1,) * (4 - mask.ndim) + tuple(mask.shape)
        if any(a not in (1, size) for a, size in zip(aligned, expected, strict=True)):
            raise ValueError("mask cannot broadcast to actual query/key layout")
        if _dtype_name(mask) not in {"bool", "bool_", "float16", "bfloat16", "float32"}:
            raise ValueError("unsupported mask dtype")


def reference_compatible_attention(q, k, v, scale, mask=None, query_tile=32):
    """Run only the candidate graph using the existing reference arithmetic.

    Both products use FP32 ``gather_mm`` with one LHS matrix row per query,
    followed by the original post-product scaling, masking, and precise softmax.
    Native KV matrices are selected using ``batch*Hkv + query_head//GQA``.
    Query tiling changes neither the key reduction width nor per-row products.
    No reference correction, custom Metal, evaluation barrier, or backend claim
    is hidden in this function. Its result is FP32 for all supported input dtypes.
    """
    _validate(q, k, v, scale, mask, query_tile)
    import mlx.core as mx

    batch, heads, queries, width = q.shape
    kv_heads, keys, value_width = k.shape[1], k.shape[2], v.shape[3]
    ratio = heads // kv_heads
    # Reference GQA repeat materializes head-major [K,D] storage before K.T.
    # A padded/noncontiguous native K cast *after* transpose can instead undergo
    # a General copy into row-major [D,K], selecting a different gather GEMV
    # reduction. Normalize only native heads before transpose to preserve the
    # reference layout without repeating heads. V needs the same normalization
    # when its FP32 cast is otherwise a no-op on a noncontiguous view.
    if ratio > 1:
        key_native = mx.contiguous(k.astype(mx.float32), allow_col_major=False)
        value_native = mx.contiguous(v.astype(mx.float32), allow_col_major=False)
        key_rhs = key_native.transpose(0, 1, 3, 2).reshape(batch * kv_heads, width, keys)
        value_rhs = value_native.reshape(batch * kv_heads, keys, value_width)
    else:
        # There is no reference head repeat for MHA. Retain its original layout
        # and cast ordering rather than imposing the GQA normalization there.
        key_rhs = k.transpose(0, 1, 3, 2).astype(mx.float32).reshape(
            batch * kv_heads, width, keys)
        value_rhs = v.astype(mx.float32).reshape(batch * kv_heads, keys, value_width)
    outputs = []
    for start in range(0, queries, query_tile):
        stop = min(start + query_tile, queries)
        tile = stop - start
        rows = batch * heads * tile
        lhs_indices = mx.arange(rows, dtype=mx.int32)
        # Each native RHS head serves ratio adjacent query heads, each with
        # tile query rows. Repeat integer indices only, never the KV tensors.
        rhs_indices = mx.repeat(mx.arange(batch * kv_heads, dtype=mx.int32), ratio * tile)
        query_rows = q[:, :, start:stop, :].astype(mx.float32).reshape(rows, 1, width)
        scores = mx.gather_mm(query_rows, key_rhs, lhs_indices=lhs_indices,
                              rhs_indices=rhs_indices).reshape(batch, heads, tile, keys) * scale
        if isinstance(mask, str):
            # Preserve addition of reference lower-right causal 0/-inf masks.
            tile_mask = mx.triu(mx.full((tile, keys), -float("inf")),
                                k=keys - queries + start + 1)
        elif mask is not None and mask.ndim >= 2 and mask.shape[-2] != 1:
            tile_mask = mask[..., start:stop, :]
        else:
            tile_mask = mask
        if tile_mask is not None:
            scores = (mx.where(tile_mask, scores, -float("inf"))
                      if _dtype_name(tile_mask) in {"bool", "bool_"} else scores + tile_mask)
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        output = mx.gather_mm(
            probabilities.astype(mx.float32).reshape(rows, 1, keys), value_rhs,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices).reshape(
                batch, heads, tile, value_width)
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
