"""Bounded equal-operand attention controls; no inferred kernel accumulation contract."""

from __future__ import annotations

import hashlib

import numpy as np
from four_path_oracle import snapshot


def attention_control(q, k, v, output, mask, scale, *, label, sdpa=None):
    import mlx.core as mx

    from mlx_one.ops.numerical_trace import compare_tensors

    dtype = str(q.dtype).rsplit(".", 1)[-1]
    sdpa = sdpa or mx.fast.scaled_dot_product_attention
    a, b, c = [snapshot(x).astype(np.float64) for x in (q[:, :, -1:, :], k, v)]
    repeated = q.shape[1] // k.shape[1]
    b, c = np.repeat(b, repeated, axis=1), np.repeat(c, repeated, axis=1)
    scores = a @ b.swapaxes(-1, -2) * scale
    last_mask = None
    if mask is not None and not isinstance(mask, str):
        last_mask = mask[..., -1:, :]
        m = snapshot(last_mask)
        scores += np.where(m, 0.0, -np.inf) if m.dtype == bool else m
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities @ c
    # A second oracle quantifies rounding K/V before the exact same FP64 arithmetic.
    rounded = [snapshot(x.astype(mx.float16).astype(q.dtype)).astype(np.float64) for x in (k, v)]
    br, cr = [np.repeat(x, repeated, axis=1) for x in rounded]
    sr = a @ br.swapaxes(-1, -2) * scale
    if last_mask is not None:
        sr += np.where(m, 0.0, -np.inf) if m.dtype == bool else m
    pr = np.exp(sr - sr.max(axis=-1, keepdims=True))
    pr /= pr.sum(axis=-1, keepdims=True)
    rounded_expected = pr @ cr
    outputs = {
        "full_shape_last": output[:, :, -1:, :],
        "single_query": sdpa(q[:, :, -1:, :], k, v, scale=scale, mask=last_mask),
        "single_contiguous": sdpa(
            mx.contiguous(q[:, :, -1:, :]),
            mx.contiguous(k),
            mx.contiguous(v),
            scale=scale,
            mask=last_mask,
        ),
    }
    # Force explicit FP32 reductions on identical operands; diagnostic F only.
    q32 = q[:, :, -1:, :].astype(mx.float32)
    k32 = mx.repeat(k, repeated, axis=1).astype(mx.float32)
    v32 = mx.repeat(v, repeated, axis=1).astype(mx.float32)
    score32 = (q32 @ k32.swapaxes(-1, -2)) * scale
    if last_mask is not None:
        score32 = (
            mx.where(last_mask, score32, -float("inf"))
            if last_mask.dtype == mx.bool_
            else score32 + last_mask
        )
    probability32 = mx.softmax(score32, axis=-1, precise=True)
    outputs["explicit_fp32"] = (probability32 @ v32).astype(q.dtype)
    results = {}
    for name, value in outputs.items():
        actual = snapshot(value).astype(np.float64)
        results[name] = {
            **compare_tensors(expected, actual, dtype=dtype),
            "rms_error": float(np.sqrt(np.mean((actual - expected) ** 2))),
        }
    return {
        "id": label,
        "operation": "attention",
        "input_dtype": dtype,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "qkv_last_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in (a, b, c)],
        "scale": scale,
        "mask": "explicit_last_row" if last_mask is not None else "all_keys_visible_last_query",
        "actual_score_range": [float(scores[np.isfinite(scores)].min()), float(scores.max())],
        "probability_range": [float(probabilities.min()), float(probabilities.max())],
        "score_fp32_max_error": float(
            np.max(np.abs(snapshot(score32)[np.isfinite(scores)] - scores[np.isfinite(scores)]))
        ),
        "probability_fp32_max_error": float(
            np.max(np.abs(snapshot(probability32) - probabilities))
        ),
        "fp64_kv_rounding_impact": {
            "max_abs": float(np.max(np.abs(expected - rounded_expected))),
            "rms": float(np.sqrt(np.mean((expected - rounded_expected) ** 2))),
        },
        "comparisons": results,
        "criterion": "unchanged frozen component tolerance, actual Q dtype",
        "accumulator_dtype": None,
        "accumulator_evidence": "Public dispatch cannot certify fused accumulator type",
    }
