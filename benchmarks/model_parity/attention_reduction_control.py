"""Exact-checkpoint, equal-Q/K/V attention equations against NumPy float64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from four_path_oracle import compare, load_reference, snapshot


def run(path):
    import mlx.core as mx
    import numpy as np

    from mlx_one.ops.attention import attention_reference
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, _ = load_reference(bundle, "stored")
        original = mx.fast.scaled_dot_product_attention
        captures = []

        def capture(q, k, v, *, scale, mask=None, **kwargs):
            output = original(q, k, v, scale=scale, mask=mask, **kwargs)
            if not captures:
                mx.eval(q, k, v, output)
                captures.append((q, k, v, output, mask, scale))
            return output

        try:
            mx.fast.scaled_dot_product_attention = capture
            seed = bundle.tokenizer.encode("The capital of France is")
            ids = (seed * 16)[:16]
            mx.eval(reference(mx.array([ids])))
        finally:
            mx.fast.scaled_dot_product_attention = original
        q, k, v, output, mask, scale = captures[0]
        dtype = str(q.dtype).rsplit(".", 1)[-1]
        cq, ck, cv = (snapshot(x).astype(np.float64) for x in (q, k, v))
        repeats = q.shape[1] // k.shape[1]
        ck, cv = (np.repeat(x, repeats, axis=1) for x in (ck, cv))
        if isinstance(mask, str):
            mask = mx.arange(k.shape[2])[None] <= mx.arange(q.shape[2])[:, None]
        additive = None
        if mask is not None:
            additive = mx.where(mask, 0.0, -float("inf")) if mask.dtype == mx.bool_ else mask
        scores = (cq @ ck.swapaxes(-1, -2)) * scale
        if additive is not None:
            scores += snapshot(additive)
        scores -= scores.max(-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(-1, keepdims=True)
        cpu = probabilities @ cv
        dense = attention_reference(q, k, v, scale=scale, mask=mask)
        rowwise = attention_reference(q, k, v, scale=scale, mask=mask, stable=True).astype(q.dtype)
        fq, fk, fv = (x.astype(mx.float32) for x in (q, k, v))
        fk, fv = (mx.repeat(x, repeats, axis=1) for x in (fk, fv))
        early_scores = (fq * scale) @ fk.swapaxes(-1, -2)
        if additive is not None:
            early_scores += additive.astype(mx.float32)
        query_scaled = (mx.softmax(early_scores, axis=-1, precise=True) @ fv).astype(q.dtype)
        return {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "precision": "stored",
            "operand_dtypes": [str(x.dtype) for x in (q, k, v, output)],
            "shapes": [list(x.shape) for x in (q, k, v)],
            "scale": scale,
            "identical_operands": True,
            "token_ids": ids,
            "reference_vs_native_dense": compare(snapshot(output), snapshot(dense), dtype),
            "reference_vs_query_scaled": compare(snapshot(output), snapshot(query_scaled), dtype),
            "reference_vs_rowwise": compare(snapshot(output), snapshot(rowwise), dtype),
            "cpu_vs_reference_sdpa": compare(cpu, snapshot(output), dtype),
            "cpu_vs_native_dense": compare(cpu, snapshot(dense), dtype),
            "cpu_vs_query_scaled": compare(cpu, snapshot(query_scaled), dtype),
            "cpu_vs_rowwise": compare(cpu, snapshot(rowwise), dtype),
            "diagnostic_only": True,
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["checkpoint"], result["reference_vs_native_dense"]["max_abs_error"], flush=True)


if __name__ == "__main__":
    main()
