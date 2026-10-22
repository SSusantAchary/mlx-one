"""FP64 checks on actual first-layer Qwen3 Q/K normalization and rotary operands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from four_path_oracle import load_reference, snapshot


def run(args):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(args.checkpoint, offline=True)

    def compute():
        nonlocal bundle
        bundle, _, module = load_reference(bundle, args.precision)
        root = Path(__file__).resolve().parents[2]
        if not Path(module.__file__).resolve().is_relative_to(root / "references_libs/mlx-lm-main"):
            raise ValueError("wrong reference")
        attn = bundle.model.model.layers[0].self_attn
        targets = {id(attn.q_norm): "q_norm", id(attn.k_norm): "k_norm"}
        original_norm = nn.RMSNorm.__call__
        original_rope = nn.RoPE.__call__
        rows = []

        def norm(owner, x):
            out = original_norm(owner, x)
            if id(owner) in targets:
                a = snapshot(x[:, -1:]).astype(np.float64)
                w = snapshot(owner.weight).astype(np.float64)
                expected = a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + owner.eps) * w
                rows.append(
                    {
                        "operation": targets[id(owner)],
                        "input_dtype": str(x.dtype),
                        "eps": owner.eps,
                        **compare_tensors(
                            expected, snapshot(out[:, -1:]), dtype=str(x.dtype).rsplit(".", 1)[-1]
                        ),
                    }
                )
            return out

        def rope(owner, x, offset=0):
            out = original_rope(owner, x, offset=offset)
            if owner is attn.rope:
                a = snapshot(x[:, :, -1:]).astype(np.float64)
                dims = owner.dims
                angle = (
                    (offset + x.shape[2] - 1)
                    * owner.scale
                    / (owner.base ** (np.arange(0, dims, 2) / dims))
                )
                expected = a.copy()
                half = dims // 2
                expected[..., :half] = a[..., :half] * np.cos(angle) - a[..., half:dims] * np.sin(
                    angle
                )
                expected[..., half:dims] = a[..., :half] * np.sin(angle) + a[
                    ..., half:dims
                ] * np.cos(angle)
                rows.append(
                    {
                        "operation": "rope",
                        "input_dtype": str(x.dtype),
                        "absolute_position": offset + x.shape[2] - 1,
                        "traditional": owner.traditional,
                        "base": owner.base,
                        **compare_tensors(
                            expected,
                            snapshot(out[:, :, -1:]),
                            dtype=str(x.dtype).rsplit(".", 1)[-1],
                        ),
                    }
                )
            return out

        nn.RMSNorm.__call__ = norm
        nn.RoPE.__call__ = rope
        try:
            tokens = (
                bundle.tokenizer.encode("The red book was on the table. The city is calm. ") * 100
            )[:128]
            for mode in ["natural", "production_fp16"]:
                cache = bundle.model.make_cache()
                if mode == "production_fp16":
                    cache = configure_kv_caches(cache, "f16", "f16")
                for start, end in [(0, 127), (127, 128)]:
                    before = len(rows)
                    out = bundle.model(mx.array([tokens[start:end]]), cache=cache)
                    mx.eval(out.logits)
                    for row in rows[before:]:
                        row.update(mode=mode, prefix_end=end)
                for c in cache:
                    c.reset()
        finally:
            nn.RMSNorm.__call__ = original_norm
            nn.RoPE.__call__ = original_rope
        return {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": args.precision,
            "reference_module": module.__file__,
            "input_ids": tokens,
            "rows": rows,
            "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--precision", default="stored")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    a.output.write_text(json.dumps(run(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
