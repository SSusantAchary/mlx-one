"""Compare actual post-RoPE K/V across full, natural-cache and FP16-cache paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from four_path_oracle import load_reference, snapshot


def run(args):
    import mlx.core as mx

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
        n = args.context
        parts = [n // 3, n // 3, n - 2 * (n // 3)]
        tokens = (
            bundle.tokenizer.encode("The red book was on the table. The city is calm. ") * 500
        )[:n]
        data = {}
        for mode in ["A", "B", "C"]:
            cache = None if mode == "A" else bundle.model.make_cache()
            if mode == "C":
                cache = configure_kv_caches(cache, "f16", "f16")
            start = 0
            captures = {}
            for length in [n] if mode == "A" else parts:
                original = mx.fast.scaled_dot_product_attention
                index = 0

                def call(q, k, v, *, scale, mask=None, original=original, captures=captures, **kw):
                    nonlocal index
                    current = index
                    index += 1
                    result = original(q, k, v, scale=scale, mask=mask, **kw)
                    captures[current] = {
                        "q": snapshot(q[:, :, -1:]),
                        "k": snapshot(k),
                        "v": snapshot(v),
                        "dtype": str(q.dtype).rsplit(".", 1)[-1],
                    }
                    return result

                mx.fast.scaled_dot_product_attention = call
                try:
                    out = bundle.model(mx.array([tokens[start : start + length]]), cache=cache)
                    mx.eval(out.logits)
                    del out
                finally:
                    mx.fast.scaled_dot_product_attention = original
                start += length
            data[mode] = captures
            if cache:
                for c in cache:
                    c.reset()
        rows = []
        for layer, a in data["A"].items():
            for modes in [("A", "B"), ("A", "C"), ("B", "C")]:
                left, right = data[modes[0]][layer], data[modes[1]][layer]
                metrics = {}
                for component in ["q", "k", "v"]:
                    x, y = left[component], right[component]
                    length = min(x.shape[2], y.shape[2])
                    x, y = x[:, :, -length:], y[:, :, -length:]
                    metrics[component] = compare_tensors(x, y, dtype=a["dtype"])
                rows.append(
                    {
                        "layer": layer,
                        "pair": "_vs_".join(modes),
                        "components": metrics,
                        "logical_tail_aligned": True,
                    }
                )
        return {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": args.precision,
            "input_ids": tokens,
            "context": n,
            "parts": parts,
            "reference_module": module.__file__,
            "rows": rows,
            "interpretation": (
                "Numerical equivalence of actual aligned post-RoPE attention operands; "
                "storage-append semantic checks are separate"
            ),
            "status": "MEASURED_NOT_RELEASE_APPROVED",
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--precision", default="stored")
    p.add_argument("--context", default=512, type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("output exists")
    a.output.write_text(json.dumps(run(a), indent=2) + "\n")


if __name__ == "__main__":
    main()
