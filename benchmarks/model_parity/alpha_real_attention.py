"""Actual trained Q/K/V operands versus FP64, preserving the historical dtype gates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from four_path_oracle import load_reference, snapshot


def run(path, precision):
    import mlx.core as mx

    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(256 * 1024**2)
    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, module = load_reference(bundle, precision)
        config = bundle.model.config
        count = config.n_layer if bundle.architecture == "gpt2" else config.num_hidden_layers
        selected = {0, count // 2, count - 1}
        if bundle.architecture == "gemma3_text":
            selected.add(5)
        report = {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "profile": precision,
            "reference_module": module.__file__,
            "selected_layers": sorted(selected),
            "rows": [],
            "qualification_eligible": False,
        }
        seed = bundle.tokenizer.encode("The red book was on the table. ")
        for length in [16, 128, 512, 1024, 2048]:
            if length > bundle.context_length:
                continue
            original = mx.fast.scaled_dot_product_attention
            captures, index = [], 0

            def capture(
                q, k, v, *, scale, mask=None, original=original, captures=captures, **kwargs
            ):
                nonlocal index
                output = original(q, k, v, scale=scale, mask=mask, **kwargs)
                current = index
                index += 1
                if current in selected:
                    last_mask = mask if mask is None or isinstance(mask, str) else mask[..., -1:, :]
                    mx.eval(q, k, v, output)
                    captures.append(
                        (current, q[:, :, -1:, :], k, v, output[:, :, -1:, :], last_mask, scale)
                    )
                return output

            mx.fast.scaled_dot_product_attention = capture
            try:
                tokens = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
                output = reference(mx.array([tokens]))
                mx.eval(output)
                del output
            finally:
                mx.fast.scaled_dot_product_attention = original
            for layer, q, k, v, full, mask, scale in captures:
                a, b, c = [snapshot(x).astype(np.float64) for x in (q, k, v)]
                repeat = q.shape[1] // k.shape[1]
                b, c = [np.repeat(x, repeat, axis=1) for x in (b, c)]
                scores = (a @ b.swapaxes(-1, -2)) * scale
                if mask is not None and not isinstance(mask, str):
                    additive = snapshot(mask)
                    if mask.dtype == mx.bool_:
                        additive = np.where(additive, 0, -np.inf)
                    scores += additive
                scores -= scores.max(axis=-1, keepdims=True)
                probability = np.exp(scores)
                probability /= probability.sum(axis=-1, keepdims=True)
                expected = probability @ c
                dtype = str(q.dtype).rsplit(".", 1)[-1]
                single_mask = None if mask is None or isinstance(mask, str) else mask
                outputs = {
                    "full_query_last": full,
                    "single_query_view": original(q, k, v, scale=scale, mask=single_mask),
                    "single_query_contiguous": original(
                        mx.contiguous(q),
                        mx.contiguous(k),
                        mx.contiguous(v),
                        scale=scale,
                        mask=single_mask,
                    ),
                }
                metrics = {}
                for name, actual in outputs.items():
                    value = snapshot(actual).astype(np.float64)
                    metrics[name] = {
                        **compare_tensors(expected, value, dtype=dtype),
                        "rms_error": float(np.sqrt(np.mean((value - expected) ** 2))),
                    }
                report["rows"].append(
                    {
                        "prefix": length,
                        "layer": layer,
                        "shapes": [list(x.shape) for x in (q, k, v)],
                        "operand_dtype": dtype,
                        "scale": scale,
                        "metrics": metrics,
                        "equal_operands": True,
                        "full_single_shape_max": float(
                            np.max(np.abs(snapshot(full) - snapshot(outputs["single_query_view"])))
                        ),
                    }
                )
            del captures
            mx.clear_cache()
        report["status"] = (
            "PASS"
            if all(v["status"] == "PASS" for r in report["rows"] for v in r["metrics"].values())
            else "FAIL"
        )
        return report

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--precision", default="stored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(args.checkpoint, args.precision)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
