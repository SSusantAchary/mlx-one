"""Actual GPT-2 first-layer linear operands and per-layer last-token growth."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from four_path_oracle import load_reference, snapshot


def run(path):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.core.cache import configure_kv_caches
    from mlx_one.ops.numerical_trace import compare_tensors
    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, _, module = load_reference(bundle, "stored")
        root = Path(__file__).resolve().parents[2]
        if not Path(module.__file__).resolve().is_relative_to(root / "references_libs/mlx-lm-main"):
            raise ValueError("reference import mismatch")
        model = bundle.model
        ids = bundle.tokenizer.encode("The red book was on the table. The city is calm. ")
        ids = (ids * 500)[:512]
        selected = {
            id(layer): name
            for name, layer in model.named_modules()
            if name
            and (
                isinstance(layer, nn.Linear)
                or name.startswith("transformer.h.")
                and name.count(".") == 2
            )
        }
        targets = {
            "qkv": model.transformer.h[0].attn.c_attn,
            "attention_output_projection": model.transformer.h[0].attn.c_proj,
        }
        result = {
            "timestamp": time.time(),
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "input_ids": ids,
            "parts": [170, 170, 172],
            "reference_module": module.__file__,
            "tf32_env": __import__("os").environ.get("MLX_ENABLE_TF32"),
            "paths": {},
            "linear_controls": {},
        }
        paths = {}
        inputs = {}
        for mode, parts in [("A", [512]), ("B", [170, 170, 172]), ("C", [170, 170, 172])]:
            cache = model.make_cache() if mode != "A" else None
            if mode == "C":
                cache = configure_kv_caches(cache, "f16", "f16")
            events = {}
            captured = {}
            originals = {}

            def wrapper(original, events=events, captured=captured):
                def call(owner, *args, **kwargs):
                    out = original(owner, *args, **kwargs)
                    name = selected.get(id(owner))
                    if name:
                        value = out[0] if isinstance(out, tuple) else out
                        if isinstance(value, mx.array) and value.ndim == 3:
                            events[name] = snapshot(value[:, -1:, :])
                    for key, target in targets.items():
                        if owner is target:
                            mx.eval(args[0], out)
                            captured[key] = (mx.array(snapshot(args[0])), snapshot(out[:, -1:, :]))
                    return out

                return call

            try:
                for cls in {
                    type(layer) for _, layer in model.named_modules() if id(layer) in selected
                }:
                    originals[cls] = ("__call__" in cls.__dict__, cls.__call__)
                    cls.__call__ = wrapper(cls.__call__)
                start = 0
                for n in parts:
                    out = model(mx.array([ids[start : start + n]]), cache=cache)
                    mx.eval(out.logits)
                    logits = snapshot(out.logits[:, -1:, :])
                    del out
                    start += n
            finally:
                for cls, (owned, original) in originals.items():
                    if owned:
                        cls.__call__ = original
                    else:
                        del cls.__call__
            paths[mode] = events
            inputs[mode] = captured
            result["paths"][mode] = {
                "logit_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                "logits_last16": logits.ravel()[:16].tolist(),
            }
        for mode in ["B", "C"]:
            comparisons = [
                {"operation": name, **compare_tensors(a, paths[mode][name], dtype="float32")}
                for name, a in paths["A"].items()
                if name in paths[mode]
            ]
            result["paths"][mode]["growth"] = comparisons
            result["paths"][mode]["first_nonzero"] = next(
                (v["operation"] for v in comparisons if v["changed_elements"]), None
            )
        for name, target in targets.items():
            x = inputs["A"][name][0]
            a = snapshot(x[:, -1:, :]).astype(np.float64)
            w = snapshot(target.weight).astype(np.float64)
            bias = snapshot(target.bias).astype(np.float64)
            expected = a @ w.T + bias
            full = target(x)
            single = target(x[:, -1:, :])
            contig = target(mx.contiguous(x[:, -1:, :]))
            results = {
                label: compare_tensors(expected, snapshot(v).astype(np.float64), dtype="float32")
                for label, v in [
                    ("full_last", full[:, -1:, :]),
                    ("single", single),
                    ("contiguous", contig),
                ]
            }
            result["linear_controls"][name] = {
                "input_shape": list(x.shape),
                "weight_shape": list(target.weight.shape),
                "input_dtype": str(x.dtype),
                "weight_dtype": str(target.weight.dtype),
                "weight_sha256": hashlib.sha256(w.tobytes()).hexdigest(),
                "comparisons": results,
                "B_input_vs_A": compare_tensors(
                    snapshot(x[:, -1:, :]),
                    snapshot(inputs["B"][name][0][:, -1:, :]),
                    dtype="float32",
                ),
                "C_input_vs_A": compare_tensors(
                    snapshot(x[:, -1:, :]),
                    snapshot(inputs["C"][name][0][:, -1:, :]),
                    dtype="float32",
                ),
            }
        result["status"] = "MEASURED_NOT_RELEASE_APPROVED"
        return result

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists")
    args.output.write_text(json.dumps(run(args.checkpoint), indent=2) + "\n")


if __name__ == "__main__":
    main()
