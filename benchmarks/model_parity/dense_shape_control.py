"""Equal-input source-weight GEMM/GEMV control with a CPU FP64 adjudicator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from four_path_oracle import compare, load_reference, snapshot


def run(path, target, length):
    import mlx.core as mx
    import numpy as np

    from mlx_one.server.model_manager import ModelManager

    manager = ModelManager()
    bundle = manager.load(path, offline=True)

    def compute():
        nonlocal bundle
        bundle, reference, _ = load_reference(bundle, "stored")
        module = dict(bundle.model.named_modules())[target]
        seed = bundle.tokenizer.encode("The capital of France is")
        tokens = mx.array([(seed * length)[:length]])
        captured = []
        cls = type(module)
        owned, original = "__call__" in cls.__dict__, cls.__call__

        def capture(self, value):
            output = original(self, value)
            if self is module:
                captured.append((mx.array(snapshot(value), dtype=value.dtype), snapshot(output)))
            return output

        try:
            cls.__call__ = capture
            mx.eval(bundle.model(tokens).logits)
            full_input, full_output = captured.pop()
            cache = bundle.model.make_cache()
            mx.eval(bundle.model(tokens[:, :-1], cache=cache).logits)
            captured.clear()
            mx.eval(bundle.model(tokens[:, -1:], cache=cache).logits)
            split_input, split_output = captured.pop()
        finally:
            if owned:
                cls.__call__ = original
            else:
                del cls.__call__
        value = full_input
        dtype = str(value.dtype).rsplit(".", 1)[-1]
        single = module(value[:, -1:])
        weight = module.weight
        fp32 = value.astype(mx.float32) @ weight.astype(mx.float32).T
        fp32_single = value[:, -1:].astype(mx.float32) @ weight.astype(mx.float32).T
        rows = value.reshape(-1, 1, value.shape[-1])
        gathered = mx.gather_mm(
            rows.astype(mx.float32),
            weight.astype(mx.float32).T,
            lhs_indices=mx.arange(rows.shape[0]),
            rhs_indices=mx.zeros((rows.shape[0],), mx.int32),
        ).reshape(*value.shape[:-1], weight.shape[0])
        if "bias" in module:
            fp32, fp32_single, gathered = (x + module.bias for x in (fp32, fp32_single, gathered))
        cpu = snapshot(value[:, -1:]).astype(np.float64) @ snapshot(weight).astype(np.float64).T
        if "bias" in module:
            cpu += snapshot(module.bias)
        return {
            "checkpoint": bundle.model_id,
            "revision": bundle.revision,
            "precision": "stored",
            "target": target,
            "length": length,
            "input_dtype": dtype,
            "weight_dtype": str(weight.dtype),
            "output_dtype": str(single.dtype),
            "full_vs_cached_input": compare(snapshot(value[:, -1:]), snapshot(split_input), dtype),
            "full_vs_cached_output": compare(full_output[:, -1:], split_output, dtype),
            "equal_input_full_vs_single": compare(full_output[:, -1:], snapshot(single), dtype),
            "fp32_full_vs_single_cast": compare(
                snapshot(fp32[:, -1:].astype(value.dtype)),
                snapshot(fp32_single.astype(value.dtype)),
                dtype,
            ),
            "cpu_vs_native_full": compare(cpu, full_output[:, -1:], dtype),
            "cpu_vs_native_single": compare(cpu, snapshot(single), dtype),
            "cpu_vs_fp32": compare(cpu, snapshot(fp32_single.astype(value.dtype)), dtype),
            "cpu_vs_rowwise_fp32": compare(
                cpu, snapshot(gathered[:, -1:].astype(value.dtype)), dtype
            ),
            "diagnostic_only": True,
        }

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = run(args.checkpoint, args.target, args.length)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["checkpoint"], result["equal_input_full_vs_single"]["max_abs_error"], flush=True)


if __name__ == "__main__":
    main()
