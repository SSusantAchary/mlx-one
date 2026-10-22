"""Opt-in actual MLX primitive-call evidence; never a GPU-kernel profiler."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path


def run(checkpoint, precision):
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_one.engine.modes import configure_execution_dtype
    from mlx_one.engine.reporting import tensor_dtypes
    from mlx_one.server.model_manager import ModelManager

    mx.set_memory_limit(20 * 1024**3)
    mx.set_cache_limit(1024**3)
    manager = ModelManager()
    bundle = manager.load(checkpoint, offline=True)

    def compute():
        nonlocal bundle
        import numpy as np

        from mlx_one.ops.numerical_trace import compare_tensors, tensor_metadata

        def single_token_split():
            seed = bundle.tokenizer.encode("The capital of France is")
            prompt = (seed * ((16 + len(seed) - 1) // len(seed)))[:16]
            tokens = mx.array([prompt])
            full = bundle.model(tokens).logits[:, -1:]
            cache = bundle.model.make_cache()
            mx.eval(bundle.model(tokens[:, :-1], cache=cache).logits)
            final = bundle.model(tokens[:, -1:], cache=cache).logits
            mx.eval(full, final)
            result = compare_tensors(
                np.asarray(full.astype(mx.float32)),
                np.asarray(final.astype(mx.float32)),
                dtype=str(full.dtype).rsplit(".", 1)[-1],
                reference_metadata=tensor_metadata(full),
                candidate_metadata=tensor_metadata(final),
            )
            for item in cache:
                item.reset()
            return result

        stored_split = single_token_split()
        norm_isolation = None
        if bundle.architecture == "qwen3_5":
            seed = bundle.tokenizer.encode("The capital of France is")
            tokens = (seed * ((16 + len(seed) - 1) // len(seed)))[:16]
            tower = bundle.model.language_model
            inputs = tower.embed_tokens(mx.array([tokens]))
            norm = tower.layers[0].input_layernorm
            native = norm(inputs)
            folded = mx.fast.rms_norm(inputs, 1 + norm.weight, norm.eps)
            unrounded = mx.fast.rms_norm(
                inputs.astype(mx.float32), 1 + norm.weight.astype(mx.float32), norm.eps
            ).astype(inputs.dtype)
            mx.eval(native, folded, unrounded)
            norm_isolation = {
                "scope": "Before quantization, same exact-checkpoint first-layer embedding operand",
                "input_dtype": str(inputs.dtype),
                "raw_weight_dtype": str(norm.weight.dtype),
                "native_vs_folded_checkpoint_precision": compare_tensors(
                    np.asarray(folded.astype(mx.float32)),
                    np.asarray(native.astype(mx.float32)),
                    dtype=str(inputs.dtype).rsplit(".", 1)[-1],
                    reference_metadata=tensor_metadata(folded),
                    candidate_metadata=tensor_metadata(native),
                ),
                "native_vs_unrounded_fp32": compare_tensors(
                    np.asarray(unrounded.astype(mx.float32)),
                    np.asarray(native.astype(mx.float32)),
                    dtype=str(inputs.dtype).rsplit(".", 1)[-1],
                    reference_metadata=tensor_metadata(unrounded),
                    candidate_metadata=tensor_metadata(native),
                ),
                "interpretation": (
                    "Local MLX-LM folds 1 + weight during sanitization; HF-style native norm "
                    "retains FP32 offset addition. This isolated replay does not excuse "
                    "downstream gate failures."
                ),
            }
        execution = None if precision == "stored" else "float16"
        if precision.startswith("affine_q"):
            bits = int(precision[-1])
            nn.quantize(bundle.model, bits=bits, group_size=64, mode="affine")
            bundle = replace(bundle, quantization={"bits": bits, "group_size": 64})
        configure_execution_dtype(bundle, execution)
        mx.eval(bundle.model.parameters())
        profile_split = single_token_split() if precision != "stored" else stored_split
        calls, counts, originals = [], Counter(), []

        def wrap(name, original):
            def call(*args, **kwargs):
                output = original(*args, **kwargs)
                counts[name] += 1
                if len(calls) < 4096:
                    calls.append(
                        {
                            "primitive": name,
                            "operands": [
                                {"shape": list(x.shape), "dtype": str(x.dtype)}
                                for x in args
                                if isinstance(x, mx.array)
                            ],
                            "output_dtype": str(output.dtype)
                            if isinstance(output, mx.array)
                            else None,
                            "bits": kwargs.get("bits"),
                            "group_size": kwargs.get("group_size"),
                        }
                    )
                return output

            return call

        try:
            for owner, names in (
                (mx, ("quantized_matmul", "gather_qmm", "gather_mm", "dequantize")),
                (mx.fast, ("scaled_dot_product_attention", "rope", "rms_norm")),
            ):
                for name in names:
                    original = getattr(owner, name)
                    originals.append((owner, name, original))
                    setattr(owner, name, wrap(name, original))
            seed = bundle.tokenizer.encode("The capital of France is")
            tokens = (seed * ((16 + len(seed) - 1) // len(seed)))[:16]
            cache = bundle.model.make_cache()
            output = bundle.model(mx.array([tokens]), cache=cache)
            mx.eval(output.logits)
            for _ in range(2):
                token = mx.argmax(output.logits[:, -1], axis=-1)[:, None]
                output = bundle.model(token, cache=cache)
                mx.eval(output.logits)
            state = []
            for layer, item in enumerate(cache):
                if hasattr(item, "conv_state"):
                    arrays = (item.conv_state, item.recurrent_state)
                elif callable(getattr(item, "state", None)):
                    arrays = item.state()
                else:
                    arrays = (getattr(item, "values", None),)
                state.append(
                    {
                        "layer": layer,
                        "dtypes": [str(x.dtype) for x in arrays if isinstance(x, mx.array)],
                        "offset": item.offset,
                        "committed_state_bytes": sum(
                            x.nbytes for x in arrays if isinstance(x, mx.array)
                        ),
                        "allocated_state_bytes": getattr(item, "allocated_nbytes", None),
                    }
                )
            return {
                "status": "PASS",
                "scope": "16-token prefill and two native greedy decode steps",
                "checkpoint": bundle.model_id,
                "revision": bundle.revision,
                "precision": precision,
                "execution_dtype": execution,
                "weight_dtypes": tensor_dtypes(bundle.model.parameters()),
                "logits_dtype": str(output.logits.dtype),
                "state": state,
                "full16_vs_prefill15_plus_last_token": {
                    "stored": stored_split,
                    "profile": profile_split,
                },
                "primitive_call_counts": dict(counts),
                "offset_norm_isolation": norm_isolation,
                "calls": calls,
                "meaning": (
                    "Observed Python-to-MLX primitive calls, not Metal GPU kernel counts. "
                    "Weight conversion occurs before instrumentation. Operator @ calls "
                    "are not intercepted."
                ),
            }
        finally:
            for owner, name, original in reversed(originals):
                setattr(owner, name, original)

    try:
        return manager._pool.executor.call(bundle.model_id, compute)
    finally:
        manager.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--precision", choices=("stored", "affine_q4", "affine_q8"), default="stored"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    try:
        report = run(args.checkpoint, args.precision)
    except Exception as exc:
        report = {"status": "FAIL_API", "error": f"{type(exc).__name__}: {exc}"}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
